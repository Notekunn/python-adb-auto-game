from task import Task, TaskConfig


def main():
    task_config = TaskConfig()
    task_config.max_strength = 4 * 1000000
    task = Task('emulator-5554', task_config, False)
    task.start()


if __name__ == '__main__':
    main()
