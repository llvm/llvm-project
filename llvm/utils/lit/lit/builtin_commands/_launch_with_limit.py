import sys
import subprocess
import resource
import os

ULIMIT_ENV_VAR_PREFIX = "LIT_INTERNAL_ULIMIT_"


def main(argv):
    command_args = argv[1:]
    for env_var in os.environ:
        if env_var.startswith(ULIMIT_ENV_VAR_PREFIX):
            limit_str = env_var[len(ULIMIT_ENV_VAR_PREFIX) :]
            limit_value = int(os.environ[env_var])
            limit = (limit_value, limit_value)
            if limit_str == "RLIMIT_AS":
                resource.setrlimit(resource.RLIMIT_AS, limit)
            elif limit_str == "RLIMIT_NOFILE":
                resource.setrlimit(resource.RLIMIT_NOFILE, limit)
    process_output = subprocess.run(command_args)
    sys.exit(process_output.returncode)


if __name__ == "__main__":
    main(sys.argv)
