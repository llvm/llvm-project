import os
import subprocess
import sys
import datetime
import logging

# Get the absolute path of the script
script_path = os.path.abspath(os.path.dirname(__file__))

# Set the LLVM repository URL
llvm_repo_url = 'https://github.com/llvm/llvm-project.git'
log_filename = 'llvm_rebase_log.txt'
logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def run_command(command):
    logging.info(f"Running command: {command}")

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()

    # Print the command output
    logging.debug(f"Command output: {output.decode()}")

    return output.decode(), error.decode(), process.returncode

def generate_branch_name(commit_hash):
    return f'llvm_main_track_upstream_{commit_hash}'

def find_commit_id(commit_message, author):
    command = f"git log --grep='{commit_message}' --author='{author}' --pretty=format:'%H' -n 1"
    logging.info(f"Running command: {command}")

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    # Print the command output
    logging.debug(f"Command output: {output.decode()}")

    commit_id = output.decode().strip()
    return commit_id, error.decode(), process.returncode

def find_commit_before(commit_hash):
    output = run_command('git rev-list --parents -n 1 {}'.format(commit_hash))[0].strip()
    logging.debug(f'Output: {output}')
    commits = output.split()
    logging.debug(f'Commits: {commits}')
    return commits[1] if len(commits) >= 2 else None

def find_commit_weeks_ahead(commit_hash, weeks=2):
    commit_date = run_command('git show -s --format=%ci {}'.format(commit_hash))[0].strip()
    commit_date = datetime.datetime.strptime(commit_date, '%Y-%m-%d %H:%M:%S %z')
    weeks_ahead = commit_date + datetime.timedelta(weeks=weeks)
    logging.info(f'One week ahead: {weeks_ahead}')
    output = run_command('git rev-list -n 1 --before="{}" llvm/main'.format(weeks_ahead, commit_hash))[0].strip()
    return output if output else None

def build_clang():
    llvm_dir_name = 'llvm'
    build_dir = os.path.join(script_path, 'Release')  # replace with the actual build directory
    username = os.getlogin()  # Get the current username
    next_home = f'/space2/users/{username}/next_home'
    llvm_build_dir = os.path.join(build_dir, llvm_dir_name)
    llvm_install_prefix = os.path.join(next_home, llvm_dir_name)

    cfg = 'Release'  # or 'Debug' depending on your configuration
    enable_asserts = 'OFF'  # or 'ON' depending on your configuration
    custom_cmake_flags = []  # add any additional flags here
    build_target = 'all'  # replace with your desired build target
    nproc = os.cpu_count()

    os.makedirs(llvm_build_dir, exist_ok=True)
    os.chdir(llvm_build_dir)

    cmake_command = [
        'cmake', '-G', 'Ninja', os.path.join(script_path, 'llvm'),
        '-C', os.path.join(script_path, 'nextsilicon/LLVMBuildSettings.cmake'),
        '-DCMAKE_BUILD_TYPE=' + cfg,
        '-DCMAKE_INSTALL_PREFIX=' + llvm_install_prefix,
        '-DLLVM_INSTALL_UTILS=true',
        '-DLLVM_ENABLE_ASSERTIONS=' + enable_asserts,
        '-DPython3_EXECUTABLE=' + subprocess.check_output(['which', 'python3']).decode().strip(),
    ]

    cmake_command.extend(custom_cmake_flags)

    subprocess.run(cmake_command)

    ninja_command = ['ninja', '-j', str(nproc), build_target]
    run_command(ninja_command)

    os.chdir('..')

def main():
    if len(sys.argv) > 2:
        print("Usage: ./llvm_rebase_onto.py [num_of_weeks]")
        return

    num_of_weeks = int(sys.argv[1]) if len(sys.argv) == 2 else 2
    if num_of_weeks not in range(1, 9):
        print("Warning: Number of weeks must be in the range 1 to 8.")
        return

    logging.info('LLVM Rebase Onto script')
    print(f'Please check detailed log file "{log_filename}" for detailed information.')
    # llvm main branch
    llvm_main_branch = 'llvm_main'
    # Set the branch name
    base_branch = 'llvm_main_track_upstream'
    commit_message = "Add next32 machine"
    author = "Ilan Tayari <ilan@nextsilicon.com>"
    first_nextsilicon_commit_hash, error, return_code = find_commit_id(commit_message, author)
    if return_code == 0 and first_nextsilicon_commit_hash:
        logging.info(f"First NextSilicon commit hash: {first_nextsilicon_commit_hash}")
    elif return_code != 0:
        logging.error(f"Error: {error}.")
        return
    else:
        logging.error(f"Error: Unable to find the first NextSilicon commit hash.")
        return

    # Pull the latest changes
    run_command('git checkout {}'.format(base_branch))
    run_command('git pull origin {}'.format(base_branch))

    # Add the LLVM remote
    run_command('git remote add llvm {}'.format(llvm_repo_url))

    # Fetch changes from LLVM remote
    run_command('git fetch llvm')

    # Find the parent commit hash dynamically
    commit_before_specified_commit = find_commit_before(first_nextsilicon_commit_hash)

    if not commit_before_specified_commit:
        logging.error(f'Error: Unable to find the parent commit of {first_nextsilicon_commit_hash}.')
        return
    else:
        # Generate a backup branch name dynamically based on the current base commit
        backup_branch = generate_branch_name(commit_before_specified_commit)

        # Create a backup branch from llvm_main_track_upstream
        run_command('git checkout -b {}'.format(backup_branch))

        # Push a backup branch to the remote
        run_command('git push --set-upstream origin {}'.format(backup_branch))

        # Find a commit around one week in the future after commit_before_specified_commit
        commit_weeks_ahead = find_commit_weeks_ahead(commit_before_specified_commit, num_of_weeks)

        if not commit_weeks_ahead:
            logging.error('Error: Unable to find a commit around one week in the future.')
            return
        else:
            # Move llvm main branch to the newer commit
            run_command('git checkout {}'.format(llvm_main_branch))

            # Rebase --onto to move the base commit to the newer commit
            output, error, return_code = run_command('git rebase --onto {} {}'.format(commit_weeks_ahead, commit_before_specified_commit))

            if return_code == 0:
                logging.info(f"Rebase llvm_main successful. {output}")
                # Push a rebased llvm main branch to the remote for the clang-format
                run_command('git push origin {}'.format(llvm_main_branch))
            elif return_code == 1:
                logging.debug("Conflicts occurred during rebase on llvm_main. Please resolve the conflicts manually.")
                return
            else:
                logging.error(f"Rebase failed with error: {error}")
                return

            # Return to the base branch
            run_command('git checkout {}'.format(base_branch))
            # Generate a new branch name dynamically based on the new base commit
            new_base_branch = generate_branch_name(commit_weeks_ahead)

            # Create a new base branch from llvm_main_track_upstream
            run_command('git checkout -b {}'.format(new_base_branch))

            # Rebase --onto to move the base commit to the newer commit
            output, error, return_code = run_command('git rebase --onto {} {}'.format(commit_weeks_ahead, commit_before_specified_commit))

            if return_code == 0:
                logging.info(f"Rebase successful. {output}")
                # Count the number of commits moved
                commit_count = run_command('git rev-list --count {}..{}'.format(commit_before_specified_commit, commit_weeks_ahead))[0].strip()

                logging.info(f'Moved {commit_count} commits.')
            elif return_code == 1:
                logging.debug("Conflicts occurred during rebase. Please resolve the conflicts manually.")
                return
            else:
                logging.error(f"Rebase failed with error: {error}")
                return

            # Build LLVM and Clang
            build_clang()

            # Remove the LLVM remote after finishing
            run_command('git remote remove llvm')

            # Print information to the user.
            print(f'Please push the new base branch {new_base_branch} to the remote and create PR, follow jenkins job for that PR.')
            print(f'If the Jenkins job continuous-integration/jenkins/pr-head is successful, you can proceed with the another check.')
            print(f'Please run the Jenkins job "Manual-end-to-end-smoke-test" with Build Parameter: TOOLCHAIN_TAG should be PR-#.')
            print(f'If the Jenkins job "Manual-end-to-end-smoke-test" is successful, you can proceed with the following steps:')
            print(f'1. Delete the {base_branch} branch on remote and local git.')
            print(f'git push origin --delete {base_branch}')
            print(f'git branch --unset-upstream {base_branch}')
            print(f'git branch -D {base_branch}')
            print(f'2. Rename {new_base_branch} to {base_branch} and push it.')
            print(f'git checkout {new_base_branch}')
            print(f'git branch -m {new_base_branch} {base_branch}')
            print(f'3. Make sure that {new_base_branch} is deleted, as it will be used as the backup branch in the next run.')
            print(f'git push origin --delete {new_base_branch}')
            print(f'4. Push updated {base_branch} to the remote.')
            print(f'git push origin -u {base_branch}')
            print(f'5. Always refer to the detailed log file "{log_filename}" for more information.')

if __name__ == "__main__":
    main()
