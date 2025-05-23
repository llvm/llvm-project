#!/usr/bin/env bash

# Modified by Sunscreen under the AGPLv3 license; see the README at the
# repository root for more information

GIT_ROOT=$(git rev-parse --show-toplevel)
# If we move this script file to other repos, we will need to adjust this
LLVM_COMPILE="${GIT_ROOT}/compile-llvm.sh"
LLVM_PROJECT_ROOT="$GIT_ROOT"

# Define colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN="\033[1;36m"
YELLOW='\033[0;33m'
BLUE="\033[1;34m"
RESET='\033[0m' # No Color

QUIET=1 # Global flag for quiet mode
SELECTION_OPTION="selectiondag" # Default selection option

# Function to display help
show_help() {
    echo -e "${GREEN}Usage: $0 [command] [options]${RESET}"
    echo -e "${CYAN}Commands:${RESET}"
    echo -e "${YELLOW}build${RESET} - Builds all programs using ./run.sh --llvm-build in each folder"
    echo -e "${YELLOW}clean${RESET} - Cleans all builds using ./run.sh --clean in each folder"
    echo -e ""
    echo -e "${CYAN}Options:${RESET}"
    echo -e "${YELLOW}--verbose${RESET}   - Run commands with verbose output"
    echo -e "${YELLOW}--selection${RESET} - Set the selection backend to '${BLUE}selectiondag${RESET}' or '${BLUE}globalisel${RESET}'."
    echo -e "              Only applies to the ${YELLOW}build${RESET} command. Default is '${BLUE}selectiondag${RESET}'."
    echo -e "${YELLOW}--help${RESET}      - Displays this help message"
    exit 0
}

# Function to perform action based on the argument
perform_action() {
    if [[ $1 == "build" ]]; then
        if [[ ! -d "${LLVM_PROJECT_ROOT}/build" ]]; then
            if ! ${LLVM_COMPILE} configure; then
                echo_error "Error: LLVM configuration failed"
                exit 1
            fi
        fi
        ${LLVM_COMPILE} build
    fi

    for dir in */ ; do
        if [[ -f "${dir}run.sh" ]]; then  # Check if run.sh exists in the directory
            cd "$dir" || exit
            if [[ $1 == "build" ]]; then
                echo -e "${GREEN}Building in $dir${RESET}"
                if [[ $QUIET -eq 1 ]]; then
                    ./run.sh --quiet --selection "${SELECTION_OPTION}"
                else
                    ./run.sh --selection "${SELECTION_OPTION}"
                fi
            elif [[ $1 == "clean" ]]; then
                echo -e "${RED}Cleaning in $dir${RESET}"
                if [[ $QUIET -eq 1 ]]; then
                    ./run.sh --clean --quiet
                else
                    ./run.sh --clean
                fi
            fi
            cd ..
        fi
    done
}

# Parse command-line options
while (( "$#" )); do
  case "$1" in
    --help)
      show_help
      ;;
    --verbose)
      QUIET=0
      shift
      ;;
    --selection)
      if [ -n "$2" ] && ([ "$2" == "selectiondag" ] || [ "$2" == "globalisel" ]); then
        SELECTION_OPTION="$2"
        shift 2
      else
        echo -e "${RED}Error: Argument for --selection must be 'selectiondag' or 'globalisel'${RESET}"
        exit 1
      fi
      ;;
    *)
      COMMAND="$1"
      shift
      ;;
  esac
done

# Perform the action based on the command
if [[ "$COMMAND" == "build" || "$COMMAND" == "clean" ]]; then
    perform_action "$COMMAND"
else
    show_help
fi
