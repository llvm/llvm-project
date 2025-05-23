#!/usr/bin/env bash

# Modified by Sunscreen under the AGPLv3 license; see the README at the
# repository root for more information

function bcat() {
  # Check if bat exists. If so, use this command. Otherwise use cat
  command -v bat > /dev/null && bat -pp "$@" || cat "$@"
}

function hexdump() {
  # Check if hexyl exists. If so, use this command. Otherwise use xxd
  command -v hexyl > /dev/null && hexyl "$@" || xxd "$@"
}

function echo_with_underline_color() {
  if [[ $QUIET == 1 ]]; then return; fi  # Check if QUIET mode is enabled

  case $1 in
    red) color=1 ;;
    green) color=2 ;;
    yellow) color=3 ;;
    blue) color=4 ;;
    magenta) color=5 ;;
    cyan) color=6 ;;
    white) color=7 ;;
    *) color=7 ;;
  esac

  text=$2
  textlength=${#text}
  underline=""
  for ((i=0; i<$textlength; i++)); do
    underline+="-"
  done

  echo -e "\033[0;3${color}m${text}\033[0m"
  echo -e "\033[0;3${color}m${underline}\033[0m"
}

function echo_header_with_underline_color() {
  if [[ $QUIET == 1 ]]; then return; fi  # Check if QUIET mode is enabled

  # Call echo_with_underline_color, but add blank lines before and after
  echo ""
  echo_with_underline_color "$1" "$2"
}

# Echo using orange
function echo_warning() {
  if [[ $QUIET == 1 ]]; then return; fi  # Check if QUIET mode is enabled

  echo -e "\033[0;33m$1\033[0m"
}

function echo_error() {
  echo -e "\033[0;31m$1\033[0m" >&2
}

function show_help() {
  echo -e "${GREEN}Usage:${RESET} $0 [OPTIONS]"
  echo -e "${CYAN}Options:${RESET}"
  echo -e "  ${YELLOW}--llvm-config${RESET}     Configure the LLVM compiler. Implies ${YELLOW}--llvm-build${RESET}"
  echo -e "  ${YELLOW}--llvm-build${RESET}      Build the LLVM compiler"
  echo -e "  ${YELLOW}--optimize OPT${RESET}    Set the optimization to OPT (e.g., ${YELLOW}--optimize${RESET} ${BLUE}2${RESET}"
  echo -e "                    is equivalent to -O2)"
  echo -e "  ${YELLOW}--selection OPT${RESET}   Set the selection backend to OPT. OPT can be:"
  echo -e "                    ${BLUE}globalisel${RESET} or ${BLUE}selectiondag${RESET}. Default is ${BLUE}selectiondag${RESET}"
  echo -e "  ${YELLOW}--view OPT${RESET}        Set the view option to OPT. OPT can include:"
  echo ""
  echo -e "                    Selection DAG views:"
  echo -e "                      ${BLUE}combine1, legalize, combine2, isel, sched, sunit${RESET}"
  echo ""
  echo -e "                    Text info (available on both backends):"
  echo -e "                      ${BLUE}debug, print-isel, print-all, print-pipeline-passes${RESET}"
  echo ""
  echo -e "                    GlobalISel views:"
  echo -e "                      ${BLUE}irtranslator, legalizer, regbankselect, instruction-select,${RESET}"
  echo -e "                      ${BLUE}finalize-isel, virtregrewriter${RESET}"
  echo ""
  echo -e "                    To choose multiple views, list them comma separated"
  echo -e "                    (e.g., '${BLUE}combine1${RESET},${BLUE}legalize${RESET}')."
  echo ""
  echo -e "                    '${BLUE}debug${RESET}' prints verbose information about every step"
  echo -e "                    of the compilation process, useful for debugging."
  echo -e "  ${YELLOW}--disassemble${RESET}     Disassemble the generated ELF file"
  echo -e "  ${YELLOW}--clean${RESET}           Clean up the generated files and exit"
  echo -e "  ${YELLOW}--quiet${RESET}           Suppress all output except for errors"
  echo -e "  ${YELLOW}--help${RESET}            Display this help message"

  exit 0
}

GIT_ROOT=$(git rev-parse --show-toplevel)
# If we move this script file to other repos, we will need to adjust this
LLVM_PROJECT_ROOT="$GIT_ROOT"

LLVM_PROJECT_BUILD="${LLVM_PROJECT_ROOT}/build"
LLVM_PROJECT_BIN="${LLVM_PROJECT_BUILD}/bin"

COMPILE_PARASOL_PROGRAM="${GIT_ROOT}/sample-programs/compile.sh"
LLVM_COMPILE="${GIT_ROOT}/compile-llvm.sh"

# Default options
OPT="-O2"
VIEW_GLOBALISEL=()
VIEW_SELECTIONDAG=()
VIEW_EITHER=()
LLVMCONFIG=0
LLVMBUILD=0
DISASSEMBLE=0
CLEAN=0
QUIET=0
SELECTION_BACKEND="selectiondag"

# Define color variables
GREEN="\033[1;32m"
CYAN="\033[1;36m"
YELLOW="\033[1;33m"
BLUE="\033[1;34m"
RESET="\033[0m"

declare -A VIEW_OPTIONS

# Populate the hashmap with options and their corresponding values
VIEW_OPTIONS=(
  ["combine1"]="-view-dag-combine1-dags"
  ["legalize"]="-view-legalize-dags"
  ["combine2"]="-view-dag-combine2-dags"
  ["isel"]="-view-isel-dags"
  ["sched"]="-view-sched-dags"
  ["sunit"]="-view-sunit-dags"
  ["debug"]="-debug"
  ["print-isel"]="--print-after-isel"
  ["print-all"]="--print-after-all"
  ["print-pipeline-passes"]="--print-pipeline-passes"
  ["irtranslator"]="-print-after=irtranslator"
  ["legalizer"]="-print-after=legalizer"
  ["regbankselect"]="-print-after=regbankselect"
  ["instruction-select"]="-print-after=instruction-select"
  ["finalize-isel"]="-print-after=finalize-isel"
  ["virtregrewriter"]="-print-after=virtregrewriter"
)

# Parse command-line options
while (( "$#" )); do
  case "$1" in
     --help)
      show_help
      ;;
    --quiet)
      QUIET=1
      shift
      ;;
    --llvm-config)
      LLVMCONFIG=1
      LLVMBUILD=1
      shift
      ;;
    --llvm-build)
      LLVMBUILD=1
      shift
      ;;
    --disassemble)
      DISASSEMBLE=1
      shift
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --optimize)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        OPT="-O$2"
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    --selection)
      if [ -n "$2" ] && { [ "$2" == "selectiondag" ] || [ "$2" == "globalisel" ]; }; then
        SELECTION_BACKEND="$2"
        shift 2
      else
        echo_error "Error: Argument for --selection must be 'selectiondag' or 'globalisel'"
        exit 1
      fi
      ;;
    --view)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        for option in $(echo $2 | tr "," "\n")
        do
          if [[ -v VIEW_OPTIONS[$option] ]]; then
            case "$option" in
              combine1|legalize|combine2|isel|sched|sunit)
                VIEW_SELECTIONDAG+=("${option}")
                ;;
              debug|print-isel|print-all|print-pipeline-passes)
                VIEW_EITHER+=("${option}")
                ;;
              irtranslator|legalizer|regbankselect|instruction-select|finalize-isel|virtregrewriter)
                VIEW_GLOBALISEL+=("$option")
                ;;
            esac
          else
            echo_error "Error: Unsupported view option $option"
            exit 1
          fi
        done
        shift 2
      else
        echo_error "Error: Argument for $1 is missing"
        exit 1
      fi
      ;;
    --)
      shift
      break
      ;;
    -*|--*=)
      echo_error "Error: Unsupported flag $1"
      exit 1
      ;;
    *)
      shift
      ;;
  esac
done

#######################################################################
# Validate SELECTION_BACKEND and combine VIEW arrays
#######################################################################

# Check for conflicting view options based on SELECTION_BACKEND
if [ "$SELECTION_BACKEND" = "selectiondag" ] && [ ${#VIEW_GLOBALISEL[@]} -ne 0 ]; then
  echo_error "Error: GlobalISel options provided when using --selection selectiondag. Bad options are: ${VIEW_GLOBALISEL[*]}"
  exit 10
elif [ "$SELECTION_BACKEND" = "globalisel" ] && [ ${#VIEW_SELECTIONDAG[@]} -ne 0 ]; then
  echo_error "Error: SelectionDAG options provided when using --selection globalisel. Bad options are: ${VIEW_SELECTIONDAG[*]}"
  exit 10
fi

# Combine VIEW arrays based on SELECTION_BACKEND
VIEW_ARRAY=()
if [ "$SELECTION_BACKEND" = "selectiondag" ]; then
  for key in "${VIEW_SELECTIONDAG[@]}" "${VIEW_EITHER[@]}"; do
    VIEW_ARRAY+=("${VIEW_OPTIONS[$key]}")
  done
elif [ "$SELECTION_BACKEND" = "globalisel" ]; then
  for key in "${VIEW_GLOBALISEL[@]}" "${VIEW_EITHER[@]}"; do
    VIEW_ARRAY+=("${VIEW_OPTIONS[$key]}")
  done
fi

# Convert VIEW_ARRAY to a string with elements separated by spaces
VIEW="${VIEW_ARRAY[*]}"


#######################################################################
# Find the C file in the directory
#######################################################################

# Find the name of the only .c file in this folder and strip off the .c ending
CFILE=$(ls *.c 2>/dev/null)
if [ -z "$CFILE" ]; then
  echo_error "Error: No .c file found in this folder"
  exit 1
fi

# Strip off the .c ending
CFILE=${CFILE%.c}


#######################################################################
# Clean and exit
#######################################################################

if [ $CLEAN -eq 1 ]; then
  echo_warning "Cleaning up generated files"
  rm -f "$CFILE".{S,ll,bc,o}

  exit 0
fi


#######################################################################
# Configure and build LLVM if it doesn't exist
#######################################################################

# Check if the build directory and clang binary exist
if [ ! -f "${LLVM_PROJECT_BUILD}/build.ninja" ]; then
  LLVMCONFIG=1
fi

if [ ! -f "${LLVM_PROJECT_BIN}/clang" ]; then
  LLVMBUILD=1
fi


#######################################################################
# Main script
#######################################################################

if [ $LLVMCONFIG -eq 1 ]; then
  echo_header_with_underline_color "yellow" "Configuring the LLVM compiler"

  "${LLVM_COMPILE}" configure

  if [ $? -ne 0 ]; then
    echo_error "Error: Failed to configure the compiler"
    exit 1
  fi

  echo_with_underline_color "green" "Configuring the LLVM compiler complete"
fi

if [ $LLVMBUILD -eq 1 ]; then
  echo_header_with_underline_color "yellow" "Building the LLVM compiler"

  "${LLVM_COMPILE}" build

  if [ $? -ne 0 ]; then
    echo_error "Error: Failed to build the compiler"
    exit 1
  fi

  echo_with_underline_color "green" "Building the LLVM compiler complete"
fi

echo_header_with_underline_color "yellow" "Compiling $CFILE.c with optimization $OPT"

"${COMPILE_PARASOL_PROGRAM}" "$CFILE" "$OPT" "$SELECTION_BACKEND" "$VIEW"

# If the command fails, then exit the script
if [ $? -ne 0 ]; then
  exit 1
fi

if [[ $QUIET == 1 ]]; then exit 0; fi

echo_header_with_underline_color "green" "Compilation of $CFILE.c complete with optimization $OPT"

echo_header_with_underline_color "magenta" "Printing out the LLVM intermediate ($CFILE.ll)"
bcat ./"$CFILE".ll

echo_header_with_underline_color "magenta" "Printing out Parasol ASM ($CFILE.S)"
bcat ./"$CFILE".S

echo_header_with_underline_color "magenta" "Printing out ELF information ($CFILE.o)"
"${LLVM_PROJECT_BIN}"/llvm-readelf "$CFILE".o --all

if [ $DISASSEMBLE -eq 1 ]; then
  echo_header_with_underline_color "magenta" "Printing out .text section of Parasol ELF disassembled ($CFILE.o)"
  "${LLVM_PROJECT_BIN}"/llvm-objdump -d "$CFILE".o
fi
