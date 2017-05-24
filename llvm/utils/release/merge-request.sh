# !/bin/bash
#===-- merge-request.sh  ---------------------------------------------------===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License.
#
#===------------------------------------------------------------------------===#
#
# Submit a merge request to bugzilla.
#
#===------------------------------------------------------------------------===#

dryrun=""
stable_version=""
revision=""
BUGZILLA_BIN=""
BUGZILLA_CMD=""
release_metabug=""
bugzilla_product="new-bugs"
bugzilla_component="new bugs"
bugzilla_assigned_to=""
bugzilla_user=""
bugzilla_version=""
bugzilla_url="https://bugs.llvm.org/xmlrpc.cgi"

function usage() {
  echo "usage: `basename $0` -user EMAIL -stable-version X.Y -r NUM"
  echo ""
  echo " -user EMAIL             Your email address for logging into bugzilla."
  echo " -stable-version X.Y     The stable release version (e.g. 4.0, 5.0)."
  echo " -r NUM                  Revision number to merge (e.g. 1234567)."
  echo " -bugzilla-bin PATH      Path to bugzilla binary (optional)."
  echo " -assign-to EMAIL        Assign bug to user with EMAIL (optional)."
  echo " -dry-run                Print commands instead of executing them."
}

while [ $# -gt 0 ]; do
  case $1 in
    -user)
      shift
      bugzilla_user="$1"
      ;;
    -stable-version)
      shift
      stable_version="$1"
      ;;
    -r)
      shift
      revision="$1"
      ;;
    -project)
      shift
      project="$1"
      ;;
    -component)
      shift
      bugzilla_component="$1"
      ;;
    -bugzilla-bin)
      shift
      BUGZILLA_BIN="$1"
      ;;
    -assign-to)
      shift
      bugzilla_assigned_to="--assigned_to=$1"
      ;;
    -dry-run)
      dryrun="echo"
      ;;
    -help | --help | -h | --h | -\? )
      usage
      exit 0
      ;;
    * )
      echo "unknown option: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

if [ -z "$stable_version" ]; then
  echo "error: no stable version specified"
  exit 1
fi

case $stable_version in
  4.0)
    release_metabug="32061"
    ;;
  *)
    echo "error: invalid stable version"
    exit 1
esac
bugzilla_version=$stable_version

if [ -z "$revision" ]; then
  echo "error: revision not specified"
  exit 1
fi

if [ -z "$bugzilla_user" ]; then
  echo "error: bugzilla username not specified."
  exit 1
fi

if [ -z "$BUGZILLA_BIN" ]; then
  BUGZILLA_BIN=`which bugzilla`
  if [ $? -ne 0 ]; then
    echo "error: could not find bugzilla executable."
    echo "Make sure the bugzilla cli tool is installed on your system: "
    echo "pip install python-bugzilla (recommended)"
    echo ""
    echo "Fedora: dnf install python-bugzilla"
    echo "Ubuntu/Debian: apt-get install bugzilla-cli"
    exit 1
  fi
fi

BUGZILLA_MAJOR_VERSION=`$BUGZILLA_BIN --version 2>&1 | cut -d . -f 1`

if [ $BUGZILLA_MAJOR_VERSION -eq 1 ]; then

  echo "***************************** Warning *******************************"
  echo "You are using an older version of the bugzilla cli tool.  You will be "
  echo "able to create bugs, but this script will crash with the following "
  echo "error when trying to read back information about the bug you created:"
  echo ""
  echo "KeyError: 'internals'"
  echo ""
  echo "To avoid this error, use version 2.0.0 or higher"
  echo "https://pypi.python.org/pypi/python-bugzilla"
  echo "*********************************************************************"
fi

BUGZILLA_CMD="$BUGZILLA_BIN --bugzilla=$bugzilla_url"

bug_url="https://reviews.llvm.org/rL$revision"

echo "Checking for duplicate bugs..."

check_duplicates=`$BUGZILLA_CMD query --url $bug_url`

if [ -n "$check_duplicates" ]; then
  echo "Duplicate bug found:"
  echo $check_duplicates
  exit 1
fi

echo "Done"

# Get short commit summary
commit_summary=''
commit_msg=`svn log -r $revision https://llvm.org/svn/llvm-project/`
if [ $? -ne 0 ]; then
  echo "warning: failed to get commit message."
  commit_msg=""
fi

if [ -n "$commit_msg" ]; then
  commit_summary=`echo "$commit_msg" | sed '4q;d' | cut -c1-80`
  commit_summary=" : ${commit_summary}"
fi

bug_summary="Merge r$revision into the $stable_version branch${commit_summary}"

if [ -z "$dryrun" ]; then
  set -x
fi

${dryrun} $BUGZILLA_CMD --login --user=$bugzilla_user new \
  -p "$bugzilla_product" \
  -c "$bugzilla_component" -u $bug_url --blocked=$release_metabug \
  -o All --priority=P --arch All -v $bugzilla_version \
  --summary "${bug_summary}" \
  -l "Is this patch OK to merge to the $stable_version branch?" \
  $bugzilla_assigned_to \
  --oneline

set +x

if [ -n "$dryrun" ]; then
  exit 0
fi

if [ $BUGZILLA_MAJOR_VERSION -eq 1 ]; then
  success=`$BUGZILLA_CMD query --url $bug_url`
  if [ -z "$success" ]; then
    echo "Failed to create bug."
    exit 1
  fi

  echo " Created new bug:"
  echo $success
fi
