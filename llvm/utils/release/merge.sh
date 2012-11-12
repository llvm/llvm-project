#!/bin/sh
#===-- merge.sh - Test the LLVM release candidates -------------------------===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License.
#
#===------------------------------------------------------------------------===#
#
# Merge a revision into a project.
#
#===------------------------------------------------------------------------===#

set -e

rev=""
proj=""

function usage() {
    echo "usage: `basename $0` [OPTIONS]"
    echo "  -proj PROJECT  The project to merge the result into"
    echo "  -rev NUM       The revision to merge into the project"
}

while [ $# -gt 0 ]; do
    case $1 in
        -rev | --rev | -r )
            shift
            rev=$1
            ;;
        -proj | --proj | -project | --project | -p )
            shift
            proj=$1
            ;;
        -h | -help | --help )
            usage
            ;;
        * )
            echo "unknown option: $1"
            echo ""
            usage
            exit 1
            ;;
    esac
    shift
done

if [ "x$rev" = "x" -o "x$proj" = "x" ]; then
    echo "error: need to specify project and revision"
    echo
    usage
    exit 1
fi

if ! svn ls http://llvm.org/svn/llvm-project/$proj/trunk > /dev/null 2>&1 ; then
    echo "error: invalid project: $proj"
    exit 1
fi

tempfile=`mktemp /tmp/merge.XXXXXX` || exit 1

echo "Merging r$rev:" > $tempfile
svn log -c $rev http://llvm.org/svn/llvm-project/$proj/trunk >> $tempfile 2>&1

cd $proj.src
echo "# Updating tree"
svn up
echo "# Merging r$rev into $proj"
svn merge -c $rev https://llvm.org/svn/llvm-project/$proj/trunk . || exit 1
echo "# Committing changes"
svn commit -F $tempfile || exit 1
rm -f $tempfile
exit 0
