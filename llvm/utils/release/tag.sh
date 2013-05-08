#!/bin/sh
#===-- tag.sh - Tag the LLVM release candidates ----------------------------===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License.
#
#===------------------------------------------------------------------------===#
#
# Create branches and release candidates for the LLVM release.
#
#===------------------------------------------------------------------------===#

set -e

release=""
rc=""
rebranch="no"

base_url="https://llvm.org/svn/llvm-project"

function usage() {
    echo "usage: `basename $0` -release <num> [-rebranch]"
    echo "usage: `basename $0` -release <num> -rc <num>"
    echo " "
    echo "  -release <num>  The version number of the release"
    echo "  -rc <num>       The release candidate number"
    echo "  -rebranch       Remove existing branch, if present, before branching"
    echo "  -final          Tag final release candidate"
}

function tag_version() {
    set -x
    for proj in llvm cfe dragonegg test-suite compiler-rt libcxx clang-tools-extra ; do
        if svn ls $base_url/$proj/branches/release_$release > /dev/null 2>&1 ; then
            if [ $rebranch = "no" ]; then
                continue
            fi
            svn remove -m "Removing old release_$release branch for rebranching." \
                $base_url/$proj/branches/release_$release
        fi
        svn copy -m "Creating release_$release branch" \
            $base_url/$proj/trunk \
            $base_url/$proj/branches/release_$release
    done
    set +x
}

function tag_release_candidate() {
    set -x
    for proj in llvm cfe dragonegg test-suite compiler-rt libcxx clang-tools-extra ; do
        if ! svn ls $base_url/$proj/tags/RELEASE_$release > /dev/null 2>&1 ; then
            svn mkdir -m "Creating release directory for release_$release." $base_url/$proj/tags/RELEASE_$release
        fi
        if ! svn ls $base_url/$proj/tags/RELEASE_$release/$rc > /dev/null 2>&1 ; then
            svn copy -m "Creating release candidate $rc from release_$release branch" \
                $base_url/$proj/branches/release_$release \
                $base_url/$proj/tags/RELEASE_$release/$rc
        fi
    done
    set +x
}

while [ $# -gt 0 ]; do
    case $1 in
        -release | --release )
            shift
            release=$1
            ;;
        -rc | --rc )
            shift
            rc="rc$1"
            ;;
        -rebranch | --rebranch )
            rebranch="yes"
            ;;
        -final | --final )
            rc="final"
            ;;
        -h | --help | -help )
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

if [ "x$release" = "x" ]; then
    echo "error: need to specify a release version"
    echo
    usage
    exit 1
fi

release=`echo $release | sed -e 's,\.,,g'`

if [ "x$rc" = "x" ]; then
    tag_version
else
    tag_release_candidate
fi

exit 1
