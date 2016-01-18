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

projects="llvm cfe test-suite compiler-rt libcxx libcxxabi clang-tools-extra polly lldb lld openmp libunwind"
base_url="https://llvm.org/svn/llvm-project"

release=""
rc=""

function usage() {
    echo "Export the SVN sources and build tarballs from them"
    echo "usage: `basename $0`"
    echo " "
    echo "  -release <num> The version number of the release"
    echo "  -rc <num>      The release candidate number"
    echo "  -final         The final tag"
}

function export_sources() {
    release_no_dot=`echo $release | sed -e 's,\.,,g'`
    tag_dir="tags/RELEASE_$release_no_dot/$rc"

    if [ "$rc" = "final" ]; then
        rc=""
    fi

    for proj in $projects; do
        echo "Exporting $proj ..."
        svn export \
            $base_url/$proj/$tag_dir \
            $proj-$release$rc.src

        echo "Creating tarball ..."
        tar cfJ $proj-$release$rc.src.tar.xz $proj-$release$rc.src
    done
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
        -final | --final )
            rc="final"
            ;;
        -h | -help | --help )
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
    exit 1
fi

# Make sure umask is not overly restrictive.
umask 0022

export_sources
exit 0
