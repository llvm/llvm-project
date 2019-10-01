#!/bin/bash
#===-- tag.sh - Tag the LLVM release candidates ----------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
# All the projects that make it into the monorepo, plus test-suite.
projects="monorepo-root cfe clang-tools-extra compiler-rt debuginfo-tests libclc libcxx libcxxabi libunwind lld lldb llgo llvm openmp parallel-libs polly pstl test-suite"
dryrun=""
revision="HEAD"

base_url="https://llvm.org/svn/llvm-project"

usage() {
    echo "usage: `basename $0` -release <num> [-rebranch] [-revision <num>] [-dry-run]"
    echo "usage: `basename $0` -release <num> -rc <num> [-dry-run]"
    echo " "
    echo "  -release <num>   The version number of the release"
    echo "  -rc <num>        The release candidate number"
    echo "  -rebranch        Remove existing branch, if present, before branching"
    echo "  -final           Tag final release candidate"
    echo "  -revision <num>  Revision to branch off (default: HEAD)"
    echo "  -dry-run         Make no changes to the repository, just print the commands"
}

tag_version() {
    local remove_args=()
    local create_args=()
    local message_prefix
    set -x
    for proj in $projects; do
        if svn ls $base_url/$proj/branches/release_$branch_release > /dev/null 2>&1 ; then
            if [ $rebranch = "no" ]; then
                continue
            fi
            remove_args+=(rm "$proj/branches/release_$branch_release")
        fi
        create_args+=(cp ${revision} "$proj/trunk" "$proj/branches/release_$branch_release")
    done
    if [[ ${#remove_args[@]} -gt 0 ]]; then
        message_prefix="Removing and recreating"
    else
        message_prefix="Creating"
    fi
    if [[ ${#create_args[@]} -gt 0 ]]; then
        ${dryrun} svnmucc --root-url "$base_url" \
            -m "$message_prefix release_$branch_release branch off revision ${revision}" \
            "${remove_args[@]}" "${create_args[@]}"
    fi
    set +x
}

tag_release_candidate() {
    local create_args=()
    set -x
    for proj in $projects ; do
        if ! svn ls $base_url/$proj/tags/RELEASE_$tag_release > /dev/null 2>&1 ; then
            create_args+=(mkdir "$proj/tags/RELEASE_$tag_release")
        fi
        if ! svn ls $base_url/$proj/tags/RELEASE_$tag_release/$rc > /dev/null 2>&1 ; then
            create_args+=(cp HEAD
                          "$proj/branches/release_$branch_release"
                          "$proj/tags/RELEASE_$tag_release/$rc")
        fi
    done
    if [[ ${#create_args[@]} -gt 0 ]]; then
        ${dryrun} svnmucc --root-url "$base_url" \
            -m "Creating release candidate $rc from release_$tag_release branch" \
            "${create_args[@]}"
    fi
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
        -revision | --revision )
            shift
            revision="$1"
            ;;
        -dry-run | --dry-run )
            dryrun="echo"
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

if [ "$release" = "" ]; then
    echo "error: need to specify a release version"
    echo
    usage
    exit 1
fi

branch_release=`echo $release | sed -e 's,\([0-9]*\.[0-9]*\).*,\1,' | sed -e 's,\.,,g'`
tag_release=`echo $release | sed -e 's,\.,,g'`

if [ "$rc" = "" ]; then
    tag_version
else
    if [ "$revision" != "HEAD" ]; then
        echo "error: cannot use -revision with -rc"
        echo
        usage
        exit 1
    fi

    tag_release_candidate
fi

exit 0
