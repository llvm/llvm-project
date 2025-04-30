#!/bin/sh -eu

REF_BASE=origin/next_release_170
if [ "$#" -ge 1 ]; then
    REF_BASE="$1"
fi
if ! git cat-file -t "$REF_BASE" >/dev/null; then
    echo "error: ref base not found!" 1>&2
    exit 1
fi

MERGE_BASE=$(git merge-base HEAD "$REF_BASE")
if git diff --quiet HEAD "$MERGE_BASE" ; then
    # No changes from merge base, no need to check
    exit 0
fi

_FOUND=false
for suffix in "-17" "-16" "-12" "-11" ""; do
    _CLANG_FORMAT="clang-format${suffix}"
    _GIT_CLANG_FORMAT="git-${_CLANG_FORMAT}"
    if command -v "${_CLANG_FORMAT}" >/dev/null; then
        echo "info: Using ${_CLANG_FORMAT}" 1>&2
        _FOUND=true
        break
    fi
done

if [ ${_FOUND} != true ]; then
    echo "error: clang-format not found!" 1>&2
    exit 1
fi

rm clang-format.diff 2>/dev/null || true

"${_GIT_CLANG_FORMAT}" --binary "${_CLANG_FORMAT}" \
    --extensions 'c,h,cc,hh,cpp,hpp,cxx,hxx,c++,h++,td' \
    --diff "$MERGE_BASE" \
    HEAD > clang-format.diff

if [ -s clang-format.diff ] && ! grep -q "no modified files to format" clang-format.diff ; then
    cat clang-format.diff 1>&2
    exit 1
fi
