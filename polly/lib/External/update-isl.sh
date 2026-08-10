#! /bin/sh
set -e

# Replace the content of the isl directory with a fresh clone from
# http://repo.or.cz/isl.git

SCRIPTPATH=`realpath --no-symlinks $(dirname $0)`
ISL_SOURCE_DIR="${SCRIPTPATH}/isl"
GITDIR=`mktemp -d --tmpdir isl-XXX`

# Checkout isl source code
git clone --recursive http://repo.or.cz/isl.git $GITDIR
if [ -n "$1" ]; then
  (cd $GITDIR && git checkout --detach $1)
  (cd $GITDIR && git submodule update --recursive)
fi

# Customize the source directory for Polly:
# - Remove the autotools build system to avoid including GPL source into
#   the LLVM repository, even if covered by the autotools exception
# - Create files that the autotools would have created
# - Save the custom isl C++ binding
# - Strip git source versioning
(cd $GITDIR && rm -rf m4 autogen.sh configure.ac)
(cd $GITDIR && find -name "Makefile.am" -execdir rm -f '{}' \;)
(cd $GITDIR && git describe > $GITDIR/GIT_HEAD_ID)
cp $ISL_SOURCE_DIR/include/isl/isl-noexceptions.h $GITDIR/include/isl/isl-noexceptions.h
rm -rf $GITDIR/.git
rm -rf $GITDIR/imath/.git

# Replace the current isl source
# IMPORTANT: Remember to `git add` any new files in LLVM's versioning
#            and add them to its CMakeLists.txt if necessary.
rm -rf $ISL_SOURCE_DIR
mv -T $GITDIR $ISL_SOURCE_DIR

# Cleanup script temporaries
rm -rf $TMPDIR
