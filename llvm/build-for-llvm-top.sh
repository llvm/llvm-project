#!/bin/sh

# This includes the Bourne shell library from llvm-top. Since this file is
# generally only used when building from llvm-top, it is safe to assume that
# llvm is checked out into llvm-top in which case .. just works.
. ../library.sh

# Process the options passed in to us by the build script into standard
# variables. 
process_arguments "$@"

# First, see if the build directory is there. If not, create it.
build_dir="$LLVM_TOP/build.llvm"
if test ! -d "$build_dir" ; then
  mkdir -p "$build_dir"
fi

# See if we have previously been configured by sensing the presence
# of the config.status scripts
config_status="$build_dir/config.status"
if test ! -f "$config_status" -o "$config_status" -ot "$0" ; then
  # We must configure so build a list of configure options
  config_options="--prefix=$PREFIX --with-llvmgccdir=$PREFIX"
  if test "$OPTIMIZED" -eq 1 ; then
    config_options="$config_options --enable-optimized"
  else
    config_options="$config_options --disable-optimized"
  fi
  if test "$DEBUG" -eq 1 ; then
    config_options="$config_options --enable-debug"
  else
    config_options="$config_options --disable-debug"
  fi
  if test "$ASSERTIONS" -eq 1 ; then
    config_options="$config_options --enable-assertions"
  else
    config_options="$config_options --disable-assertions"
  fi
  if test "$CHECKING" -eq 1 ; then
    config_options="$config_options --enable-expensive-checks"
  else
    config_options="$config_options --disable-expensive-checks"
  fi
  if test "$DOXYGEN" -eq 1 ; then
    config_options="$config_options --enable-doxygen"
  else
    config_options="$config_options --disable-doxygen"
  fi
  if test "$THREADS" -eq 1 ; then
    config_options="$config_options --enable-threads"
  else
    config_options="$config_options --disable-threads"
  fi
  config_options="$config_options $OPTIONS_DASH $OPTIONS_DASH_DASH"
  src_dir=`pwd`
  cd "$build_dir"
  msg 0 Configuring $module with:
  msg 0 "  $src_dir/configure" $config_options
  $src_dir/configure $config_options || \
    die $? "Configuring $module module failed"
else
  msg 0 Module $module already configured, ignoring configure options.
  cd "$build_dir"
fi

msg 0 Building $module with:
msg 0 "  make" $OPTIONS_ASSIGN tools-only
make $OPTIONS_ASSIGN tools-only
