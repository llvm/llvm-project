#
# Get the linker version string.
#
# This macro is specific to LLVM.
#
AC_DEFUN([AC_LINK_GET_VERSION],
  [AC_CACHE_CHECK([for linker version],[llvm_cv_link_version],
  [
   version_string="$(ld -v 2>&1 | head -1)"

   # Check for ld64.
   if (echo "$version_string" | grep -q "ld64"); then
     llvm_cv_link_version=$(echo "$version_string" | sed -e "s#.*ld64-\([^ ]*\)\( (.*)\)\{0,1\}#\1#")
   else
     llvm_cv_link_version=$(echo "$version_string" | sed -e "s#[^0-9]*\([0-9.]*\).*#\1#")
   fi
  ])
  AC_DEFINE_UNQUOTED([HOST_LINK_VERSION],"$llvm_cv_link_version",
                     [Linker version detected at compile time.])
])

#
# Determine if the system can handle the -R option being passed to the linker.
#
# This macro is specific to LLVM.
#
AC_DEFUN([AC_LINK_USE_R],
[AC_CACHE_CHECK([for compiler -Wl,-R<path> option],[llvm_cv_link_use_r],
[ AC_LANG_PUSH([C])
  oldcflags="$CFLAGS"
  CFLAGS="$CFLAGS -Wl,-R."
  AC_LINK_IFELSE([AC_LANG_PROGRAM([[]],[[]])],
    [llvm_cv_link_use_r=yes],[llvm_cv_link_use_r=no])
  CFLAGS="$oldcflags"
  AC_LANG_POP([C])
])
if test "$llvm_cv_link_use_r" = yes ; then
  AC_DEFINE([HAVE_LINK_R],[1],[Define if you can use -Wl,-R. to pass -R. to the linker, in order to add the current directory to the dynamic linker search path.])
  fi
])

#
# Determine if the system can handle the -R option being passed to the linker.
#
# This macro is specific to LLVM.
#
AC_DEFUN([AC_LINK_EXPORT_DYNAMIC],
[AC_CACHE_CHECK([for compiler -Wl,-export-dynamic option],
                [llvm_cv_link_use_export_dynamic],
[ AC_LANG_PUSH([C])
  oldcflags="$CFLAGS"
  CFLAGS="$CFLAGS -Wl,-export-dynamic"
  AC_LINK_IFELSE([AC_LANG_PROGRAM([[]],[[]])],
    [llvm_cv_link_use_export_dynamic=yes],[llvm_cv_link_use_export_dynamic=no])
  CFLAGS="$oldcflags"
  AC_LANG_POP([C])
])
if test "$llvm_cv_link_use_export_dynamic" = yes ; then
  AC_DEFINE([HAVE_LINK_EXPORT_DYNAMIC],[1],[Define if you can use -Wl,-export-dynamic.])
  fi
])

#
# Determine if the system can handle the --version-script option being
# passed to the linker.
#
# This macro is specific to LLVM.
#
AC_DEFUN([AC_LINK_VERSION_SCRIPT],
[AC_CACHE_CHECK([for compiler -Wl,--version-script option],
                [llvm_cv_link_use_version_script],
[ AC_LANG_PUSH([C])
  oldcflags="$CFLAGS"

  # The following code is from the autoconf manual,
  # "11.13: Limitations of Usual Tools".
  # Create a temporary directory $tmp in $TMPDIR (default /tmp).
  # Use mktemp if possible; otherwise fall back on mkdir,
  # with $RANDOM to make collisions less likely.
  : ${TMPDIR=/tmp}
  {
    tmp=`
      (umask 077 && mktemp -d "$TMPDIR/fooXXXXXX") 2>/dev/null
    ` &&
    test -n "$tmp" && test -d "$tmp"
  } || {
    tmp=$TMPDIR/foo$$-$RANDOM
    (umask 077 && mkdir "$tmp")
  } || exit $?

  echo "{" > "$tmp/export.map"
  echo "  global: main;" >> "$tmp/export.map"
  echo "  local: *;" >> "$tmp/export.map"
  echo "};" >> "$tmp/export.map"

  CFLAGS="$CFLAGS -Wl,--version-script=$tmp/export.map"
  AC_LINK_IFELSE([AC_LANG_PROGRAM([[]],[[]])],
    [llvm_cv_link_use_version_script=yes],[llvm_cv_link_use_version_script=no])
  rm "$tmp/export.map"
  rmdir "$tmp"
  CFLAGS="$oldcflags"
  AC_LANG_POP([C])
])
if test "$llvm_cv_link_use_version_script" = yes ; then
  AC_SUBST(HAVE_LINK_VERSION_SCRIPT,1)
  fi
])

