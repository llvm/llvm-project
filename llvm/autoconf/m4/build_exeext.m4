# Check for the extension used for executables on build platform.
# This is necessary for cross-compiling where the build platform
# may differ from the host platform.
AC_DEFUN([AC_BUILD_EXEEXT],
[
AC_MSG_CHECKING([for executable suffix on build platform])
AC_CACHE_VAL(ac_cv_build_exeext,
[if test "$CYGWIN" = yes || test "$MINGW32" = yes; then
  ac_cv_build_exeext=.exe
else
  ac_build_prefix=${build_alias}-

  AC_CHECK_PROG(BUILD_CC, ${ac_build_prefix}gcc, ${ac_build_prefix}gcc)
  if test -z "$BUILD_CC"; then
     AC_CHECK_PROG(BUILD_CC, gcc, gcc)
     if test -z "$BUILD_CC"; then
       AC_CHECK_PROG(BUILD_CC, cc, cc, , , /usr/ucb/cc)
     fi
  fi
  test -z "$BUILD_CC" && AC_MSG_ERROR([no acceptable cc found in \$PATH])
  ac_build_link='${BUILD_CC-cc} -o conftest $CFLAGS $CPPFLAGS $LDFLAGS conftest.$ac_ext $LIBS 1>&AS_MESSAGE_LOG_FD'
  rm -f conftest*
  echo 'int main () { return 0; }' > conftest.$ac_ext
  ac_cv_build_exeext=
  if AC_TRY_EVAL(ac_build_link); then
    for file in conftest.*; do
      case $file in
      *.c | *.o | *.obj | *.dSYM) ;;
      *) ac_cv_build_exeext=`echo $file | sed -e s/conftest//` ;;
      esac
    done
  else
    AC_MSG_ERROR([installation or configuration problem: compiler cannot create executables.])
  fi
  rm -f conftest*
  test x"${ac_cv_build_exeext}" = x && ac_cv_build_exeext=blank
fi])
BUILD_EXEEXT=""
test x"${ac_cv_build_exeext}" != xblank && BUILD_EXEEXT=${ac_cv_build_exeext}
AC_MSG_RESULT(${ac_cv_build_exeext})
ac_build_exeext=$BUILD_EXEEXT
AC_SUBST(BUILD_EXEEXT)])
