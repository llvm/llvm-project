dnl This macro checks for tclsh which is required to run dejagnu. On some 
dnl platforms (notably FreeBSD), tclsh is named tclshX.Y - this handles
dnl that for us so we can get the latest installed tclsh version.
dnl 
AC_DEFUN([DJ_AC_PATH_TCLSH], [
no_itcl=true
AC_MSG_CHECKING(for the tclsh program in tclinclude directory)
AC_ARG_WITH(tclinclude,
  AS_HELP_STRING([--with-tclinclude],
                [directory where tcl headers are]), 
  [with_tclinclude=${withval}],[with_tclinclude=''])
AC_CACHE_VAL(ac_cv_path_tclsh,[
dnl first check to see if --with-itclinclude was specified
if test x"${with_tclinclude}" != x ; then
  if test -f ${with_tclinclude}/tclsh ; then
    ac_cv_path_tclsh=`(cd ${with_tclinclude}; pwd)`
  elif test -f ${with_tclinclude}/src/tclsh ; then
    ac_cv_path_tclsh=`(cd ${with_tclinclude}/src; pwd)`
  else
    AC_MSG_ERROR([${with_tclinclude} directory doesn't contain tclsh])
  fi
fi

dnl see if one is installed
if test x"${ac_cv_path_tclsh}" = x ; then
  AC_MSG_RESULT(none)
  AC_PATH_PROGS([TCLSH],[tclsh8.4 tclsh8.4.8 tclsh8.4.7 tclsh8.4.6 tclsh8.4.5 tclsh8.4.4 tclsh8.4.3 tclsh8.4.2 tclsh8.4.1 tclsh8.4.0 tclsh8.3 tclsh8.3.5 tclsh8.3.4 tclsh8.3.3 tclsh8.3.2 tclsh8.3.1 tclsh8.3.0 tclsh])
  if test x"${TCLSH}" = x ; then
    ac_cv_path_tclsh='';
  else
    ac_cv_path_tclsh="${TCLSH}";
  fi
else
  AC_MSG_RESULT(${ac_cv_path_tclsh})
  TCLSH="${ac_cv_path_tclsh}"
  AC_SUBST(TCLSH)
fi
])])

