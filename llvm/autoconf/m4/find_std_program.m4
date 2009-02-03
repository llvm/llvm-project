dnl Check for a standard program that has a bin, include and lib directory
dnl 
dnl Parameters:
dnl   $1 - prefix directory to check
dnl   $2 - program name to check
dnl   $3 - header file to check 
dnl   $4 - library file to check 
AC_DEFUN([CHECK_STD_PROGRAM],
[m4_define([allcapsname],translit($2,a-z,A-Z))
if test -n "$1" -a -d "$1" -a -n "$2" -a -d "$1/bin" -a -x "$1/bin/$2" ; then
  AC_SUBST([USE_]allcapsname(),["USE_]allcapsname()[ = 1"])
  AC_SUBST(allcapsname(),[$1/bin/$2])
  AC_SUBST(allcapsname()[_BIN],[$1/bin])
  AC_SUBST(allcapsname()[_DIR],[$1])
  if test -n "$3" -a -d "$1/include" -a -f "$1/include/$3" ; then 
    AC_SUBST(allcapsname()[_INC],[$1/include])
  fi
  if test -n "$4" -a -d "$1/lib" -a -f "$1/lib/$4" ; then
    AC_SUBST(allcapsname()[_LIB],[$1/lib])
  fi
fi
])

dnl Find a program via --with options, in the path, or well known places
dnl
dnl Parameters:
dnl   $1 - program's executable name
dnl   $2 - header file name to check (optional)
dnl   $3 - library file name to check (optional)
dnl   $4 - alternate (long) name for the program
AC_DEFUN([FIND_STD_PROGRAM],
[m4_define([allcapsname],translit($1,a-z,A-Z))
m4_define([stdprog_long_name],ifelse($4,,translit($1,[ !@#$%^&*()-+={}[]:;"',./?],[-]),translit($4,[ !@#$%^&*()-+={}[]:;"',./?],[-])))
AC_MSG_CHECKING([for ]stdprog_long_name()[ bin/lib/include locations])
AC_ARG_WITH($1,
  AS_HELP_STRING([--with-]stdprog_long_name()[=DIR],
  [Specify that the ]stdprog_long_name()[ install prefix is DIR]),
  $1[pfxdir=$withval],$1[pfxdir=nada])
AC_ARG_WITH($1[-bin],
  AS_HELP_STRING([--with-]stdprog_long_name()[-bin=DIR],
  [Specify that the ]stdprog_long_name()[ binary is in DIR]),
    $1[bindir=$withval],$1[bindir=nada])
AC_ARG_WITH($1[-lib],
  AS_HELP_STRING([--with-]stdprog_long_name()[-lib=DIR],
  [Specify that ]stdprog_long_name()[ libraries are in DIR]),
  $1[libdir=$withval],$1[libdir=nada])
AC_ARG_WITH($1[-inc],
  AS_HELP_STRING([--with-]stdprog_long_name()[-inc=DIR],
  [Specify that the ]stdprog_long_name()[ includes are in DIR]),
  $1[incdir=$withval],$1[incdir=nada])
eval pfxval=\$\{$1pfxdir\}
eval binval=\$\{$1bindir\}
eval incval=\$\{$1incdir\}
eval libval=\$\{$1libdir\}
if test "${pfxval}" != "nada" ; then
  CHECK_STD_PROGRAM(${pfxval},$1,$2,$3)
elif test "${binval}" != "nada" ; then
  if test "${libval}" != "nada" ; then
    if test "${incval}" != "nada" ; then
      if test -d "${binval}" ; then
        if test -d "${incval}" ; then
          if test -d "${libval}" ; then
            AC_SUBST(allcapsname(),${binval}/$1)
            AC_SUBST(allcapsname()[_BIN],${binval})
            AC_SUBST(allcapsname()[_INC],${incval})
            AC_SUBST(allcapsname()[_LIB],${libval})
            AC_SUBST([USE_]allcapsname(),["USE_]allcapsname()[ = 1"])
            AC_MSG_RESULT([found via --with options])
          else
            AC_MSG_RESULT([failed])
            AC_MSG_ERROR([The --with-]$1[-libdir value must be a directory])
          fi
        else
          AC_MSG_RESULT([failed])
          AC_MSG_ERROR([The --with-]$1[-incdir value must be a directory])
        fi
      else
        AC_MSG_RESULT([failed])
        AC_MSG_ERROR([The --with-]$1[-bindir value must be a directory])
      fi
    else
      AC_MSG_RESULT([failed])
      AC_MSG_ERROR([The --with-]$1[-incdir option must be specified])
    fi
  else
    AC_MSG_RESULT([failed])
    AC_MSG_ERROR([The --with-]$1[-libdir option must be specified])
  fi
else
  tmppfxdir=`which $1 2>&1`
  if test -n "$tmppfxdir" -a -d "${tmppfxdir%*$1}" -a \
          -d "${tmppfxdir%*$1}/.." ; then
    tmppfxdir=`cd "${tmppfxdir%*$1}/.." ; pwd`
    CHECK_STD_PROGRAM($tmppfxdir,$1,$2,$3)
    AC_MSG_RESULT([found in PATH at ]$tmppfxdir)
  else
    checkresult="yes"
    eval checkval=\$\{"USE_"allcapsname()\}
    CHECK_STD_PROGRAM([/usr],$1,$2,$3)
    if test -z "${checkval}" ; then
      CHECK_STD_PROGRAM([/usr/local],$1,$2,$3)
      if test -z "${checkval}" ; then
        CHECK_STD_PROGRAM([/sw],$1,$2,$3)
        if test -z "${checkval}" ; then
          CHECK_STD_PROGRAM([/opt],$1,$2,$3)
          if test -z "${checkval}" ; then
            CHECK_STD_PROGRAM([/],$1,$2,$3)
            if test -z "${checkval}" ; then
              checkresult="no"
            fi
          fi
        fi
      fi
    fi
    AC_MSG_RESULT($checkresult)
  fi
fi
])
