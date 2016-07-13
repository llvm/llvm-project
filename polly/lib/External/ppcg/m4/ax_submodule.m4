AC_DEFUN([_AX_SUBMODULE],
[

m4_if(m4_bregexp($3,|,choice),choice,
	[AC_ARG_WITH($2,
		[AS_HELP_STRING([--with-$1=$3],
				[Which $1 to use [default=$4]])])])
case "system" in
$3)
	AC_ARG_WITH($2_prefix,
		    [AS_HELP_STRING([--with-$1-prefix=DIR],
				    [Prefix of $1 installation])])
	AC_ARG_WITH($2_exec_prefix,
		    [AS_HELP_STRING([--with-$1-exec-prefix=DIR],
				    [Exec prefix of $1 installation])])
esac
m4_if(m4_bregexp($3,build,build),build,
	[AC_ARG_WITH($2_builddir,
		[AS_HELP_STRING([--with-$1-builddir=DIR],
				[Location of $1 builddir])])])
if test "x$with_$2_prefix" != "x" -a "x$with_$2_exec_prefix" = "x"; then
	with_$2_exec_prefix=$with_$2_prefix
fi
if test "x$with_$2_prefix" != "x" -o "x$with_$2_exec_prefix" != "x"; then
	if test "x$with_$2" != "x" -a "x$with_$2" != "xsystem"; then
		AC_MSG_ERROR([Setting $with_$2_prefix implies use of system $1])
	fi
	with_$2="system"
fi
if test "x$with_$2_builddir" != "x"; then
	if test "x$with_$2" != "x" -a "x$with_$2" != "xbuild"; then
		AC_MSG_ERROR([Setting $with_$2_builddir implies use of build $1])
	fi
	with_$2="build"
	$2_srcdir=`echo @abs_srcdir@ | $with_$2_builddir/config.status --file=-`
	AC_MSG_NOTICE($1 sources in $$2_srcdir)
fi
if test "x$with_$2_exec_prefix" != "x"; then
	export PKG_CONFIG_PATH="$with_$2_exec_prefix/lib/pkgconfig${PKG_CONFIG_PATH+:$PKG_CONFIG_PATH}"
fi
case "$with_$2" in
$3)
	;;
*)
	case "$4" in
	bundled)
		if test -d $srcdir/.git -a \
			-d $srcdir/$1 -a \
			"`cd $srcdir; git submodule status $1 | cut -c1`" = '-'; then
			AC_MSG_WARN([git repo detected, but submodule $1 not initialized])
			AC_MSG_WARN([You may want to run])
			AC_MSG_WARN([	git submodule init])
			AC_MSG_WARN([	git submodule update])
			AC_MSG_WARN([	sh autogen.sh])
		fi
		if test -f $srcdir/$1/configure; then
			with_$2="bundled"
		else
			case "system" in
			$3)
				with_$2="system"
				;;
			*)
				with_$2="no"
				;;
			esac
		fi
		;;
	*)
		with_$2="$4"
		;;
	esac
	;;
esac
AC_MSG_CHECKING([which $1 to use])
AC_MSG_RESULT($with_$2)

])

AC_DEFUN([AX_SUBMODULE], [
	_AX_SUBMODULE($1, m4_bpatsubst([$1],
			[[^_abcdefghijklmnopqrstuvwxyz0123456789]],[_]), $2, $3)
])
