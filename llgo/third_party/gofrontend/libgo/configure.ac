# configure.ac -- Go library configure script.

# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Process this file with autoreconf to produce configure.

AC_PREREQ(2.64)
AC_INIT(package-unused, version-unused,, libgo)
AC_CONFIG_SRCDIR(Makefile.am)
AC_CONFIG_HEADER(config.h)

libtool_VERSION=8:0:0
AC_SUBST(libtool_VERSION)

AM_ENABLE_MULTILIB(, ..)

AC_CANONICAL_SYSTEM
target_alias=${target_alias-$host_alias}

AM_INIT_AUTOMAKE([1.9.3 no-define foreign no-dist -Wall -Wno-portability])
AH_TEMPLATE(PACKAGE, [Name of package])
AH_TEMPLATE(VERSION, [Version number of package])

m4_rename([_AC_ARG_VAR_PRECIOUS],[glibgo_PRECIOUS])
m4_define([_AC_ARG_VAR_PRECIOUS],[])
AC_PROG_CC
AC_PROG_GO
m4_rename_force([glibgo_PRECIOUS],[_AC_ARG_VAR_PRECIOUS])

AC_SUBST(CFLAGS)

AM_MAINTAINER_MODE

AC_PROG_LD
AC_PROG_RANLIB
AC_CHECK_TOOL(OBJCOPY, objcopy, missing-objcopy)

AC_LIBTOOL_DLOPEN
AM_PROG_LIBTOOL
AC_SUBST(enable_shared)
AC_SUBST(enable_static)

CC_FOR_BUILD=${CC_FOR_BUILD:-gcc}
AC_SUBST(CC_FOR_BUILD)

AC_PROG_AWK

WARN_FLAGS='-Wall -Wextra -Wwrite-strings -Wcast-qual'
AC_SUBST(WARN_FLAGS)

AC_ARG_ENABLE(werror, [AS_HELP_STRING([--enable-werror],
                                      [turns on -Werror @<:@default=yes@:>@])])
if test "x$enable_werror" != "xno"; then
  WERROR="-Werror"
fi
AC_SUBST(WERROR)

glibgo_toolexecdir=no
glibgo_toolexeclibdir=no

AC_MSG_CHECKING([for --enable-version-specific-runtime-libs])
AC_ARG_ENABLE([version-specific-runtime-libs],
  AC_HELP_STRING([--enable-version-specific-runtime-libs],
                 [Specify that runtime libraries should be installed in a compiler-specific directory]),
  [case "$enableval" in
    yes) version_specific_libs=yes ;;
    no)  version_specific_libs=no ;;
    *)   AC_MSG_ERROR([Unknown argument to enable/disable version-specific libs]);;
   esac],
  [version_specific_libs=no])
AC_MSG_RESULT($version_specific_libs)

# Version-specific runtime libs processing.
if test $version_specific_libs = yes; then
  glibgo_toolexecdir='${libdir}/gcc/${host_alias}'
  glibgo_toolexeclibdir='${toolexecdir}/${gcc_version}$(MULTISUBDIR)'
fi

# Calculate glibgo_toolexecdir, glibgo_toolexeclibdir
# Install a library built with a cross compiler in tooldir, not libdir.
if test -n "$with_cross_host" &&
   test x"$with_cross_host" != x"no"; then
  nover_glibgo_toolexecdir='${exec_prefix}/${host_alias}'
  nover_glibgo_toolexeclibdir='${toolexecdir}/lib'
else
  nover_glibgo_toolexecdir='${libdir}/gcc/${host_alias}'
  nover_glibgo_toolexeclibdir='${libdir}'
fi
multi_os_directory=`$GOC -print-multi-os-directory`
case $multi_os_directory in
  .) ;; # Avoid trailing /.
  *) nover_glibgo_toolexeclibdir=${nover_glibgo_toolexeclibdir}/${multi_os_directory} ;;
esac

if test x"$glibgo_toolexecdir" = x"no"; then
  glibgo_toolexecdir="${nover_glibgo_toolexecdir}"
  glibgo_toolexeclibdir="${nover_glibgo_toolexeclibdir}"
fi

AC_SUBST(glibgo_toolexecdir)
AC_SUBST(glibgo_toolexeclibdir)
AC_SUBST(nover_glibgo_toolexeclibdir)

# See if the user wants to configure without libffi.  Some
# architectures don't support it.  FIXME: We should set a default
# based on the host.
AC_ARG_WITH(libffi,
  AS_HELP_STRING([--without-libffi],
                 [don't use libffi]),
  [:],
  [with_libffi=${with_libffi_default-yes}])

LIBFFI=
LIBFFIINCS=
if test "$with_libffi" != no; then
   AC_DEFINE(USE_LIBFFI, 1, [Define if we're to use libffi.])
   LIBFFI=../libffi/libffi_convenience.la
   LIBFFIINCS='-I$(top_srcdir)/../libffi/include -I../libffi/include'
fi
AC_SUBST(LIBFFI)
AC_SUBST(LIBFFIINCS)

# See if the user wants to configure without libatomic. This is useful if we are
# on an architecture for which libgo does not need an atomic support library and
# libatomic does not support our C compiler.
AC_ARG_WITH(libatomic,
  AS_HELP_STRING([--without-libatomic],
                 [don't use libatomic]),
  [:],
  [with_libatomic=${with_libatomic_default-yes}])

LIBATOMIC=
if test "$with_libatomic" != no; then
   LIBATOMIC=../libatomic/libatomic_convenience.la
fi
AC_SUBST(LIBATOMIC)

# Used to tell GNU make to include a file without telling automake to
# include it.
go_include="-include"
AC_SUBST(go_include)

is_darwin=no
is_freebsd=no
is_irix=no
is_linux=no
is_netbsd=no
is_openbsd=no
is_dragonfly=no
is_rtems=no
is_solaris=no
GOOS=unknown
case ${host} in
  *-*-darwin*)   is_darwin=yes;  GOOS=darwin ;;
  *-*-freebsd*)  is_freebsd=yes; GOOS=freebsd ;;
  *-*-irix6*)    is_irix=yes;    GOOS=irix ;;
  *-*-linux*)    is_linux=yes;   GOOS=linux ;;
  *-*-netbsd*)	 is_netbsd=yes;  GOOS=netbsd ;;
  *-*-openbsd*)  is_openbsd=yes; GOOS=openbsd ;;
  *-*-dragonfly*) is_dragonfly=yes; GOOS=dragonfly ;;
  *-*-rtems*)    is_rtems=yes;   GOOS=rtems ;;
  *-*-solaris2*) is_solaris=yes; GOOS=solaris ;;
esac
AM_CONDITIONAL(LIBGO_IS_DARWIN, test $is_darwin = yes)
AM_CONDITIONAL(LIBGO_IS_FREEBSD, test $is_freebsd = yes)
AM_CONDITIONAL(LIBGO_IS_IRIX, test $is_irix = yes)
AM_CONDITIONAL(LIBGO_IS_LINUX, test $is_linux = yes)
AM_CONDITIONAL(LIBGO_IS_NETBSD, test $is_netbsd = yes)
AM_CONDITIONAL(LIBGO_IS_OPENBSD, test $is_openbsd = yes)
AM_CONDITIONAL(LIBGO_IS_DRAGONFLY, test $is_dragonfly = yes)
AM_CONDITIONAL(LIBGO_IS_RTEMS, test $is_rtems = yes)
AM_CONDITIONAL(LIBGO_IS_SOLARIS, test $is_solaris = yes)
AC_SUBST(GOOS)

dnl Test whether we need to use DejaGNU or whether we can use the
dnl simpler gotest approach.  We can only use gotest for a native
dnl build.
USE_DEJAGNU=no
case ${host} in
  *-*-rtems*) USE_DEJAGNU=yes ;;
  ${build}) ;;
  *) USE_DEJAGNU=yes ;;
esac
AC_SUBST(USE_DEJAGNU)

dnl N.B. Keep in sync with gcc/testsuite/go.test/go-test.exp (go-set-goarch).
is_386=no
is_alpha=no
is_arm=no
is_arm64=no
is_m68k=no
mips_abi=unknown
is_ppc=no
is_ppc64=no
is_ppc64le=no
is_s390=no
is_s390x=no
is_sparc=no
is_sparc64=no
is_x86_64=no
GOARCH=unknown
case ${host} in
  alpha*-*-*)
    is_alpha=yes
    GOARCH=alpha
    ;;
  aarch64-*-*)
    is_arm64=yes
    GOARCH=arm64
    ;;
  arm*-*-* | strongarm*-*-* | ep9312*-*-* | xscale-*-*)
    is_arm=yes
    GOARCH=arm
    ;;
changequote(,)dnl
  i[34567]86-*-* | x86_64-*-*)
changequote([,])dnl
    AC_COMPILE_IFELSE([
#ifdef __x86_64__
#error 64-bit
#endif],
[is_386=yes], [is_x86_64=yes])
    if test "$is_386" = "yes"; then
      GOARCH=386
    else
      GOARCH=amd64
    fi
    ;;
  m68k*-*-*)
    is_m68k=yes
    GOARCH=m68k
    ;;
  mips*-*-*)
    AC_COMPILE_IFELSE([
#if _MIPS_SIM != _ABIO32
#error not o32
#endif],
[mips_abi="o32"],
	[AC_COMPILE_IFELSE([
#if _MIPS_SIM != _ABIN32
#error not n32
#endif],
[mips_abi="n32"],
	[AC_COMPILE_IFELSE([
#if _MIPS_SIM != _ABI64
#error not n64
#endif],
[mips_abi="n64"],
	[AC_COMPILE_IFELSE([
#if _MIPS_SIM != _ABIO64
#error not o64
#endif],
[mips_abi="o64"],
	[AC_MSG_ERROR([unknown MIPS ABI])
[mips_abi="n32"]])])])])
    case "$mips_abi" in
    "o32") GOARCH=mipso32 ;;
    "n32") GOARCH=mipsn32 ;;
    "n64") GOARCH=mipsn64 ;;
    "o64") GOARCH=mipso64 ;;
    esac
    ;;
  rs6000*-*-* | powerpc*-*-*)
    AC_COMPILE_IFELSE([
#ifdef _ARCH_PPC64
#error 64-bit
#endif],
[is_ppc=yes],
    [AC_COMPILE_IFELSE([
#if defined(_BIG_ENDIAN) || defined(__BIG_ENDIAN__)
#error 64be
#endif],
[is_ppc64le=yes],[is_ppc64=yes])])
    if test "$is_ppc" = "yes"; then
      GOARCH=ppc
    elif test "$is_ppc64" = "yes"; then
      GOARCH=ppc64
    else
      GOARCH=ppc64le
    fi
    ;;
  s390*-*-*)
    AC_COMPILE_IFELSE([
#if defined(__s390x__)
#error 64-bit
#endif],
[is_s390=yes], [is_s390x=yes])
    if test "$is_s390" = "yes"; then
      GOARCH=s390
    else
      GOARCH=s390x
    fi
    ;;
  sparc*-*-*)
    AC_COMPILE_IFELSE([
#if defined(__sparcv9) || defined(__arch64__)
#error 64-bit
#endif],
[is_sparc=yes], [is_sparc64=yes])
    if test "$is_sparc" = "yes"; then
      GOARCH=sparc
    else
      GOARCH=sparc64
    fi
    ;;
esac
AM_CONDITIONAL(LIBGO_IS_386, test $is_386 = yes)
AM_CONDITIONAL(LIBGO_IS_ALPHA, test $is_alpha = yes)
AM_CONDITIONAL(LIBGO_IS_ARM, test $is_arm = yes)
AM_CONDITIONAL(LIBGO_IS_ARM64, test $is_arm64 = yes)
AM_CONDITIONAL(LIBGO_IS_M68K, test $is_m68k = yes)
AM_CONDITIONAL(LIBGO_IS_MIPS, test $mips_abi != unknown)
AM_CONDITIONAL(LIBGO_IS_MIPSO32, test $mips_abi = o32)
AM_CONDITIONAL(LIBGO_IS_MIPSN32, test $mips_abi = n32)
AM_CONDITIONAL(LIBGO_IS_MIPSN64, test $mips_abi = n64)
AM_CONDITIONAL(LIBGO_IS_MIPSO64, test $mips_abi = o64)
AM_CONDITIONAL(LIBGO_IS_PPC, test $is_ppc = yes)
AM_CONDITIONAL(LIBGO_IS_PPC64, test $is_ppc64 = yes)
AM_CONDITIONAL(LIBGO_IS_PPC64LE, test $is_ppc64le = yes)
AM_CONDITIONAL(LIBGO_IS_S390, test $is_s390 = yes)
AM_CONDITIONAL(LIBGO_IS_S390X, test $is_s390x = yes)
AM_CONDITIONAL(LIBGO_IS_SPARC, test $is_sparc = yes)
AM_CONDITIONAL(LIBGO_IS_SPARC64, test $is_sparc64 = yes)
AM_CONDITIONAL(LIBGO_IS_X86_64, test $is_x86_64 = yes)
AC_SUBST(GOARCH)

dnl Some files are only present when needed for specific architectures.
GO_LIBCALL_OS_FILE=
GO_LIBCALL_OS_ARCH_FILE=
GO_SYSCALL_OS_FILE=
GO_SYSCALL_OS_ARCH_FILE=
if test -f "${srcdir}/go/syscall/libcall_${GOOS}.go"; then
  GO_LIBCALL_OS_FILE="go/syscall/libcall_${GOOS}.go"
fi
if test -f "${srcdir}/go/syscall/libcall_${GOOS}_${GOARCH}.go"; then
  GO_LIBCALL_OS_ARCH_FILE="go/syscall/libcall_${GOOS}_${GOARCH}.go"
fi
if test -f "${srcdir}/go/syscall/syscall_${GOOS}.go"; then
  GO_SYSCALL_OS_FILE="go/syscall/syscall_${GOOS}.go"
fi
if test -f "${srcdir}/go/syscall/syscall_${GOOS}_${GOARCH}.go"; then
  GO_SYSCALL_OS_ARCH_FILE="go/syscall/syscall_${GOOS}_${GOARCH}.go"
fi
AC_SUBST(GO_LIBCALL_OS_FILE)
AC_SUBST(GO_LIBCALL_OS_ARCH_FILE)
AC_SUBST(GO_SYSCALL_OS_FILE)
AC_SUBST(GO_SYSCALL_OS_ARCH_FILE)

dnl Special flags used to generate sysinfo.go.
OSCFLAGS="-D_GNU_SOURCE -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64"
case "$target" in
    mips-sgi-irix6.5*)
	# IRIX 6 needs _XOPEN_SOURCE=500 for the XPG5 version of struct
	# msghdr in <sys/socket.h>.
	OSCFLAGS="$OSCFLAGS -D_XOPEN_SOURCE=500"
	;;
    *-*-solaris2.1[[01]])
	# Solaris 10+ needs this so struct msghdr gets the msg_control
	# etc. fields in <sys/socket.h> (_XPG4_2).  _XOPEN_SOURCE=600 as
	# above doesn't work with C99.
	OSCFLAGS="$OSCFLAGS -std=gnu99 -D_XOPEN_SOURCE=600 -D__EXTENSIONS__"
	;;
esac
AC_SUBST(OSCFLAGS)

dnl Use -fsplit-stack when compiling C code if available.
AC_CACHE_CHECK([whether -fsplit-stack is supported],
[libgo_cv_c_split_stack_supported],
[CFLAGS_hold=$CFLAGS
CFLAGS="$CFLAGS -fsplit-stack"
AC_COMPILE_IFELSE([[int i;]],
[libgo_cv_c_split_stack_supported=yes],
[libgo_cv_c_split_stack_supported=no])
CFLAGS=$CFLAGS_hold])

dnl Make sure the linker permits -fsplit-stack.  Old versions of gold will
dnl reject split-stack code calling non-split-stack code on targets
dnl they don't support.
AC_CACHE_CHECK([whether linker supports split/non-split linked together],
[libgo_cv_c_linker_split_non_split],
[cat > conftest1.c << EOF
extern void f();
int main() { f(); return 0; }
EOF
cat > conftest2.c << EOF
void f() {}
EOF
$CC -c -fsplit-stack $CFLAGS $CPPFLAGS conftest1.c
$CC -c $CFLAGS $CPPFLAGS conftest2.c
if $CC -o conftest conftest1.$ac_objext conftest2.$ac_objext; then
  libgo_cv_c_linker_split_non_split=yes
else
  libgo_cv_c_linker_split_non_split=no
fi
rm -f conftest1.* conftest2.* conftest])

if test "$libgo_cv_c_split_stack_supported" = yes -a "$libgo_cv_c_linker_split_non_split" = yes; then
  SPLIT_STACK=-fsplit-stack
  AC_DEFINE(USING_SPLIT_STACK, 1,
		[Define if the compiler supports -fsplit-stack])
else
  SPLIT_STACK=
fi
AC_SUBST(SPLIT_STACK)
AM_CONDITIONAL(USING_SPLIT_STACK,
	test "$libgo_cv_c_split_stack_supported" = yes -a "$libgo_cv_c_linker_split_non_split" = yes)

dnl If the compiler supports split-stack but the linker does not, then
dnl we need to explicitly disable split-stack for Go.
if test "$libgo_cv_c_split_stack_supported" = yes -a "$libgo_cv_c_linker_split_non_split" = no; then
  GO_SPLIT_STACK=-fno-split-stack
else
  GO_SPLIT_STACK=
fi
AC_SUBST(GO_SPLIT_STACK)

dnl Check whether the linker does stack munging when calling from
dnl split-stack into non-split-stack code.  We check this by looking
dnl at the --help output.  FIXME: This is only half right: it's
dnl possible for the linker to support this for some targets but not
dnl others.
dnl This is slightly different from the above check, which is whether
dnl the linker permits the call at all.
AC_CACHE_CHECK([whether linker supports split stack],
[libgo_cv_c_linker_supports_split_stack],
[libgo_cv_c_linker_supports_split_stack=no
if $GOC -Wl,--help 2>/dev/null | grep split-stack-adjust-size >/dev/null 2>&1; then
  libgo_cv_c_linker_supports_split_stack=yes
fi])
if test "$libgo_cv_c_linker_supports_split_stack" = yes; then
  AC_DEFINE(LINKER_SUPPORTS_SPLIT_STACK, 1,
	    [Define if the linker support split stack adjustments])
fi

AC_CACHE_CHECK([whether compiler is llgo],
[libgo_cv_c_goc_is_llgo],
[libgo_cv_c_goc_is_llgo=no
if $GOC -dumpversion 2>/dev/null | grep llgo >/dev/null 2>&1; then
  libgo_cv_c_goc_is_llgo=yes
fi])
AM_CONDITIONAL(GOC_IS_LLGO, test "$libgo_cv_c_goc_is_llgo" = yes)

dnl Test for the -lm library.
MATH_LIBS=
AC_CHECK_LIB([m], [sqrt], MATH_LIBS=-lm)
AC_SUBST(MATH_LIBS)

dnl Test for -lsocket and -lnsl.  Copied from libjava/configure.ac.
AC_CACHE_CHECK([for socket libraries], libgo_cv_lib_sockets,
  [libgo_cv_lib_sockets=
   libgo_check_both=no
   AC_CHECK_FUNC(connect, libgo_check_socket=no, libgo_check_socket=yes)
   if test "$libgo_check_socket" = "yes"; then
     unset ac_cv_func_connect
     AC_CHECK_LIB(socket, main, libgo_cv_lib_sockets="-lsocket",
     		  libgo_check_both=yes)
   fi
   if test "$libgo_check_both" = "yes"; then
     libgo_old_libs=$LIBS
     LIBS="$LIBS -lsocket -lnsl"
     unset ac_cv_func_accept
     AC_CHECK_FUNC(accept,
		   [libgo_check_nsl=no
		    libgo_cv_lib_sockets="-lsocket -lnsl"])
     unset ac_cv_func_accept
     LIBS=$libgo_old_libs
   fi
   unset ac_cv_func_gethostbyname
   libgo_old_libs="$LIBS"
   AC_CHECK_FUNC(gethostbyname, ,
		 [AC_CHECK_LIB(nsl, main,
		 	[libgo_cv_lib_sockets="$libgo_cv_lib_sockets -lnsl"])])
   unset ac_cv_func_gethostbyname
   AC_CHECK_FUNC(sendfile, ,
		 [AC_CHECK_LIB(sendfile, main,
		 	[libgo_cv_lib_sockets="$libgo_cv_lib_sockets -lsendfile"])])
   LIBS=$libgo_old_libs
])
NET_LIBS="$libgo_cv_lib_sockets"
AC_SUBST(NET_LIBS)

dnl Test whether the compiler supports the -pthread option.
AC_CACHE_CHECK([whether -pthread is supported],
[libgo_cv_lib_pthread],
[CFLAGS_hold=$CFLAGS
CFLAGS="$CFLAGS -pthread"
AC_COMPILE_IFELSE([[int i;]],
[libgo_cv_lib_pthread=yes],
[libgo_cv_lib_pthread=no])
CFLAGS=$CFLAGS_hold])
PTHREAD_CFLAGS=
if test "$libgo_cv_lib_pthread" = yes; then
  PTHREAD_CFLAGS=-pthread
fi
AC_SUBST(PTHREAD_CFLAGS)

dnl Test for the -lpthread library.
PTHREAD_LIBS=
AC_CHECK_LIB([pthread], [pthread_create], PTHREAD_LIBS=-lpthread)
AC_SUBST(PTHREAD_LIBS)

dnl Test if -lrt is required for sched_yield or nanosleep or clock_gettime.
AC_SEARCH_LIBS([sched_yield], [rt])
AC_SEARCH_LIBS([nanosleep], [rt])
AC_SEARCH_LIBS([clock_gettime], [rt])

AC_C_BIGENDIAN

GCC_CHECK_UNWIND_GETIPINFO

AC_CHECK_HEADERS(sched.h sys/file.h sys/mman.h syscall.h sys/epoll.h sys/inotify.h sys/ptrace.h sys/syscall.h sys/user.h sys/utsname.h sys/select.h sys/socket.h net/if.h net/if_arp.h net/route.h netpacket/packet.h sys/prctl.h sys/mount.h sys/vfs.h sys/statfs.h sys/timex.h sys/sysinfo.h utime.h linux/ether.h linux/fs.h linux/reboot.h netinet/icmp6.h netinet/in_syst.h netinet/ip.h netinet/ip_mroute.h netinet/if_ether.h)

AC_CHECK_HEADERS([linux/filter.h linux/if_addr.h linux/if_ether.h linux/if_tun.h linux/netlink.h linux/rtnetlink.h], [], [],
[#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
])

AC_CACHE_CHECK([whether <ustat.h> can be used],
[libgo_cv_c_ustat_h],
[CFLAGS_hold=$CFLAGS
CFLAGS="$CFLAGS -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE $OSCFLAGS"
AC_COMPILE_IFELSE(
[AC_LANG_SOURCE([
#include <sys/types.h>
#ifdef HAVE_LINUX_FILTER_H
#include <linux/filter.h>
#endif
#include <ustat.h>
])], [libgo_cv_c_ustat_h=yes], [libgo_cv_c_ustat_h=no])
CFLAGS=$CFLAGS_hold])
if test $libgo_cv_c_ustat_h = yes; then
  AC_DEFINE(HAVE_USTAT_H, 1,
    [Define to 1 if you have the <ustat.h> header file and it works.])
fi

AM_CONDITIONAL(HAVE_SYS_MMAN_H, test "$ac_cv_header_sys_mman_h" = yes)

AC_CHECK_FUNCS(strerror_r strsignal wait4 mincore setenv unsetenv dl_iterate_phdr)
AM_CONDITIONAL(HAVE_STRERROR_R, test "$ac_cv_func_strerror_r" = yes)
AM_CONDITIONAL(HAVE_WAIT4, test "$ac_cv_func_wait4" = yes)

AC_CHECK_FUNCS(accept4 dup3 epoll_create1 faccessat fallocate fchmodat fchownat futimesat getxattr inotify_add_watch inotify_init inotify_init1 inotify_rm_watch listxattr mkdirat mknodat open64 openat pipe2 removexattr renameat setxattr sync_file_range splice tee unlinkat unshare utimensat)
AC_TYPE_OFF_T
AC_CHECK_TYPES([loff_t])

LIBS_hold="$LIBS"
LIBS="$LIBS -lm"
AC_CHECK_FUNCS(cosl expl logl sinl tanl acosl asinl atanl atan2l expm1l ldexpl log10l log1pl)
LIBS="$LIBS_hold"

CFLAGS_hold="$CFLAGS"
CFLAGS="$CFLAGS $PTHREAD_CFLAGS"
LIBS_hold="$LIBS"
LIBS="$LIBS $PTHREAD_LIBS"
AC_CHECK_FUNCS(sem_timedwait)
CFLAGS="$CFLAGS_hold"
LIBS="$LIBS_hold"

LIBS_hold="$LIBS"
LIBS="$LIBS $MATH_LIBS"
AC_CHECK_FUNCS(matherr)
LIBS="$LIBS_hold"

AC_CACHE_CHECK([for __sync_bool_compare_and_swap_4],
[libgo_cv_func___sync_bool_compare_and_swap_4],
[AC_LINK_IFELSE([
typedef unsigned int uint32  __attribute__ ((mode (SI)));
uint32 i;
int main() { return __sync_bool_compare_and_swap (&i, 0, 1); }
],
[libgo_cv_func___sync_bool_compare_and_swap_4=yes],
[libgo_cv_func___sync_bool_compare_and_swap_4=no])])
if test "$libgo_cv_func___sync_bool_compare_and_swap_4" = "yes"; then
  AC_DEFINE(HAVE_SYNC_BOOL_COMPARE_AND_SWAP_4, 1,
    [Define to 1 if the compiler provides the __sync_bool_compare_and_swap function for uint32])
fi

AC_CACHE_CHECK([for __sync_bool_compare_and_swap_8],
[libgo_cv_func___sync_bool_compare_and_swap_8],
[AC_LINK_IFELSE([
typedef unsigned int uint64  __attribute__ ((mode (DI)));
uint64 i;
int main() { return __sync_bool_compare_and_swap (&i, 0, 1); }
],
[libgo_cv_func___sync_bool_compare_and_swap_8=yes],
[libgo_cv_func___sync_bool_compare_and_swap_8=no])])
if test "$libgo_cv_func___sync_bool_compare_and_swap_8" = "yes"; then
  AC_DEFINE(HAVE_SYNC_BOOL_COMPARE_AND_SWAP_8, 1,
    [Define to 1 if the compiler provides the __sync_bool_compare_and_swap function for uint64])
fi

AC_CACHE_CHECK([for __sync_fetch_and_add_4],
[libgo_cv_func___sync_fetch_and_add_4],
[AC_LINK_IFELSE([
typedef unsigned int uint32  __attribute__ ((mode (SI)));
uint32 i;
int main() { return __sync_fetch_and_add (&i, 1); }
],
[libgo_cv_func___sync_fetch_and_add_4=yes],
[libgo_cv_func___sync_fetch_and_add_4=no])])
if test "$libgo_cv_func___sync_fetch_and_add_4" = "yes"; then
  AC_DEFINE(HAVE_SYNC_FETCH_AND_ADD_4, 1,
    [Define to 1 if the compiler provides the __sync_fetch_and_add function for uint32])
fi

AC_CACHE_CHECK([for __sync_add_and_fetch_8],
[libgo_cv_func___sync_add_and_fetch_8],
[AC_LINK_IFELSE([
typedef unsigned int uint64  __attribute__ ((mode (DI)));
uint64 i;
int main() { return __sync_add_and_fetch (&i, 1); }
],
[libgo_cv_func___sync_add_and_fetch_8=yes],
[libgo_cv_func___sync_add_and_fetch_8=no])])
if test "$libgo_cv_func___sync_add_and_fetch_8" = "yes"; then
  AC_DEFINE(HAVE_SYNC_ADD_AND_FETCH_8, 1,
    [Define to 1 if the compiler provides the __sync_add_and_fetch function for uint64])
fi

dnl For x86 we want to use the -minline-all-stringops option to avoid
dnl forcing a stack split when calling memcpy and friends.
AC_CACHE_CHECK([whether compiler supports -minline-all-stringops],
[libgo_cv_c_stringops],
[CFLAGS_hold=$CFLAGS
CFLAGS="$CFLAGS -minline-all-stringops"
AC_COMPILE_IFELSE([int i;],
[libgo_cv_c_stringops=yes],
[libgo_cv_c_stringops=no])
CFLAGS=$CFLAGS_hold])
STRINGOPS_FLAG=
if test "$libgo_cv_c_stringops" = yes; then
  STRINGOPS_FLAG=-minline-all-stringops
fi
AC_SUBST(STRINGOPS_FLAG)

dnl For x86 we want to compile the math library with -mfancy-math-387
dnl -funsafe-math-optimizations so that we can use the builtin
dnl instructions directly.
AC_CACHE_CHECK([whether compiler supports -mfancy-math-387],
[libgo_cv_c_fancymath],
[CFLAGS_hold=$CFLAGS
CFLAGS="$CFLAGS -mfancy-math-387"
AC_COMPILE_IFELSE([int i;],
[libgo_cv_c_fancymath=yes],
[libgo_cv_c_fancymath=no])
CFLAGS=$CFLAGS_hold])
MATH_FLAG=
if test "$libgo_cv_c_fancymath" = yes; then
  MATH_FLAG="-mfancy-math-387 -funsafe-math-optimizations"
else
  MATH_FLAG="-ffp-contract=off"
fi
AC_SUBST(MATH_FLAG)

CFLAGS_hold=$CFLAGS
CFLAGS="$CFLAGS -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE"
AC_CHECK_TYPES([off64_t])
CFLAGS=$CFLAGS_hold

dnl Work out the size of the epoll_events struct on GNU/Linux.
AC_CACHE_CHECK([epoll_event size],
[libgo_cv_c_epoll_event_size],
[AC_COMPUTE_INT(libgo_cv_c_epoll_event_size,
[sizeof (struct epoll_event)],
[#include <sys/epoll.h>],
[libgo_cv_c_epoll_event_size=0])])
SIZEOF_STRUCT_EPOLL_EVENT=${libgo_cv_c_epoll_event_size}
AC_SUBST(SIZEOF_STRUCT_EPOLL_EVENT)

dnl Work out the offset of the fd field in the epoll_events struct on
dnl GNU/Linux.
AC_CACHE_CHECK([epoll_event data.fd offset],
[libgo_cv_c_epoll_event_fd_offset],
[AC_COMPUTE_INT(libgo_cv_c_epoll_event_fd_offset,
[offsetof (struct epoll_event, data.fd)],
[#include <stddef.h>
#include <sys/epoll.h>],
[libgo_cv_c_epoll_event_fd_offset=0])])
STRUCT_EPOLL_EVENT_FD_OFFSET=${libgo_cv_c_epoll_event_fd_offset}
AC_SUBST(STRUCT_EPOLL_EVENT_FD_OFFSET)

dnl Check if <sys/stat.h> uses timespec_t for st_?tim members.  Introduced
dnl in Solaris 12 for XPG7 compatibility.
AC_EGREP_HEADER([timespec_t.*st_atim], [sys/stat.h],
		[have_stat_timespec=yes], [have_stat_timespec=no])
AM_CONDITIONAL(HAVE_STAT_TIMESPEC, test $have_stat_timespec = yes)

dnl See if struct exception is defined in <math.h>.
AC_CHECK_TYPE([struct exception],
[libgo_has_struct_exception=yes],
[libgo_has_struct_exception=no],
[#include <math.h>])
if test "$libgo_has_struct_exception" = "yes"; then
  AC_DEFINE(HAVE_STRUCT_EXCEPTION, 1,
            [Define to 1 if <math.h> defines struct exception])
fi

dnl See whether setcontext changes the value of TLS variables.
AC_CACHE_CHECK([whether setcontext clobbers TLS variables],
[libgo_cv_lib_setcontext_clobbers_tls],
[CFLAGS_hold="$CFLAGS"
CFLAGS="$PTHREAD_CFLAGS"
LIBS_hold="$LIBS"
LIBS="$LIBS $PTHREAD_LIBS"
AC_CHECK_SIZEOF([void *])
AS_VAR_ARITH([ptr_type_size], [$ac_cv_sizeof_void_p \* 8])
AC_RUN_IFELSE(
  [AC_LANG_SOURCE([
#include <pthread.h>
#include <stdlib.h>
#include <ucontext.h>
#include <unistd.h>

__thread int tls;

static char stack[[10 * 1024 * 1024]];
static ucontext_t c;

/* Called via makecontext/setcontext.  */

static void
cfn (void)
{
  exit (tls);
}

/* Called via pthread_create.  */

static void *
tfn (void *dummy)
{
  /* The thread should still see this value after calling
     setcontext.  */
  tls = 0;

  setcontext (&c);

  /* The call to setcontext should not return.  */
  abort ();
}

int
main ()
{
  pthread_t tid;

  /* The thread should not see this value.  */
  tls = 1;

  if (getcontext (&c) < 0)
    abort ();

  c.uc_stack.ss_sp = stack;
#ifdef MAKECONTEXT_STACK_TOP
  c.uc_stack.ss_sp += sizeof stack;
#endif
  c.uc_stack.ss_flags = 0;
  c.uc_stack.ss_size = sizeof stack;
  c.uc_link = NULL;
  makecontext (&c, cfn, 0);

  if (pthread_create (&tid, NULL, tfn, NULL) != 0)
    abort ();

  if (pthread_join (tid, NULL) != 0)
    abort ();

  /* The thread should have called exit.  */
  abort ();
}
])],
[libgo_cv_lib_setcontext_clobbers_tls=no],
[libgo_cv_lib_setcontext_clobbers_tls=yes],
[case "$target:$ptr_type_size" in
  i?86-*-solaris2.1[[01]]:64 | x86_64*-*-solaris2.1[[01]]:64)
    libgo_cv_lib_setcontext_clobbers_tls=yes ;;
  *)
    libgo_cv_lib_setcontext_clobbers_tls=no ;;
 esac
])
CFLAGS="$CFLAGS_hold"
LIBS="$LIBS_hold"
])
if test "$libgo_cv_lib_setcontext_clobbers_tls" = "yes"; then
  AC_DEFINE(SETCONTEXT_CLOBBERS_TLS, 1,
	    [Define if setcontext clobbers TLS variables])
fi

AC_CACHE_CHECK([whether .eh_frame section should be read-only],
libgo_cv_ro_eh_frame, [
libgo_cv_ro_eh_frame=no
echo 'extern void foo (void); void bar (void) { foo (); foo (); }' > conftest.c
if $CC $CFLAGS -S -fpic -fexceptions -o conftest.s conftest.c > /dev/null 2>&1; then
  if grep '.section.*eh_frame.*"a"' conftest.s > /dev/null; then
    libgo_cv_ro_eh_frame=yes
  elif grep '.section.*eh_frame.*#alloc' conftest.c \
       | grep -v '#write' > /dev/null; then
    libgo_cv_ro_eh_frame=yes
  fi
fi
rm -f conftest.*
])
if test "x$libgo_cv_ro_eh_frame" = xyes; then
  AC_DEFINE(EH_FRAME_FLAGS, "a",
	    [Define to the flags needed for the .section .eh_frame directive.])
else
  AC_DEFINE(EH_FRAME_FLAGS, "aw",
	    [Define to the flags needed for the .section .eh_frame directive.])
fi

AC_CACHE_CHECK([if compiler supports -Qunused-arguments],
[libgo_cv_c_unused_arguments],
[CFLAGS_hold=$CFLAGS
CFLAGS="$CFLAGS -Qunused-arguments"
AC_COMPILE_IFELSE([[int i;]],
[libgo_cv_c_unused_arguments=yes],
[libgo_cv_c_unused_arguments=no])
CFLAGS=$CFLAGS_hold])

AC_CACHE_CHECK([if assembler supports GNU comdat group syntax],
libgo_cv_as_comdat_gnu, [
echo '.section .text,"axG",@progbits,.foo,comdat' > conftest.s
CFLAGS_hold=$CFLAGS
if test "$libgo_cv_c_unused_arguments" = yes; then
  CFLAGS="$CFLAGS -Qunused-arguments"
fi
if $CC $CFLAGS -c conftest.s > /dev/null 2>&1; then
  libgo_cv_as_comdat_gnu=yes
else
  libgo_cv_as_comdat_gnu=no
fi
CFLAGS=$CFLAGS_hold
])
if test "x$libgo_cv_as_comdat_gnu" = xyes; then
  AC_DEFINE(HAVE_AS_COMDAT_GAS, 1,
	    [Define if your assembler supports GNU comdat group syntax.])
fi

AC_CACHE_CHECK([assembler supports pc related relocs],
libgo_cv_as_x86_pcrel, [
libgo_cv_as_x86_pcrel=yes
echo '.text; foo: nop; .data; .long foo-.; .text' > conftest.s
CFLAGS_hold=$CFLAGS
if test "$libgo_cv_c_unused_arguments" = yes; then
  CFLAGS="$CFLAGS -Qunused-arguments"
fi
if $CC $CFLAGS -c conftest.s 2>&1 | $EGREP -i 'illegal|warning' > /dev/null; then
    libgo_cv_as_x86_pcrel=no
fi
CFLAGS=$CFLAGS_hold
])
if test "x$libgo_cv_as_x86_pcrel" = xyes; then
  AC_DEFINE(HAVE_AS_X86_PCREL, 1,
	    [Define if your assembler supports PC relative relocs.])
fi

AC_CACHE_CHECK([assembler supports unwind section type],
libgo_cv_as_x86_64_unwind_section_type, [
libgo_cv_as_x86_64_unwind_section_type=yes
echo '.section .eh_frame,"a",@unwind' > conftest.s
CFLAGS_hold=$CFLAGS
if test "$libgo_cv_c_unused_arguments" = yes; then
  CFLAGS="$CFLAGS -Qunused-arguments"
fi
if $CC $CFLAGS -c conftest.s 2>&1 | grep -i warning > /dev/null; then
    libgo_cv_as_x86_64_unwind_section_type=no
fi
CFLAGS=$CFLAGS_hold
])
if test "x$libgo_cv_as_x86_64_unwind_section_type" = xyes; then
  AC_DEFINE(HAVE_AS_X86_64_UNWIND_SECTION_TYPE, 1,
	    [Define if your assembler supports unwind section type.])
fi

AC_CACHE_SAVE

if test ${multilib} = yes; then
  multilib_arg="--enable-multilib"
else
  multilib_arg=
fi

AC_CONFIG_FILES(Makefile testsuite/Makefile)

AC_CONFIG_COMMANDS([default],
[if test -n "$CONFIG_FILES"; then
   # Multilibs need MULTISUBDIR defined correctly in certain makefiles so
   # that multilib installs will end up installed in the correct place.
   # The testsuite needs it for multilib-aware ABI baseline files.
   # To work around this not being passed down from config-ml.in ->
   # srcdir/Makefile.am -> srcdir/{src,libsupc++,...}/Makefile.am, manually
   # append it here.  Only modify Makefiles that have just been created.
   #
   # Also, get rid of this simulated-VPATH thing that automake does.
   cat > vpsed << \_EOF
s!`test -f '$<' || echo '$(srcdir)/'`!!
_EOF
   for i in $SUBDIRS; do
    case $CONFIG_FILES in
     *${i}/Makefile*)
       #echo "Adding MULTISUBDIR to $i/Makefile"
       sed -f vpsed $i/Makefile > tmp
       grep '^MULTISUBDIR =' Makefile >> tmp
       mv tmp $i/Makefile
       ;;
    esac
   done
   rm vpsed
 fi
],
[
# Variables needed in config.status (file generation) which aren't already
# passed by autoconf.
SUBDIRS="$SUBDIRS"
])

AC_OUTPUT
