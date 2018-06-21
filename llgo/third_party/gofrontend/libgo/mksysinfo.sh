#!/bin/sh

# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Create sysinfo.go.

# This shell script creates the sysinfo.go file which holds types and
# constants extracted from the system header files.  This relies on a
# hook in gcc: the -fdump-go-spec option will generate debugging
# information in Go syntax.

# We currently #include all the files at once, which works, but leads
# to exposing some names which ideally should not be exposed, as they
# match grep patterns.  E.g., WCHAR_MIN gets exposed because it starts
# with W, like the wait flags.

CC=${CC:-gcc}
OUT=tmp-sysinfo.go

set -e

rm -f sysinfo.c
cat > sysinfo.c <<EOF
#include "config.h"

#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
/* <netinet/tcp.h> needs u_char/u_short, but <sys/bsd_types> is only
   included by <netinet/in.h> if _SGIAPI (i.e. _SGI_SOURCE
   && !_XOPEN_SOURCE.
   <sys/termios.h> only defines TIOCNOTTY if !_XOPEN_SOURCE, while
   <sys/ttold.h> does so unconditionally.  */
#ifdef __sgi__
#include <sys/bsd_types.h>
#include <sys/ttold.h>
#endif
#include <netinet/tcp.h>
#if defined(HAVE_NETINET_IN_SYSTM_H)
#include <netinet/in_systm.h>
#endif
#if defined(HAVE_NETINET_IP_H)
#include <netinet/ip.h>
#endif
#if defined(HAVE_NETINET_IP_MROUTE_H)
#include <netinet/ip_mroute.h>
#endif
#if defined(HAVE_NETINET_IF_ETHER_H)
#include <netinet/if_ether.h>
#endif
#include <signal.h>
#include <sys/ioctl.h>
#include <termios.h>
#if defined(HAVE_SYSCALL_H)
#include <syscall.h>
#endif
#if defined(HAVE_SYS_SYSCALL_H)
#include <sys/syscall.h>
#endif
#if defined(HAVE_SYS_EPOLL_H)
#include <sys/epoll.h>
#endif
#if defined(HAVE_SYS_FILE_H)
#include <sys/file.h>
#endif
#if defined(HAVE_SYS_MMAN_H)
#include <sys/mman.h>
#endif
#if defined(HAVE_SYS_PRCTL_H)
#include <sys/prctl.h>
#endif
#if defined(HAVE_SYS_PTRACE_H)
#include <sys/ptrace.h>
#endif
#include <sys/resource.h>
#include <sys/uio.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/wait.h>
#include <sys/un.h>
#if defined(HAVE_SYS_USER_H)
#include <sys/user.h>
#endif
#if defined(HAVE_SYS_UTSNAME_H)
#include <sys/utsname.h>
#endif
#if defined(HAVE_SYS_SELECT_H)
#include <sys/select.h>
#endif
#include <time.h>
#include <unistd.h>
#include <netdb.h>
#include <pwd.h>
#if defined(HAVE_LINUX_FILTER_H)
#include <linux/filter.h>
#endif
#if defined(HAVE_LINUX_IF_ADDR_H)
#include <linux/if_addr.h>
#endif
#if defined(HAVE_LINUX_IF_ETHER_H)
#include <linux/if_ether.h>
#endif
#if defined(HAVE_LINUX_IF_TUN_H)
#include <linux/if_tun.h>
#endif
#if defined(HAVE_LINUX_NETLINK_H)
#include <linux/netlink.h>
#endif
#if defined(HAVE_LINUX_RTNETLINK_H)
#include <linux/rtnetlink.h>
#endif
#if defined(HAVE_NET_IF_H)
#include <net/if.h>
#endif
#if defined(HAVE_NET_IF_ARP_H)
#include <net/if_arp.h>
#endif
#if defined(HAVE_NET_ROUTE_H)
#include <net/route.h>
#endif
#if defined (HAVE_NETPACKET_PACKET_H)
#include <netpacket/packet.h>
#endif
#if defined(HAVE_SYS_MOUNT_H)
#include <sys/mount.h>
#endif
#if defined(HAVE_SYS_VFS_H)
#include <sys/vfs.h>
#endif
#if defined(HAVE_STATFS_H)
#include <sys/statfs.h>
#endif
#if defined(HAVE_SYS_TIMEX_H)
#include <sys/timex.h>
#endif
#if defined(HAVE_SYS_SYSINFO_H)
#include <sys/sysinfo.h>
#endif
#if defined(HAVE_USTAT_H)
#include <ustat.h>
#endif
#if defined(HAVE_UTIME_H)
#include <utime.h>
#endif
#if defined(HAVE_LINUX_ETHER_H)
#include <linux/ether.h>
#endif
#if defined(HAVE_LINUX_FS_H)
#include <linux/fs.h>
#endif
#if defined(HAVE_LINUX_REBOOT_H)
#include <linux/reboot.h>
#endif
#if defined(HAVE_SYS_INOTIFY_H)
#include <sys/inotify.h>
#endif
#if defined(HAVE_NETINET_ICMP6_H)
#include <netinet/icmp6.h>
#endif
#if defined(HAVE_SCHED_H)
#include <sched.h>
#endif

/* Constants that may only be defined as expressions on some systems,
   expressions too complex for -fdump-go-spec to handle.  These are
   handled specially below.  */
enum {
#ifdef TIOCGWINSZ
  TIOCGWINSZ_val = TIOCGWINSZ,
#endif
#ifdef TIOCSWINSZ
  TIOCSWINSZ_val = TIOCSWINSZ,
#endif
#ifdef TIOCNOTTY
  TIOCNOTTY_val = TIOCNOTTY,
#endif
#ifdef TIOCSCTTY
  TIOCSCTTY_val = TIOCSCTTY,
#endif
#ifdef TIOCGPGRP
  TIOCGPGRP_val = TIOCGPGRP,
#endif
#ifdef TIOCSPGRP
  TIOCSPGRP_val = TIOCSPGRP,
#endif
#ifdef TIOCGPTN
  TIOCGPTN_val = TIOCGPTN,
#endif
#ifdef TIOCSPTLCK
  TIOCSPTLCK_val = TIOCSPTLCK,
#endif
#ifdef TIOCGDEV
  TIOCGDEV_val = TIOCGDEV,
#endif
#ifdef TIOCSIG
  TIOCSIG_val = TIOCSIG,
#endif
#ifdef TCGETS
  TCGETS_val = TCGETS,
#endif
#ifdef TCSETS
  TCSETS_val = TCSETS,
#endif
#ifdef TUNSETIFF
  TUNSETIFF_val = TUNSETIFF,
#endif
#ifdef TUNSETNOCSUM
  TUNSETNOCSUM_val = TUNSETNOCSUM,
#endif
#ifdef TUNSETDEBUG
  TUNSETDEBUG_val = TUNSETDEBUG,
#endif
#ifdef TUNSETPERSIST
  TUNSETPERSIST_val = TUNSETPERSIST,
#endif
#ifdef TUNSETOWNER
  TUNSETOWNER_val = TUNSETOWNER,
#endif
#ifdef TUNSETLINK
  TUNSETLINK_val = TUNSETLINK,
#endif
#ifdef TUNSETGROUP
  TUNSETGROUP_val = TUNSETGROUP,
#endif
#ifdef TUNGETFEATURES
  TUNGETFEATURES_val = TUNGETFEATURES,
#endif
#ifdef TUNSETOFFLOAD
  TUNSETOFFLOAD_val = TUNSETOFFLOAD,
#endif
#ifdef TUNSETTXFILTER
  TUNSETTXFILTER_val = TUNSETTXFILTER,
#endif
#ifdef TUNGETIFF
  TUNGETIFF_val = TUNGETIFF,
#endif
#ifdef TUNGETSNDBUF
  TUNGETSNDBUF_val = TUNGETSNDBUF,
#endif
#ifdef TUNSETSNDBUF
  TUNSETSNDBUF_val = TUNSETSNDBUF,
#endif
#ifdef TUNATTACHFILTER
  TUNATTACHFILTER_val = TUNATTACHFILTER,
#endif
#ifdef TUNDETACHFILTER
  TUNDETACHFILTER_val = TUNDETACHFILTER,
#endif
#ifdef TUNGETVNETHDRSZ
  TUNGETVNETHDRSZ_val = TUNGETVNETHDRSZ,
#endif
#ifdef TUNSETVNETHDRSZ
  TUNSETVNETHDRSZ_val = TUNSETVNETHDRSZ,
#endif
#ifdef TUNSETQUEUE
  TUNSETQUEUE_val = TUNSETQUEUE,
#endif
#ifdef TUNSETIFINDEX
  TUNSETIFINDEX_val = TUNSETIFINDEX,
#endif
#ifdef TUNGETFILTER
  TUNGETFILTER_val = TUNGETFILTER,
#endif

};
EOF

${CC} -fdump-go-spec=gen-sysinfo.go -std=gnu99 -S -o sysinfo.s sysinfo.c

echo 'package syscall' > ${OUT}
echo 'import "unsafe"' >> ${OUT}
echo 'type _ unsafe.Pointer' >> ${OUT}

# Get all the consts and types, skipping ones which could not be
# represented in Go and ones which we need to rewrite.  We also skip
# function declarations, as we don't need them here.  All the symbols
# will all have a leading underscore.
grep -v '^// ' gen-sysinfo.go | \
  grep -v '^func' | \
  grep -v '^type _timeval ' | \
  grep -v '^type _timespec_t ' | \
  grep -v '^type _timespec ' | \
  grep -v '^type _timestruc_t ' | \
  grep -v '^type _epoll_' | \
  grep -v 'in6_addr' | \
  grep -v 'sockaddr_in6' | \
  sed -e 's/\([^a-zA-Z0-9_]\)_timeval\([^a-zA-Z0-9_]\)/\1Timeval\2/g' \
      -e 's/\([^a-zA-Z0-9_]\)_timespec_t\([^a-zA-Z0-9_]\)/\1Timespec\2/g' \
      -e 's/\([^a-zA-Z0-9_]\)_timespec\([^a-zA-Z0-9_]\)/\1Timespec\2/g' \
      -e 's/\([^a-zA-Z0-9_]\)_timestruc_t\([^a-zA-Z0-9_]\)/\1Timestruc\2/g' \
    >> ${OUT}

# The errno constants.  These get type Errno.
echo '#include <errno.h>' | ${CC} -x c - -E -dM | \
  egrep '#define E[A-Z0-9_]+ ' | \
  sed -e 's/^#define \(E[A-Z0-9_]*\) .*$/const \1 = Errno(_\1)/' >> ${OUT}

# The O_xxx flags.
egrep '^const _(O|F|FD)_' gen-sysinfo.go | \
  sed -e 's/^\(const \)_\([^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
if ! grep '^const O_ASYNC' ${OUT} >/dev/null 2>&1; then
  echo "const O_ASYNC = 0" >> ${OUT}
fi
if ! grep '^const O_CLOEXEC' ${OUT} >/dev/null 2>&1; then
  echo "const O_CLOEXEC = 0" >> ${OUT}
fi

# The os package requires F_DUPFD_CLOEXEC to be defined.
if ! grep '^const F_DUPFD_CLOEXEC' ${OUT} >/dev/null 2>&1; then
  echo "const F_DUPFD_CLOEXEC = 0" >> ${OUT}
fi

# These flags can be lost on i386 GNU/Linux when using
# -D_FILE_OFFSET_BITS=64, because we see "#define F_SETLK F_SETLK64"
# before we see the definition of F_SETLK64.
for flag in F_GETLK F_SETLK F_SETLKW; do
  if ! grep "^const ${flag} " ${OUT} >/dev/null 2>&1 \
      && grep "^const ${flag}64 " ${OUT} >/dev/null 2>&1; then
    echo "const ${flag} = ${flag}64" >> ${OUT}
  fi
done

# The Flock_t struct for fcntl.
grep '^type _flock ' gen-sysinfo.go | \
    sed -e 's/type _flock/type Flock_t/' \
      -e 's/l_type/Type/' \
      -e 's/l_whence/Whence/' \
      -e 's/l_start/Start/' \
      -e 's/l_len/Len/' \
      -e 's/l_pid/Pid/' \
    >> ${OUT}

# The signal numbers.
grep '^const _SIG[^_]' gen-sysinfo.go | \
  grep -v '^const _SIGEV_' | \
  sed -e 's/^\(const \)_\(SIG[^= ]*\)\(.*\)$/\1\2 = Signal(_\2)/' >> ${OUT}
if ! grep '^const SIGPOLL ' ${OUT} >/dev/null 2>&1; then
  if grep '^const SIGIO ' ${OUT} > /dev/null 2>&1; then
    echo "const SIGPOLL = SIGIO" >> ${OUT}
  fi
fi
if ! grep '^const SIGCLD ' ${OUT} >/dev/null 2>&1; then
  if grep '^const SIGCHLD ' ${OUT} >/dev/null 2>&1; then
    echo "const SIGCLD = SIGCHLD" >> ${OUT}
  fi
fi

# The syscall numbers.  We force the names to upper case.
grep '^const _SYS_' gen-sysinfo.go | \
  sed -e 's/const _\(SYS_[^= ]*\).*$/\1/' | \
  while read sys; do
    sup=`echo $sys | tr abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ`
    echo "const $sup = _$sys" >> ${OUT}
  done

# The GNU/Linux support wants to use SYS_GETDENTS64 if available.
if ! grep '^const SYS_GETDENTS ' ${OUT} >/dev/null 2>&1; then
  echo "const SYS_GETDENTS = 0" >> ${OUT}
fi
if ! grep '^const SYS_GETDENTS64 ' ${OUT} >/dev/null 2>&1; then
  echo "const SYS_GETDENTS64 = 0" >> ${OUT}
fi

# Stat constants.
grep '^const _S_' gen-sysinfo.go | \
  sed -e 's/^\(const \)_\(S_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# Mmap constants.
grep '^const _PROT_' gen-sysinfo.go | \
  sed -e 's/^\(const \)_\(PROT_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
grep '^const _MAP_' gen-sysinfo.go | \
  sed -e 's/^\(const \)_\(MAP_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
grep '^const _MADV_' gen-sysinfo.go | \
  sed -e 's/^\(const \)_\(MADV_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
grep '^const _MCL_' gen-sysinfo.go | \
  sed -e 's/^\(const \)_\(MCL_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# Process status constants.
grep '^const _W' gen-sysinfo.go |
  sed -e 's/^\(const \)_\(W[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
# WSTOPPED was introduced in glibc 2.3.4.
if ! grep '^const _WSTOPPED = ' gen-sysinfo.go >/dev/null 2>&1; then
  if grep '^const _WUNTRACED = ' gen-sysinfo.go > /dev/null 2>&1; then
    echo 'const WSTOPPED = _WUNTRACED' >> ${OUT}
  else
    echo 'const WSTOPPED = 2' >> ${OUT}
  fi
fi
if grep '^const ___WALL = ' gen-sysinfo.go >/dev/null 2>&1 \
   && ! grep '^const _WALL = ' gen-sysinfo.go >/dev/null 2>&1; then
  echo 'const WALL = ___WALL' >> ${OUT}
fi

# Networking constants.
egrep '^const _(AF|ARPHRD|ETH|IN|SOCK|SOL|SO|IPPROTO|TCP|IP|IPV6)_' gen-sysinfo.go |
  sed -e 's/^\(const \)_\([^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
grep '^const _SOMAXCONN' gen-sysinfo.go |
  sed -e 's/^\(const \)_\(SOMAXCONN[^= ]*\)\(.*\)$/\1\2 = _\2/' \
    >> ${OUT}
grep '^const _SHUT_' gen-sysinfo.go |
  sed -e 's/^\(const \)_\(SHUT[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# The net package requires some const definitions.
for m in IP_PKTINFO IPV6_V6ONLY IPPROTO_IPV6 IPV6_JOIN_GROUP IPV6_LEAVE_GROUP IPV6_TCLASS SO_REUSEPORT; do
  if ! grep "^const $m " ${OUT} >/dev/null 2>&1; then
    echo "const $m = 0" >> ${OUT}
  fi
done
for m in SOCK_CLOEXEC SOCK_NONBLOCK; do
  if ! grep "^const $m " ${OUT} >/dev/null 2>&1; then
    echo "const $m = -1" >> ${OUT}
  fi
done

# The syscall package requires AF_LOCAL.
if ! grep '^const AF_LOCAL ' ${OUT} >/dev/null 2>&1; then
  if grep '^const AF_UNIX ' ${OUT} >/dev/null 2>&1; then
    echo "const AF_LOCAL = AF_UNIX" >> ${OUT}
  fi
fi

# pathconf constants.
grep '^const __PC' gen-sysinfo.go |
  sed -e 's/^\(const \)__\(PC[^= ]*\)\(.*\)$/\1\2 = __\2/' >> ${OUT}

# The PATH_MAX constant.
if grep '^const _PATH_MAX ' gen-sysinfo.go >/dev/null 2>&1; then
  echo 'const PathMax = _PATH_MAX' >> ${OUT}
fi

# epoll constants.
grep '^const _EPOLL' gen-sysinfo.go |
  sed -e 's/^\(const \)_\(EPOLL[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
# Make sure EPOLLRDHUP and EPOLL_CLOEXEC are defined.
if ! grep '^const EPOLLRDHUP' ${OUT} >/dev/null 2>&1; then
  echo "const EPOLLRDHUP = 0x2000" >> ${OUT}
fi
if ! grep '^const EPOLL_CLOEXEC' ${OUT} >/dev/null 2>&1; then
  echo "const EPOLL_CLOEXEC = 02000000" >> ${OUT}
fi

# Prctl constants.
grep '^const _PR_' gen-sysinfo.go |
  sed -e 's/^\(const \)_\(PR_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# Ptrace constants.
grep '^const _PTRACE' gen-sysinfo.go |
  sed -e 's/^\(const \)_\(PTRACE[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
# We need some ptrace options that are not defined in older versions
# of glibc.
if ! grep '^const PTRACE_SETOPTIONS' ${OUT} > /dev/null 2>&1; then
  echo "const PTRACE_SETOPTIONS = 0x4200" >> ${OUT}
fi
if ! grep '^const PTRACE_O_TRACESYSGOOD' ${OUT} > /dev/null 2>&1; then
  echo "const PTRACE_O_TRACESYSGOOD = 0x1" >> ${OUT}
fi
if ! grep '^const PTRACE_O_TRACEFORK' ${OUT} > /dev/null 2>&1; then
  echo "const PTRACE_O_TRACEFORK = 0x2" >> ${OUT}
fi
if ! grep '^const PTRACE_O_TRACEVFORK' ${OUT} > /dev/null 2>&1; then
  echo "const PTRACE_O_TRACEVFORK = 0x4" >> ${OUT}
fi
if ! grep '^const PTRACE_O_TRACECLONE' ${OUT} > /dev/null 2>&1; then
  echo "const PTRACE_O_TRACECLONE = 0x8" >> ${OUT}
fi
if ! grep '^const PTRACE_O_TRACEEXEC' ${OUT} > /dev/null 2>&1; then
  echo "const PTRACE_O_TRACEEXEC = 0x10" >> ${OUT}
fi
if ! grep '^const PTRACE_O_TRACEVFORKDONE' ${OUT} > /dev/null 2>&1; then
  echo "const PTRACE_O_TRACEVFORKDONE = 0x20" >> ${OUT}
fi
if ! grep '^const PTRACE_O_TRACEEXIT' ${OUT} > /dev/null 2>&1; then
  echo "const PTRACE_O_TRACEEXIT = 0x40" >> ${OUT}
fi
if ! grep '^const PTRACE_O_MASK' ${OUT} > /dev/null 2>&1; then
  echo "const PTRACE_O_MASK = 0x7f" >> ${OUT}
fi
if ! grep '^const _PTRACE_GETEVENTMSG' ${OUT} > /dev/null 2>&1; then
  echo "const PTRACE_GETEVENTMSG = 0x4201" >> ${OUT}
fi
if ! grep '^const PTRACE_EVENT_FORK' ${OUT} > /dev/null 2>&1; then
  echo "const PTRACE_EVENT_FORK = 1" >> ${OUT}
fi
if ! grep '^const PTRACE_EVENT_VFORK' ${OUT} > /dev/null 2>&1; then
  echo "const PTRACE_EVENT_VFORK = 2" >> ${OUT}
fi
if ! grep '^const PTRACE_EVENT_CLONE' ${OUT} > /dev/null 2>&1; then
  echo "const PTRACE_EVENT_CLONE = 3" >> ${OUT}
fi
if ! grep '^const PTRACE_EVENT_EXEC' ${OUT} > /dev/null 2>&1; then
  echo "const PTRACE_EVENT_EXEC = 4" >> ${OUT}
fi
if ! grep '^const PTRACE_EVENT_VFORK_DONE' ${OUT} > /dev/null 2>&1; then
  echo "const PTRACE_EVENT_VFORK_DONE = 5" >> ${OUT}
fi
if ! grep '^const PTRACE_EVENT_EXIT' ${OUT} > /dev/null 2>&1; then
  echo "const PTRACE_EVENT_EXIT = 6" >> ${OUT}
fi
if ! grep '^const _PTRACE_TRACEME' ${OUT} > /dev/null 2>&1; then
  echo "const _PTRACE_TRACEME = 0" >> ${OUT}
fi

# A helper function that prints a structure from gen-sysinfo.go with the first
# letter of the field names in upper case.  $1 is the name of structure.  If $2
# is not empty, the structure or type is renamed to $2.
upcase_fields () {
  name="$1"
  def=`grep "^type $name" gen-sysinfo.go`
  fields=`echo $def | sed -e 's/^[^{]*{\(.*\)}$/\1/'`
  prefix=`echo $def | sed -e 's/{.*//'`
  if test "$2" != ""; then
    prefix=`echo $prefix | sed -e "s/$1/$2/"`
  fi
  if test "$fields" != ""; then
    nfields=
    while test -n "$fields"; do
      field=`echo $fields | sed -e 's/^\([^;]*\);.*$/\1/'`
      fields=`echo $fields | sed -e 's/^[^;]*; *\(.*\)$/\1/'`
      # capitalize the next character.
      f=`echo $field | sed -e 's/^\(.\).*$/\1/'`
      r=`echo $field | sed -e 's/^.\(.*\)$/\1/'`
      f=`echo $f | tr abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ`
      field="$f$r"
      nfields="$nfields $field;"
    done
    echo "${prefix} {$nfields }"
  fi
}

# The registers returned by PTRACE_GETREGS.  This is probably
# GNU/Linux specific; it should do no harm if there is no
# _user_regs_struct.
regs=`grep '^type _user_regs_struct struct' gen-sysinfo.go || true`
if test "$regs" = ""; then
  # s390
  regs=`grep '^type __user_regs_struct struct' gen-sysinfo.go || true`
  if test "$regs" != ""; then
    # Substructures of __user_regs_struct on s390
    upcase_fields "__user_psw_struct" "PtracePsw" >> ${OUT} || true
    upcase_fields "__user_fpregs_struct" "PtraceFpregs" >> ${OUT} || true
    upcase_fields "__user_per_struct" "PtracePer" >> ${OUT} || true
  fi
fi
if test "$regs" != ""; then
  regs=`echo $regs |
    sed -e 's/type __*user_regs_struct struct //' -e 's/[{}]//g'`
  regs=`echo $regs | sed -e s'/^ *//'`
  nregs=
  while test -n "$regs"; do
    field=`echo $regs | sed -e 's/^\([^;]*\);.*$/\1/'`
    regs=`echo $regs | sed -e 's/^[^;]*; *\(.*\)$/\1/'`
    # Capitalize the first character of the field.
    f=`echo $field | sed -e 's/^\(.\).*$/\1/'`
    r=`echo $field | sed -e 's/^.\(.*\)$/\1/'`
    f=`echo $f | tr abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ`
    field="$f$r"
    field=`echo "$field" | sed \
      -e 's/__user_psw_struct/PtracePsw/' \
      -e 's/__user_fpregs_struct/PtraceFpregs/' \
      -e 's/__user_per_struct/PtracePer/'`
    nregs="$nregs $field;"
  done
  echo "type PtraceRegs struct {$nregs }" >> ${OUT}
fi

# Some basic types.
echo 'type Size_t _size_t' >> ${OUT}
echo "type Ssize_t _ssize_t" >> ${OUT}
if grep '^const _HAVE_OFF64_T = ' gen-sysinfo.go > /dev/null 2>&1; then
  echo "type Offset_t _off64_t" >> ${OUT}
else
  echo "type Offset_t _off_t" >> ${OUT}
fi
echo "type Mode_t _mode_t" >> ${OUT}
echo "type Pid_t _pid_t" >> ${OUT}
echo "type Uid_t _uid_t" >> ${OUT}
echo "type Gid_t _gid_t" >> ${OUT}
echo "type Socklen_t _socklen_t" >> ${OUT}

# The C int type.
sizeof_int=`grep '^const ___SIZEOF_INT__ = ' gen-sysinfo.go | sed -e 's/.*= //'`
if test "$sizeof_int" = "4"; then
  echo "type _C_int int32" >> ${OUT}
  echo "type _C_uint uint32" >> ${OUT}
elif test "$sizeof_int" = "8"; then
  echo "type _C_int int64" >> ${OUT}
  echo "type _C_uint uint64" >> ${OUT}
else
  echo 1>&2 "mksysinfo.sh: could not determine size of int (got $sizeof_int)"
  exit 1
fi

# The C long type, needed because that is the type that ptrace returns.
sizeof_long=`grep '^const ___SIZEOF_LONG__ = ' gen-sysinfo.go | sed -e 's/.*= //'`
if test "$sizeof_long" = "4"; then
  echo "type _C_long int32" >> ${OUT}
elif test "$sizeof_long" = "8"; then
  echo "type _C_long int64" >> ${OUT}
else
  echo 1>&2 "mksysinfo.sh: could not determine size of long (got $sizeof_long)"
  exit 1
fi

# Solaris 2 needs _u?pad128_t, but its default definition in terms of long
# double is commented by -fdump-go-spec.
if grep "^// type _pad128_t" gen-sysinfo.go > /dev/null 2>&1; then
  echo "type _pad128_t struct { _l [4]int32; }" >> ${OUT}
fi
if grep "^// type _upad128_t" gen-sysinfo.go > /dev/null 2>&1; then
  echo "type _upad128_t struct { _l [4]uint32; }" >> ${OUT}
fi

# The time_t type.
if grep '^type _time_t ' gen-sysinfo.go > /dev/null 2>&1; then
  echo 'type Time_t _time_t' >> ${OUT}
fi

# The time structures need special handling: we need to name the
# types, so that we can cast integers to the right types when
# assigning to the structures.
timeval=`grep '^type _timeval ' gen-sysinfo.go`
timeval_sec=`echo $timeval | sed -n -e 's/^.*tv_sec \([^ ]*\);.*$/\1/p'`
timeval_usec=`echo $timeval | sed -n -e 's/^.*tv_usec \([^ ]*\);.*$/\1/p'`
echo "type Timeval_sec_t $timeval_sec" >> ${OUT}
echo "type Timeval_usec_t $timeval_usec" >> ${OUT}
echo $timeval | \
  sed -e 's/type _timeval /type Timeval /' \
      -e 's/tv_sec *[a-zA-Z0-9_]*/Sec Timeval_sec_t/' \
      -e 's/tv_usec *[a-zA-Z0-9_]*/Usec Timeval_usec_t/' >> ${OUT}
timespec=`grep '^type _timespec ' gen-sysinfo.go || true`
if test "$timespec" = ""; then
  # IRIX 6.5 has __timespec instead.
  timespec=`grep '^type ___timespec ' gen-sysinfo.go || true`
fi
timespec_sec=`echo $timespec | sed -n -e 's/^.*tv_sec \([^ ]*\);.*$/\1/p'`
timespec_nsec=`echo $timespec | sed -n -e 's/^.*tv_nsec \([^ ]*\);.*$/\1/p'`
echo "type Timespec_sec_t $timespec_sec" >> ${OUT}
echo "type Timespec_nsec_t $timespec_nsec" >> ${OUT}
echo $timespec | \
  sed -e 's/^type ___timespec /type Timespec /' \
      -e 's/^type _timespec /type Timespec /' \
      -e 's/tv_sec *[a-zA-Z0-9_]*/Sec Timespec_sec_t/' \
      -e 's/tv_nsec *[a-zA-Z0-9_]*/Nsec Timespec_nsec_t/' >> ${OUT}

timestruc=`grep '^type _timestruc_t ' gen-sysinfo.go || true`
if test "$timestruc" != ""; then
  timestruc_sec=`echo $timestruc | sed -n -e 's/^.*tv_sec \([^ ]*\);.*$/\1/p'`
  timestruc_nsec=`echo $timestruc | sed -n -e 's/^.*tv_nsec \([^ ]*\);.*$/\1/p'`
  echo "type Timestruc_sec_t $timestruc_sec" >> ${OUT}
  echo "type Timestruc_nsec_t $timestruc_nsec" >> ${OUT}
  echo $timestruc | \
    sed -e 's/^type _timestruc_t /type Timestruc /' \
        -e 's/tv_sec *[a-zA-Z0-9_]*/Sec Timestruc_sec_t/' \
        -e 's/tv_nsec *[a-zA-Z0-9_]*/Nsec Timestruc_nsec_t/' >> ${OUT}
fi

# The tms struct.
grep '^type _tms ' gen-sysinfo.go | \
    sed -e 's/type _tms/type Tms/' \
      -e 's/tms_utime/Utime/' \
      -e 's/tms_stime/Stime/' \
      -e 's/tms_cutime/Cutime/' \
      -e 's/tms_cstime/Cstime/' \
    >> ${OUT}

# The stat type.
# Prefer largefile variant if available.
stat=`grep '^type _stat64 ' gen-sysinfo.go || true`
if test "$stat" != ""; then
  grep '^type _stat64 ' gen-sysinfo.go
else
  grep '^type _stat ' gen-sysinfo.go
fi | sed -e 's/type _stat64/type Stat_t/' \
         -e 's/type _stat/type Stat_t/' \
         -e 's/st_dev/Dev/' \
         -e 's/st_ino/Ino/g' \
         -e 's/st_nlink/Nlink/' \
         -e 's/st_mode/Mode/' \
         -e 's/st_uid/Uid/' \
         -e 's/st_gid/Gid/' \
         -e 's/st_rdev/Rdev/' \
         -e 's/st_size/Size/' \
         -e 's/st_blksize/Blksize/' \
         -e 's/st_blocks/Blocks/' \
         -e 's/st_atim/Atim/' \
         -e 's/st_mtim/Mtim/' \
         -e 's/st_ctim/Ctim/' \
         -e 's/\([^a-zA-Z0-9_]\)_timeval\([^a-zA-Z0-9_]\)/\1Timeval\2/g' \
         -e 's/\([^a-zA-Z0-9_]\)_timespec_t\([^a-zA-Z0-9_]\)/\1Timespec\2/g' \
         -e 's/\([^a-zA-Z0-9_]\)_timespec\([^a-zA-Z0-9_]\)/\1Timespec\2/g' \
         -e 's/\([^a-zA-Z0-9_]\)_timestruc_t\([^a-zA-Z0-9_]\)/\1Timestruc\2/g' \
         -e 's/Godump_[0-9] struct { \([^;]*;\) };/\1/g' \
       >> ${OUT}

# The directory searching types.
# Prefer largefile variant if available.
dirent=`grep '^type _dirent64 ' gen-sysinfo.go || true`
if test "$dirent" != ""; then
  grep '^type _dirent64 ' gen-sysinfo.go
else
  grep '^type _dirent ' gen-sysinfo.go
fi | sed -e 's/type _dirent64/type Dirent/' \
         -e 's/type _dirent/type Dirent/' \
         -e 's/d_name \[0+1\]/d_name [0+256]/' \
         -e 's/d_name/Name/' \
         -e 's/]int8/]byte/' \
         -e 's/d_ino/Ino/' \
         -e 's/d_off/Off/' \
         -e 's/d_reclen/Reclen/' \
         -e 's/d_type/Type/' \
      >> ${OUT}
echo "type DIR _DIR" >> ${OUT}

# Values for d_type field in dirent.
grep '^const _DT_' gen-sysinfo.go |
  sed -e 's/^\(const \)_\(DT_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# The rusage struct.
rusage=`grep '^type _rusage struct' gen-sysinfo.go`
if test "$rusage" != ""; then
  # Remove anonymous unions from GNU/Linux <bits/resource.h>.
  rusage=`echo $rusage | sed -e 's/Godump_[0-9][0-9]* struct {\([^}]*\)};/\1/g'`
  rusage=`echo $rusage | sed -e 's/type _rusage struct //' -e 's/[{}]//g'`
  rusage=`echo $rusage | sed -e 's/^ *//'`
  nrusage=
  while test -n "$rusage"; do
    field=`echo $rusage | sed -e 's/^\([^;]*\);.*$/\1/'`
    rusage=`echo $rusage | sed -e 's/^[^;]*; *\(.*\)$/\1/'`
    # Drop the leading ru_, capitalize the next character.
    field=`echo $field | sed -e 's/^ru_//'`
    f=`echo $field | sed -e 's/^\(.\).*$/\1/'`
    r=`echo $field | sed -e 's/^.\(.*\)$/\1/'`
    f=`echo $f | tr abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ`
    # Fix _timeval _timespec, and _timestruc_t.
    r=`echo $r | sed -e s'/ _timeval$/ Timeval/'`
    r=`echo $r | sed -e s'/ _timespec$/ Timespec/'`
    r=`echo $r | sed -e s'/ _timestruc_t$/ Timestruc/'`
    field="$f$r"
    nrusage="$nrusage $field;"
  done
  echo "type Rusage struct {$nrusage }" >> ${OUT}
else
  echo "type Rusage struct {}" >> ${OUT}
fi

# The RUSAGE constants.
grep '^const _RUSAGE_' gen-sysinfo.go | \
  sed -e 's/^\(const \)_\(RUSAGE_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# The utsname struct.
grep '^type _utsname ' gen-sysinfo.go | \
    sed -e 's/_utsname/Utsname/' \
      -e 's/sysname/Sysname/' \
      -e 's/nodename/Nodename/' \
      -e 's/release/Release/' \
      -e 's/version/Version/' \
      -e 's/machine/Machine/' \
      -e 's/domainname/Domainname/' \
    >> ${OUT}

# The iovec struct.
iovec=`grep '^type _iovec ' gen-sysinfo.go`
iovec_len=`echo $iovec | sed -n -e 's/^.*iov_len \([^ ]*\);.*$/\1/p'`
echo "type Iovec_len_t $iovec_len" >> ${OUT}
echo $iovec | \
    sed -e 's/_iovec/Iovec/' \
      -e 's/iov_base/Base/' \
      -e 's/iov_len *[a-zA-Z0-9_]*/Len Iovec_len_t/' \
    >> ${OUT}

# The msghdr struct.
msghdr=`grep '^type _msghdr ' gen-sysinfo.go`
msghdr_controllen=`echo $msghdr | sed -n -e 's/^.*msg_controllen \([^ ]*\);.*$/\1/p'`
echo "type Msghdr_controllen_t $msghdr_controllen" >> ${OUT}
echo $msghdr | \
    sed -e 's/_msghdr/Msghdr/' \
      -e 's/msg_name/Name/' \
      -e 's/msg_namelen/Namelen/' \
      -e 's/msg_iov/Iov/' \
      -e 's/msg_iovlen/Iovlen/' \
      -e 's/_iovec/Iovec/' \
      -e 's/msg_control/Control/' \
      -e 's/msg_controllen *[a-zA-Z0-9_]*/Controllen Msghdr_controllen_t/' \
      -e 's/msg_flags/Flags/' \
    >> ${OUT}

# The MSG_ flags for Msghdr.
grep '^const _MSG_' gen-sysinfo.go | \
  sed -e 's/^\(const \)_\(MSG_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# The cmsghdr struct.
cmsghdr=`grep '^type _cmsghdr ' gen-sysinfo.go`
if test -n "$cmsghdr"; then
  cmsghdr_len=`echo $cmsghdr | sed -n -e 's/^.*cmsg_len \([^ ]*\);.*$/\1/p'`
  echo "type Cmsghdr_len_t $cmsghdr_len" >> ${OUT}
  echo "$cmsghdr" | \
      sed -e 's/_cmsghdr/Cmsghdr/' \
        -e 's/cmsg_len *[a-zA-Z0-9_]*/Len Cmsghdr_len_t/' \
        -e 's/cmsg_level/Level/' \
        -e 's/cmsg_type/Type/' \
        -e 's/\[\]/[0]/' \
      >> ${OUT}
fi

# The SCM_ flags for Cmsghdr.
grep '^const _SCM_' gen-sysinfo.go | \
  sed -e 's/^\(const \)_\(SCM_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# The ucred struct.
upcase_fields "_ucred" "Ucred" >> ${OUT} || true

# The ip_mreq struct.
grep '^type _ip_mreq ' gen-sysinfo.go | \
    sed -e 's/_ip_mreq/IPMreq/' \
      -e 's/imr_multiaddr/Multiaddr/' \
      -e 's/imr_interface/Interface/' \
      -e 's/_in_addr/[4]byte/g' \
    >> ${OUT}

# We need IPMreq to compile the net package.
if ! grep 'type IPMreq ' ${OUT} >/dev/null 2>&1; then
  echo 'type IPMreq struct { Multiaddr [4]byte; Interface [4]byte; }' >> ${OUT}
fi

# The ipv6_mreq struct.
grep '^type _ipv6_mreq ' gen-sysinfo.go | \
    sed -e 's/_ipv6_mreq/IPv6Mreq/' \
      -e 's/ipv6mr_multiaddr/Multiaddr/' \
      -e 's/ipv6mr_interface/Interface/' \
      -e 's/_in6_addr/[16]byte/' \
    >> ${OUT}

# We need IPv6Mreq to compile the net package.
if ! grep 'type IPv6Mreq ' ${OUT} >/dev/null 2>&1; then
  echo 'type IPv6Mreq struct { Multiaddr [16]byte; Interface uint32; }' >> ${OUT}
fi

# The ip_mreqn struct.
grep '^type _ip_mreqn ' gen-sysinfo.go | \
    sed -e 's/_ip_mreqn/IPMreqn/' \
      -e 's/imr_multiaddr/Multiaddr/' \
      -e 's/imr_address/Address/' \
      -e 's/imr_ifindex/Ifindex/' \
      -e 's/_in_addr/[4]byte/g' \
    >> ${OUT}

# We need IPMreq to compile the net package.
if ! grep 'type IPMreqn ' ${OUT} >/dev/null 2>&1; then
  echo 'type IPMreqn struct { Multiaddr [4]byte; Interface [4]byte; Ifindex int32 }' >> ${OUT}
fi

# The icmp6_filter struct.
grep '^type _icmp6_filter ' gen-sysinfo.go | \
    sed -e 's/_icmp6_filter/ICMPv6Filter/' \
      -e 's/data/Data/' \
      -e 's/filt/Filt/' \
    >> ${OUT}

# We need ICMPv6Filter to compile the syscall package.
if ! grep 'type ICMPv6Filter ' ${OUT} > /dev/null 2>&1; then
  echo 'type ICMPv6Filter struct { Data [8]uint32 }' >> ${OUT}
fi

# Try to guess the type to use for fd_set.
fd_set=`grep '^type _fd_set ' gen-sysinfo.go || true`
fds_bits_type="_C_long"
if test "$fd_set" != ""; then
    fds_bits_type=`echo $fd_set | sed -e 's/.*[]]\([^;]*\); }$/\1/'`
fi
echo "type fds_bits_type $fds_bits_type" >> ${OUT}

# The addrinfo struct.
grep '^type _addrinfo ' gen-sysinfo.go | \
    sed -e 's/_addrinfo/Addrinfo/g' \
      -e 's/ ai_/ Ai_/g' \
    >> ${OUT}

# The addrinfo and nameinfo flags and errors.
grep '^const _AI_' gen-sysinfo.go | \
  sed -e 's/^\(const \)_\(AI_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
grep '^const _EAI_' gen-sysinfo.go | \
  sed -e 's/^\(const \)_\(EAI_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
grep '^const _NI_' gen-sysinfo.go | \
  sed -e 's/^\(const \)_\(NI_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# The passwd struct.
grep '^type _passwd ' gen-sysinfo.go | \
    sed -e 's/_passwd/Passwd/' \
      -e 's/ pw_/ Pw_/g' \
    >> ${OUT}

# The ioctl flags for the controlling TTY.
grep '^const _TIOC' gen-sysinfo.go | \
    grep -v '_val =' | \
    sed -e 's/^\(const \)_\(TIOC[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
grep '^const _TUNSET' gen-sysinfo.go | \
    grep -v '_val =' | \
    sed -e 's/^\(const \)_\(TUNSET[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
# We need TIOCGWINSZ.
if ! grep '^const TIOCGWINSZ' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TIOCGWINSZ_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TIOCGWINSZ = _TIOCGWINSZ_val' >> ${OUT}
  fi
fi
if ! grep '^const TIOCSWINSZ' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TIOCSWINSZ_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TIOCSWINSZ = _TIOCSWINSZ_val' >> ${OUT}
  fi
fi
if ! grep '^const TIOCNOTTY' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TIOCNOTTY_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TIOCNOTTY = _TIOCNOTTY_val' >> ${OUT}
  fi
fi
if ! grep '^const TIOCSCTTY' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TIOCSCTTY_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TIOCSCTTY = _TIOCSCTTY_val' >> ${OUT}
  fi
fi
if ! grep '^const TIOCGPGRP' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TIOCGPGRP_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TIOCGPGRP = _TIOCGPGRP_val' >> ${OUT}
  fi
fi
if ! grep '^const TIOCSPGRP' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TIOCSPGRP_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TIOCSPGRP = _TIOCSPGRP_val' >> ${OUT}
  fi
fi
if ! grep '^const TIOCGPTN' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TIOCGPTN_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TIOCGPTN = _TIOCGPTN_val' >> ${OUT}
  fi
fi
if ! grep '^const TIOCSPTLCK' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TIOCSPTLCK_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TIOCSPTLCK = _TIOCSPTLCK_val' >> ${OUT}
  fi
fi
if ! grep '^const TIOCGDEV' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TIOCGDEV_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TIOCGDEV = _TIOCGDEV_val' >> ${OUT}
  fi
fi
if ! grep '^const TIOCSIG' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TIOCSIG_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TIOCSIG = _TIOCSIG_val' >> ${OUT}
  fi
fi

if ! grep '^const TUNSETNOCSUM' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TUNSETNOCSUM_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TUNSETNOCSUM = _TUNSETNOCSUM_val' >> ${OUT}
  fi
fi

if ! grep '^const TUNSETDEBUG' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TUNSETDEBUG_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TUNSETDEBUG = _TUNSETDEBUG_val' >> ${OUT}
  fi
fi

if ! grep '^const TUNSETIFF' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TUNSETIFF_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TUNSETIFF = _TUNSETIFF_val' >> ${OUT}
  fi
fi

if ! grep '^const TUNSETPERSIST' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TUNSETPERSIST_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TUNSETPERSIST = _TUNSETPERSIST_val' >> ${OUT}
  fi
fi

if ! grep '^const TUNSETOWNER' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TUNSETOWNER_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TUNSETOWNER = _TUNSETOWNER_val' >> ${OUT}
  fi
fi

if ! grep '^const TUNSETLINK' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TUNSETLINK_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TUNSETLINK = _TUNSETLINK_val' >> ${OUT}
  fi
fi

if ! grep '^const TUNSETGROUP' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TUNSETGROUP_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TUNSETGROUP = _TUNSETGROUP_val' >> ${OUT}
  fi
fi

if ! grep '^const TUNGETFEATURES' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TUNGETFEATURES_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TUNGETFEATURES = _TUNGETFEATURES_val' >> ${OUT}
  fi
fi

if ! grep '^const TUNSETOFFLOAD' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TUNSETOFFLOAD_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TUNSETOFFLOAD = _TUNSETOFFLOAD_val' >> ${OUT}
  fi
fi

if ! grep '^const TUNSETTXFILTER' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TUNSETTXFILTER_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TUNSETTXFILTER = _TUNSETTXFILTER_val' >> ${OUT}
  fi
fi

if ! grep '^const TUNGETIFF' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TUNGETIFF_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TUNGETIFF = _TUNGETIFF_val' >> ${OUT}
  fi
fi

if ! grep '^const TUNGETSNDBUF' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TUNGETSNDBUF_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TUNGETSNDBUF = _TUNGETSNDBUF_val' >> ${OUT}
  fi
fi

if ! grep '^const TUNSETSNDBUF' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TUNSETSNDBUF_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TUNSETSNDBUF = _TUNSETSNDBUF_val' >> ${OUT}
  fi
fi

if ! grep '^const TUNATTACHFILTER' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TUNATTACHFILTER_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TUNATTACHFILTER = _TUNATTACHFILTER_val' >> ${OUT}
  fi
fi

if ! grep '^const TUNDETACHFILTER' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TUNDETACHFILTER_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TUNDETACHFILTER = _TUNDETACHFILTER_val' >> ${OUT}
  fi
fi

if ! grep '^const TUNGETVNETHDRSZ' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TUNGETVNETHDRSZ_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TUNGETVNETHDRSZ = _TUNGETVNETHDRSZ_val' >> ${OUT}
  fi
fi

if ! grep '^const TUNSETVNETHDRSZ' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TUNSETVNETHDRSZ_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TUNSETVNETHDRSZ = _TUNSETVNETHDRSZ_val' >> ${OUT}
  fi
fi

if ! grep '^const TUNSETQUEUE' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TUNSETQUEUE_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TUNSETQUEUE = _TUNSETQUEUE_val' >> ${OUT}
  fi
fi


if ! grep '^const TUNSETIFINDEX' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TUNSETIFINDEX_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TUNSETIFINDEX = _TUNSETIFINDEX_val' >> ${OUT}
  fi
fi

if ! grep '^const TUNGETFILTER' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TUNGETFILTER_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TUNGETFILTER = _TUNGETFILTER_val' >> ${OUT}
  fi
fi



# The ioctl flags for terminal control
grep '^const _TC[GS]ET' gen-sysinfo.go | grep -v _val | \
    sed -e 's/^\(const \)_\(TC[GS]ET[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
if ! grep '^const TCGETS' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TCGETS_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TCGETS = _TCGETS_val' >> ${OUT}
  fi
fi
if ! grep '^const TCSETS' ${OUT} >/dev/null 2>&1; then
  if grep '^const _TCSETS_val' ${OUT} >/dev/null 2>&1; then
    echo 'const TCSETS = _TCSETS_val' >> ${OUT}
  fi
fi

# ioctl constants.  Might fall back to 0 if TIOCNXCL is missing, too, but
# needs handling in syscalls.exec.go.
if ! grep '^const _TIOCSCTTY ' gen-sysinfo.go >/dev/null 2>&1; then
  if grep '^const _TIOCNXCL ' gen-sysinfo.go >/dev/null 2>&1; then
    echo "const TIOCSCTTY = TIOCNXCL" >> ${OUT}
  fi
fi

# The nlmsghdr struct.
grep '^type _nlmsghdr ' gen-sysinfo.go | \
    sed -e 's/_nlmsghdr/NlMsghdr/' \
      -e 's/nlmsg_len/Len/' \
      -e 's/nlmsg_type/Type/' \
      -e 's/nlmsg_flags/Flags/' \
      -e 's/nlmsg_seq/Seq/' \
      -e 's/nlmsg_pid/Pid/' \
    >> ${OUT}

# The nlmsg flags and operators.
grep '^const _NLM' gen-sysinfo.go | \
    sed -e 's/^\(const \)_\(NLM[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# NLMSG_HDRLEN is defined as an expression using sizeof.
if ! grep '^const NLMSG_HDRLEN' ${OUT} > /dev/null 2>&1; then
  if grep '^const _sizeof_nlmsghdr ' ${OUT} > /dev/null 2>&1; then
    echo 'const NLMSG_HDRLEN = (_sizeof_nlmsghdr + (NLMSG_ALIGNTO-1)) &^ (NLMSG_ALIGNTO-1)' >> ${OUT}
  fi
fi

# The rtmsg struct.
grep '^type _rtmsg ' gen-sysinfo.go | \
    sed -e 's/_rtmsg/RtMsg/' \
      -e 's/rtm_family/Family/' \
      -e 's/rtm_dst_len/Dst_len/' \
      -e 's/rtm_src_len/Src_len/' \
      -e 's/rtm_tos/Tos/' \
      -e 's/rtm_table/Table/' \
      -e 's/rtm_protocol/Protocol/' \
      -e 's/rtm_scope/Scope/' \
      -e 's/rtm_type/Type/' \
      -e 's/rtm_flags/Flags/' \
    >> ${OUT}

# The rtgenmsg struct.
grep '^type _rtgenmsg ' gen-sysinfo.go | \
    sed -e 's/_rtgenmsg/RtGenmsg/' \
      -e 's/rtgen_family/Family/' \
    >> ${OUT}

# The routing message flags.
grep '^const _RT_' gen-sysinfo.go | \
    sed -e 's/^\(const \)_\(RT_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
grep '^const _RTA' gen-sysinfo.go | \
    sed -e 's/^\(const \)_\(RTA[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
grep '^const _RTF' gen-sysinfo.go | \
    sed -e 's/^\(const \)_\(RTF[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
grep '^const _RTCF' gen-sysinfo.go | \
    sed -e 's/^\(const \)_\(RTCF[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
grep '^const _RTM' gen-sysinfo.go | \
    sed -e 's/^\(const \)_\(RTM[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
grep '^const _RTN' gen-sysinfo.go | \
    sed -e 's/^\(const \)_\(RTN[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
grep '^const _RTPROT' gen-sysinfo.go | \
    sed -e 's/^\(const \)_\(RTPROT[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# The ifinfomsg struct.
grep '^type _ifinfomsg ' gen-sysinfo.go | \
    sed -e 's/_ifinfomsg/IfInfomsg/' \
      -e 's/ifi_family/Family/' \
      -e 's/ifi_type/Type/' \
      -e 's/ifi_index/Index/' \
      -e 's/ifi_flags/Flags/' \
      -e 's/ifi_change/Change/' \
    >> ${OUT}

# The interface information types and flags.
grep '^const _IFA' gen-sysinfo.go | \
    sed -e 's/^\(const \)_\(IFA[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
grep '^const _IFLA' gen-sysinfo.go | \
    sed -e 's/^\(const \)_\(IFLA[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
grep '^const _IFF' gen-sysinfo.go | \
    sed -e 's/^\(const \)_\(IFF[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
grep '^const _IFNAMSIZ' gen-sysinfo.go | \
    sed -e 's/^\(const \)_\(IFNAMSIZ[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
grep '^const _SIOC' gen-sysinfo.go |
    sed -e 's/^\(const \)_\(SIOC[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# The ifaddrmsg struct.
grep '^type _ifaddrmsg ' gen-sysinfo.go | \
    sed -e 's/_ifaddrmsg/IfAddrmsg/' \
      -e 's/ifa_family/Family/' \
      -e 's/ifa_prefixlen/Prefixlen/' \
      -e 's/ifa_flags/Flags/' \
      -e 's/ifa_scope/Scope/' \
      -e 's/ifa_index/Index/' \
    >> ${OUT}

# The rtattr struct.
grep '^type _rtattr ' gen-sysinfo.go | \
    sed -e 's/_rtattr/RtAttr/' \
      -e 's/rta_len/Len/' \
      -e 's/rta_type/Type/' \
    >> ${OUT}

# The in_pktinfo struct.
grep '^type _in_pktinfo ' gen-sysinfo.go | \
    sed -e 's/_in_pktinfo/Inet4Pktinfo/' \
      -e 's/ipi_ifindex/Ifindex/' \
      -e 's/ipi_spec_dst/Spec_dst/' \
      -e 's/ipi_addr/Addr/' \
      -e 's/_in_addr/[4]byte/g' \
    >> ${OUT}

# The in6_pktinfo struct.
grep '^type _in6_pktinfo ' gen-sysinfo.go | \
    sed -e 's/_in6_pktinfo/Inet6Pktinfo/' \
      -e 's/ipi6_addr/Addr/' \
      -e 's/ipi6_ifindex/Ifindex/' \
      -e 's/_in6_addr/[16]byte/' \
    >> ${OUT}

# The termios struct.
grep '^type _termios ' gen-sysinfo.go | \
    sed -e 's/_termios/Termios/' \
      -e 's/c_iflag/Iflag/' \
      -e 's/c_oflag/Oflag/' \
      -e 's/c_cflag/Cflag/' \
      -e 's/c_lflag/Lflag/' \
      -e 's/c_line/Line/' \
      -e 's/c_cc/Cc/' \
      -e 's/c_ispeed/Ispeed/' \
      -e 's/c_ospeed/Ospeed/' \
    >> ${OUT}

# The termios constants.
for n in IGNBRK BRKINT IGNPAR PARMRK INPCK ISTRIP INLCR IGNCR ICRNL IUCLC \
    IXON IXANY IXOFF IMAXBEL IUTF8 OPOST OLCUC ONLCR OCRNL ONOCR ONLRET \
    OFILL OFDEL NLDLY NL0 NL1 CRDLY CR0 CR1 CR2 CR3 CS5 CS6 CS7 CS8 TABDLY \
    BSDLY VTDLY FFDLY CBAUD CBAUDEX CSIZE CSTOPB CREAD PARENB PARODD HUPCL \
    CLOCAL LOBLK CIBAUD CMSPAR CRTSCTS ISIG ICANON XCASE ECHO ECHOE ECHOK \
    ECHONL ECHOCTL ECHOPRT ECHOKE DEFECHO FLUSHO NOFLSH TOSTOP PENDIN IEXTEN \
    VINTR VQUIT VERASE VKILL VEOF VMIN VEOL VTIME VEOL2 VSWTCH VSTART VSTOP \
    VSUSP VDSUSP VLNEXT VWERASE VREPRINT VDISCARD VSTATUS TCSANOW TCSADRAIN \
    TCSAFLUSH TCIFLUSH TCOFLUSH TCIOFLUSH TCOOFF TCOON TCIOFF TCION B0 B50 \
    B75 B110 B134 B150 B200 B300 B600 B1200 B1800 B2400 B4800 B9600 B19200 \
    B38400 B57600 B115200 B230400 B460800 B500000 B576000 B921600 B1000000 \
    B1152000 B1500000 B2000000 B2500000 B3000000 B3500000 B4000000; do

    grep "^const _$n " gen-sysinfo.go | \
	sed -e 's/^\(const \)_\([^=]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
done

# The mount flags
grep '^const _MNT_' gen-sysinfo.go |
    sed -e 's/^\(const \)_\(MNT_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
grep '^const _MS_' gen-sysinfo.go |
    sed -e 's/^\(const \)_\(MS_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# The fallocate flags.
grep '^const _FALLOC_' gen-sysinfo.go |
    sed -e 's/^\(const \)_\(FALLOC_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# The statfs struct.
# Prefer largefile variant if available.
statfs=`grep '^type _statfs64 ' gen-sysinfo.go || true`
if test "$statfs" != ""; then
  grep '^type _statfs64 ' gen-sysinfo.go
else
  grep '^type _statfs ' gen-sysinfo.go
fi | sed -e 's/type _statfs64/type Statfs_t/' \
	 -e 's/type _statfs/type Statfs_t/' \
	 -e 's/f_type/Type/' \
	 -e 's/f_bsize/Bsize/' \
	 -e 's/f_blocks/Blocks/' \
	 -e 's/f_bfree/Bfree/' \
	 -e 's/f_bavail/Bavail/' \
	 -e 's/f_files/Files/' \
	 -e 's/f_ffree/Ffree/' \
	 -e 's/f_fsid/Fsid/' \
	 -e 's/f_namelen/Namelen/' \
	 -e 's/f_frsize/Frsize/' \
	 -e 's/f_flags/Flags/' \
	 -e 's/f_spare/Spare/' \
    >> ${OUT}

# The timex struct.
timex=`grep '^type _timex ' gen-sysinfo.go || true`
if test "$timex" = ""; then
  timex=`grep '^// type _timex ' gen-sysinfo.go || true`
  if test "$timex" != ""; then
    timex=`echo $timex | sed -e 's|// ||' -e 's/INVALID-bit-field/int32/g'`
  fi
fi
if test "$timex" != ""; then
  echo "$timex" | \
    sed -e 's/_timex/Timex/' \
      -e 's/modes/Modes/' \
      -e 's/offset/Offset/' \
      -e 's/freq/Freq/' \
      -e 's/maxerror/Maxerror/' \
      -e 's/esterror/Esterror/' \
      -e 's/status/Status/' \
      -e 's/constant/Constant/' \
      -e 's/precision/Precision/' \
      -e 's/tolerance/Tolerance/' \
      -e 's/ time / Time /' \
      -e 's/tick/Tick/' \
      -e 's/ppsfreq/Ppsfreq/' \
      -e 's/jitter/Jitter/' \
      -e 's/shift/Shift/' \
      -e 's/stabil/Stabil/' \
      -e 's/jitcnt/Jitcnt/' \
      -e 's/calcnt/Calcnt/' \
      -e 's/errcnt/Errcnt/' \
      -e 's/stbcnt/Stbcnt/' \
      -e 's/tai/Tai/' \
      -e 's/_timeval/Timeval/' \
    >> ${OUT}
fi

# The rlimit struct.
grep '^type _rlimit ' gen-sysinfo.go | \
    sed -e 's/_rlimit/Rlimit/' \
      -e 's/rlim_cur/Cur/' \
      -e 's/rlim_max/Max/' \
    >> ${OUT}

# The RLIMIT constants.
grep '^const _RLIMIT_' gen-sysinfo.go |
    sed -e 's/^\(const \)_\(RLIMIT_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
grep '^const _RLIM_' gen-sysinfo.go |
    sed -e 's/^\(const \)_\(RLIM_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# The sysinfo struct.
grep '^type _sysinfo ' gen-sysinfo.go | \
    sed -e 's/_sysinfo/Sysinfo_t/' \
      -e 's/uptime/Uptime/' \
      -e 's/loads/Loads/' \
      -e 's/totalram/Totalram/' \
      -e 's/freeram/Freeram/' \
      -e 's/sharedram/Sharedram/' \
      -e 's/bufferram/Bufferram/' \
      -e 's/totalswap/Totalswap/' \
      -e 's/freeswap/Freeswap/' \
      -e 's/procs/Procs/' \
      -e 's/totalhigh/Totalhigh/' \
      -e 's/freehigh/Freehigh/' \
      -e 's/mem_unit/Unit/' \
    >> ${OUT}

# The ustat struct.
grep '^type _ustat ' gen-sysinfo.go | \
    sed -e 's/_ustat/Ustat_t/' \
      -e 's/f_tfree/Tfree/' \
      -e 's/f_tinode/Tinoe/' \
      -e 's/f_fname/Fname/' \
      -e 's/f_fpack/Fpack/' \
    >> ${OUT}
# Force it to be defined, as on some older GNU/Linux systems the
# header file fails when using with <linux/filter.h>.
if ! grep 'type _ustat ' gen-sysinfo.go >/dev/null 2>&1; then
  echo 'type Ustat_t struct { Tfree int32; Tinoe uint64; Fname [5+1]int8; Fpack [5+1]int8; }' >> ${OUT}
fi

# The utimbuf struct.
grep '^type _utimbuf ' gen-sysinfo.go | \
    sed -e 's/_utimbuf/Utimbuf/' \
      -e 's/actime/Actime/' \
      -e 's/modtime/Modtime/' \
    >> ${OUT}

# The LOCK flags for flock.
grep '^const _LOCK_' gen-sysinfo.go |
    sed -e 's/^\(const \)_\(LOCK_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# The PRIO constants.
grep '^const _PRIO_' gen-sysinfo.go | \
  sed -e 's/^\(const \)_\(PRIO_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# The GNU/Linux LINUX_REBOOT flags.
grep '^const _LINUX_REBOOT_' gen-sysinfo.go |
    sed -e 's/^\(const \)_\(LINUX_REBOOT_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# The GNU/Linux sock_filter struct.
grep '^type _sock_filter ' gen-sysinfo.go | \
    sed -e 's/_sock_filter/SockFilter/' \
      -e 's/code/Code/' \
      -e 's/jt/Jt/' \
      -e 's/jf/Jf/' \
      -e 's/k /K /' \
    >> ${OUT}

# The GNU/Linux sock_fprog struct.
grep '^type _sock_fprog ' gen-sysinfo.go | \
    sed -e 's/_sock_fprog/SockFprog/' \
      -e 's/len/Len/' \
      -e 's/filter/Filter/' \
      -e 's/_sock_filter/SockFilter/' \
    >> ${OUT}

# The GNU/Linux filter flags.
grep '^const _BPF_' gen-sysinfo.go | \
  sed -e 's/^\(const \)_\(BPF_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# The GNU/Linux nlattr struct.
grep '^type _nlattr ' gen-sysinfo.go | \
    sed -e 's/_nlattr/NlAttr/' \
      -e 's/nla_len/Len/' \
      -e 's/nla_type/Type/' \
    >> ${OUT}

# The GNU/Linux nlmsgerr struct.
grep '^type _nlmsgerr ' gen-sysinfo.go | \
    sed -e 's/_nlmsgerr/NlMsgerr/' \
      -e 's/error/Error/' \
      -e 's/msg/Msg/' \
      -e 's/_nlmsghdr/NlMsghdr/' \
    >> ${OUT}

# The GNU/Linux rtnexthop struct.
grep '^type _rtnexthop ' gen-sysinfo.go | \
    sed -e 's/_rtnexthop/RtNexthop/' \
      -e 's/rtnh_len/Len/' \
      -e 's/rtnh_flags/Flags/' \
      -e 's/rtnh_hops/Hops/' \
      -e 's/rtnh_ifindex/Ifindex/' \
    >> ${OUT}

# The GNU/Linux netlink flags.
grep '^const _NETLINK_' gen-sysinfo.go | \
  sed -e 's/^\(const \)_\(NETLINK_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
grep '^const _NLA_' gen-sysinfo.go | \
  sed -e 's/^\(const \)_\(NLA_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# The GNU/Linux packet socket flags.
grep '^const _PACKET_' gen-sysinfo.go | \
  sed -e 's/^\(const \)_\(PACKET_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}

# The GNU/Linux inotify_event struct.
grep '^type _inotify_event ' gen-sysinfo.go | \
    sed -e 's/_inotify_event/InotifyEvent/' \
      -e 's/wd/Wd/' \
      -e 's/mask/Mask/' \
      -e 's/cookie/Cookie/' \
      -e 's/len/Len/' \
      -e 's/name/Name/' \
      -e 's/\[\]/[0]/' \
      -e 's/\[0\]byte/[0]int8/' \
    >> ${OUT}

# The GNU/Linux CLONE flags.
grep '^const _CLONE_' gen-sysinfo.go | \
  sed -e 's/^\(const \)_\(CLONE_[^= ]*\)\(.*\)$/\1\2 = _\2/' >> ${OUT}
# We need some CLONE constants that are not defined in older versions
# of glibc.
if ! grep '^const CLONE_NEWUSER ' ${OUT} > /dev/null 2>&1; then
  echo "const CLONE_NEWUSER = 0x10000000" >> ${OUT}
fi

# Struct sizes.
set cmsghdr Cmsghdr ip_mreq IPMreq ip_mreqn IPMreqn ipv6_mreq IPv6Mreq \
    ifaddrmsg IfAddrmsg ifinfomsg IfInfomsg in_pktinfo Inet4Pktinfo \
    in6_pktinfo Inet6Pktinfo inotify_event InotifyEvent linger Linger \
    msghdr Msghdr nlattr NlAttr nlmsgerr NlMsgerr nlmsghdr NlMsghdr \
    rtattr RtAttr rtgenmsg RtGenmsg rtmsg RtMsg rtnexthop RtNexthop \
    sock_filter SockFilter sock_fprog SockFprog ucred Ucred \
    icmp6_filter ICMPv6Filter
while test $# != 0; do
    nc=$1
    ngo=$2
    shift
    shift
    if grep "^const _sizeof_$nc =" gen-sysinfo.go >/dev/null 2>&1; then
	echo "const Sizeof$ngo = _sizeof_$nc" >> ${OUT}
    fi
done

# In order to compile the net package, we need some sizes to exist
# even if the types do not.
if ! grep 'const SizeofIPMreq ' ${OUT} >/dev/null 2>&1; then
    echo 'const SizeofIPMreq = 8' >> ${OUT}
fi
if ! grep 'const SizeofIPv6Mreq ' ${OUT} >/dev/null 2>&1; then
    echo 'const SizeofIPv6Mreq = 20' >> ${OUT}
fi
if ! grep 'const SizeofIPMreqn ' ${OUT} >/dev/null 2>&1; then
    echo 'const SizeofIPMreqn = 12' >> ${OUT}
fi
if ! grep 'const SizeofICMPv6Filter ' ${OUT} >/dev/null 2>&1; then
    echo 'const SizeofICMPv6Filter = 32' >> ${OUT}
fi

# The Solaris 11 Update 1 _zone_net_addr_t struct.
grep '^type _zone_net_addr_t ' gen-sysinfo.go | \
    sed -e 's/_in6_addr/[16]byte/' \
    >> ${OUT}

# The Solaris 12 _flow_arp_desc_t struct.
grep '^type _flow_arp_desc_t ' gen-sysinfo.go | \
    sed -e 's/_in6_addr_t/[16]byte/g' \
    >> ${OUT}

# The Solaris 12 _flow_l3_desc_t struct.
grep '^type _flow_l3_desc_t ' gen-sysinfo.go | \
    sed -e 's/_in6_addr_t/[16]byte/g' \
    >> ${OUT}

# The Solaris 12 _mac_ipaddr_t struct.
grep '^type _mac_ipaddr_t ' gen-sysinfo.go | \
    sed -e 's/_in6_addr_t/[16]byte/g' \
    >> ${OUT}

# The Solaris 12 _mactun_info_t struct.
grep '^type _mactun_info_t ' gen-sysinfo.go | \
    sed -e 's/_in6_addr_t/[16]byte/g' \
    >> ${OUT}

exit $?
