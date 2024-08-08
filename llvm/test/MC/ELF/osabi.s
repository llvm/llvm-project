# REQUIRES: aarch64-registered-target
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s | llvm-readobj -h - | \
# RUN:   FileCheck %s --check-prefix=SYSV
# SYSV: OS/ABI: SystemV

# RUN: llvm-mc -filetype=obj -triple=amd64-solaris %s | llvm-readobj -h - | \
# RUN:   FileCheck %s --check-prefix=SOLARIS
# SOLARIS: OS/ABI: Solaris

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-freebsd %s | llvm-readobj -h - | \
# RUN:   FileCheck %s --check-prefix=FREEBSD
# FREEBSD: OS/ABI: FreeBSD

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-openbsd %s | llvm-readobj -h - | \
# RUN:   FileCheck %s --check-prefix=OPENBSD
# OPENBSD: OS/ABI: OpenBSD
