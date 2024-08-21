; RUN: llvm-mc -triple=arm64e-apple-ios -filetype=obj %s -o - | llvm-objdump --macho -d -p - | FileCheck %s

; CHECK: _foo:
; CHECK: 0: c0 03 5f d6   ret

; CHECK: Mach header
; CHECK:       magic cputype cpusubtype  caps    filetype ncmds sizeofcmds      flags
; CHECK: MH_MAGIC_64   ARM64          E  PAC00     OBJECT     3        256 0x00000000

.globl _foo
_foo:
  ret

; Upstream doesn't support the version directive and flags yet.  So the
; upstream version of this test doesn't have the directive.
; Instead it relies on the v0 default.
.ptrauth_abi_version 0
