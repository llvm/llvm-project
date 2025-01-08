; REQUIRES: x86-registered-target

;; Validate that we produce RTTI proxies for 32-bit x86.
; RUN: llc %s -mtriple=i686-elf -o - | FileCheck %s

;; Validate that we produce a valid object file.
; RUN: llc %s -mtriple=i686-elf --filetype=obj -o %t.o
; RUN: llvm-readobj --relocs %t.o | FileCheck --check-prefix=RELOCS %s

@vtable = dso_local unnamed_addr constant i32 trunc (i64 sub (i64 ptrtoint (ptr @rtti.proxy to i64), i64 ptrtoint (ptr @vtable to i64)) to i32), align 4
@rtti = external global i8, align 8
@rtti.proxy = linkonce_odr hidden unnamed_addr constant ptr @rtti

; CHECK-LABEL: vtable:
; CHECK-NEXT:    .long   rtti.proxy-vtable

; CHECK-LABEL: rtti.proxy:
; CHECK-NEXT:    .long   rtti

; RELOCS: R_386_32 rtti
