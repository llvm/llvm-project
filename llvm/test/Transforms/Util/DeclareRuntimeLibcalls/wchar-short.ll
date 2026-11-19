; Check wchar_t width and the its extension attribute with wchar_t size of 2,
; simulating -fshort-wchar.

; wchar_t is the same as int, no extension required.
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=avr < %s | FileCheck -check-prefixes=CHECK,I16 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=msp430 < %s | FileCheck -check-prefixes=CHECK,I16 %s

; No extension on x86_64-windows and non-darwin aarch64.
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=aarch64-linux < %s | FileCheck -check-prefixes=CHECK,I16 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=aarch64-windows < %s | FileCheck -check-prefixes=CHECK,I16 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-windows < %s | FileCheck -check-prefixes=CHECK,I16 %s

; Zero extension is the default.
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=aarch64-darwin < %s | FileCheck -check-prefixes=CHECK,Z16 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=armv7-linux < %s | FileCheck -check-prefixes=CHECK,Z16 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=armv7-darwin < %s | FileCheck -check-prefixes=CHECK,Z16 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=armv7-windows < %s | FileCheck -check-prefixes=CHECK,Z16 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=i386-linux < %s | FileCheck -check-prefixes=CHECK,Z16 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=i386-windows < %s | FileCheck -check-prefixes=CHECK,Z16 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=riscv32 < %s | FileCheck -check-prefixes=CHECK,Z16 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=riscv64 < %s | FileCheck -check-prefixes=CHECK,Z16 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-linux < %s | FileCheck -check-prefixes=CHECK,Z16 %s

; CHECK: declare{{.*}}wmemset
; I16-SAME: (ptr, i16,
; Z16-SAME: (ptr, i16 zeroext

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"wchar_size", i32 2}
