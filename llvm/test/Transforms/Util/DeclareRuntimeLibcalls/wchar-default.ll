; Check wchar_t width and the its extension attribute with target-default
; wchar_t size.

; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=aarch64-linux < %s | FileCheck -check-prefixes=CHECK,I32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=aarch64-darwin < %s | FileCheck -check-prefixes=CHECK,I32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=aarch64-windows < %s | FileCheck -check-prefixes=CHECK,I16 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=armv7-linux < %s | FileCheck -check-prefixes=CHECK,I32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=armv7-darwin < %s | FileCheck -check-prefixes=CHECK,I32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=armv7-windows < %s | FileCheck -check-prefixes=CHECK,Z16 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=avr < %s | FileCheck -check-prefixes=CHECK,I16 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=i386-linux < %s | FileCheck -check-prefixes=CHECK,I32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=i386-windows < %s | FileCheck -check-prefixes=CHECK,Z16 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=loongarch32 < %s | FileCheck -check-prefixes=CHECK,S32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=loongarch64 < %s | FileCheck -check-prefixes=CHECK,S32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=mips < %s | FileCheck -check-prefixes=CHECK,S32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=mips64 < %s | FileCheck -check-prefixes=CHECK,S32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=msp430 < %s | FileCheck -check-prefixes=CHECK,I16 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=powerpc-linux < %s | FileCheck -check-prefixes=CHECK,I32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=powerpc-aix < %s | FileCheck -check-prefixes=CHECK,Z16 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=powerpc64-linux < %s | FileCheck -check-prefixes=CHECK,S32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=powerpc64-aix < %s | FileCheck -check-prefixes=CHECK,Z32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=riscv32 < %s | FileCheck -check-prefixes=CHECK,I32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=riscv64 < %s | FileCheck -check-prefixes=CHECK,S32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=s390x-linux < %s | FileCheck -check-prefixes=CHECK,S32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=s390x-zos < %s | FileCheck -check-prefixes=CHECK,Z32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=sparc < %s | FileCheck -check-prefixes=CHECK,I32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=sparcv9 < %s | FileCheck -check-prefixes=CHECK,S32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=ve < %s | FileCheck -check-prefixes=CHECK,Z32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-linux < %s | FileCheck -check-prefixes=CHECK,I32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-windows < %s | FileCheck -check-prefixes=CHECK,I16 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=xcore < %s | FileCheck -check-prefixes=CHECK,Z8 %s

; CHECK: declare{{.*}}wmemset
; Z8-SAME:  (ptr, i8 zeroext
; I16-SAME: (ptr, i16,
; Z16-SAME: (ptr, i16 zeroext
; I32-SAME: (ptr, i32,
; Z32-SAME: (ptr, i32 zeroext
; S32-SAME: (ptr, i32 signext
