; REQUIRES: aarch64-registered-target, arm-registered-target, x86-registered-target

; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=i386-apple-macosx10.5 < %s | FileCheck -check-prefixes=HAS-MEMSET-PATTERN-32,MACOS %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=i386-apple-macosx10.4 < %s | FileCheck -check-prefixes=NO-MEMSET-PATTERN,MACOS %s

; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-apple-macosx10.5 < %s | FileCheck -check-prefix=HAS-MEMSET-PATTERN-64 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-apple-macosx10.4 < %s | FileCheck -check-prefix=NO-MEMSET-PATTERN %s

; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=arm64-apple-macos10.5 < %s | FileCheck -check-prefixes=HAS-MEMSET-PATTERN-64,MACOS %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=arm64-apple-ios3 < %s | FileCheck -check-prefix=HAS-MEMSET-PATTERN-64 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=arm64-apple-ios2 < %s | FileCheck -check-prefix=NO-MEMSET-PATTERN %s

; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=thumbv7-apple-ios3 < %s | FileCheck -check-prefix=HAS-MEMSET-PATTERN-32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=thumbv7-apple-ios2 < %s | FileCheck -check-prefix=NO-MEMSET-PATTERN %s

; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=arm64_32-apple-watchos < %s | FileCheck -check-prefix=HAS-MEMSET-PATTERN-32 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=armv7k-apple-watchos < %s | FileCheck -check-prefix=HAS-MEMSET-PATTERN-32 %s

; MACOS: declare i32 @getc_unlocked(ptr)
; MACOS: declare i32 @getchar_unlocked()

; HAS-MEMSET-PATTERN-32: declare void @memset_pattern16(ptr, ptr, i32)
; HAS-MEMSET-PATTERN-32: declare void @memset_pattern4(ptr, ptr, i32)
; HAS-MEMSET-PATTERN-32: declare void @memset_pattern8(ptr, ptr, i32)
; HAS-MEMSET-PATTERN-64: declare void @memset_pattern16(ptr, ptr, i64)
; HAS-MEMSET-PATTERN-64: declare void @memset_pattern4(ptr, ptr, i64)
; HAS-MEMSET-PATTERN-64: declare void @memset_pattern8(ptr, ptr, i64)

; MACOS: declare i32 @putc_unlocked(i32, ptr)
; MACOS: declare i32 @putchar_unlocked(i32)

; NO-MEMSET-PATTERN-NOT: memset_pattern
