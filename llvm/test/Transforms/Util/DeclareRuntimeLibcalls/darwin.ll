; REQUIRES: aarch64-registered-target, arm-registered-target, x86-registered-target

; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=i386-apple-macosx10.5 < %s | FileCheck -check-prefixes=HAS-MEMSET-PATTERN,MACOS %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=i386-apple-macosx10.4 < %s | FileCheck -check-prefixes=NO-MEMSET-PATTERN,MACOS %s

; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-apple-macosx10.5 < %s | FileCheck -check-prefix=HAS-MEMSET-PATTERN %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-apple-macosx10.4 < %s | FileCheck -check-prefix=NO-MEMSET-PATTERN %s

; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=arm64-apple-macos10.5 < %s | FileCheck -check-prefixes=HAS-MEMSET-PATTERN,MACOS %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=arm64-apple-ios3 < %s | FileCheck -check-prefix=HAS-MEMSET-PATTERN %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=arm64-apple-ios2 < %s | FileCheck -check-prefix=NO-MEMSET-PATTERN %s

; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=thumbv7-apple-ios3 < %s | FileCheck -check-prefix=HAS-MEMSET-PATTERN %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=thumbv7-apple-ios2 < %s | FileCheck -check-prefix=NO-MEMSET-PATTERN %s

; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=arm64_32-apple-watchos < %s | FileCheck -check-prefix=HAS-MEMSET-PATTERN %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=armv7k-apple-watchos < %s | FileCheck -check-prefix=HAS-MEMSET-PATTERN %s

; MACOS: declare void @getc_unlocked(...)
; MACOS: declare void @getchar_unlocked(...)

; HAS-MEMSET-PATTERN: declare void @memset_pattern16(...)
; HAS-MEMSET-PATTERN: declare void @memset_pattern4(...)
; HAS-MEMSET-PATTERN: declare void @memset_pattern8(...)

; MACOS: declare void @putc_unlocked(...)
; MACOS: declare void @putchar_unlocked(...)

; NO-MEMSET-PATTERN-NOT: memset_pattern
