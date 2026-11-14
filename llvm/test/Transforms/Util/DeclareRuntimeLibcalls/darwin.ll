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

; MACOS: declare noundef i32 @getc_unlocked(ptr noundef captures(none)) [[STDIO_ATTRS:#[0-9]+]]
; MACOS: declare noundef i32 @getchar_unlocked() [[STDIO_ATTRS]]

; HAS-MEMSET-PATTERN-32: declare void @memset_pattern16(ptr writeonly captures(none), ptr readonly captures(none), i32) [[MEMSET_PATTERN_ATTRS:#[0-9]+]]
; HAS-MEMSET-PATTERN-32: declare void @memset_pattern4(ptr writeonly captures(none), ptr readonly captures(none), i32) [[MEMSET_PATTERN_ATTRS]]
; HAS-MEMSET-PATTERN-32: declare void @memset_pattern8(ptr writeonly captures(none), ptr readonly captures(none), i32) [[MEMSET_PATTERN_ATTRS]]
; HAS-MEMSET-PATTERN-64: declare void @memset_pattern16(ptr writeonly captures(none), ptr readonly captures(none), i64) [[MEMSET_PATTERN_ATTRS:#[0-9]+]]
; HAS-MEMSET-PATTERN-64: declare void @memset_pattern4(ptr writeonly captures(none), ptr readonly captures(none), i64) [[MEMSET_PATTERN_ATTRS]]
; HAS-MEMSET-PATTERN-64: declare void @memset_pattern8(ptr writeonly captures(none), ptr readonly captures(none), i64) [[MEMSET_PATTERN_ATTRS]]

; MACOS: declare noundef i32 @putc_unlocked(i32 noundef, ptr noundef captures(none)) [[STDIO_ATTRS]]
; MACOS: declare noundef i32 @putchar_unlocked(i32 noundef) [[STDIO_ATTRS]]

; HAS-MEMSET-PATTERN-32: attributes [[MEMSET_PATTERN_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
; HAS-MEMSET-PATTERN-64: attributes [[MEMSET_PATTERN_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }

; NO-MEMSET-PATTERN-NOT: memset_pattern

; MACOS: attributes [[STDIO_ATTRS]] = { nofree nounwind }
