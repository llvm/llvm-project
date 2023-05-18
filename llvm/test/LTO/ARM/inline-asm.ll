; Check that we don't crash on target-specific inline asm directives.
;
; RUN: llvm-as < %s > %t
; RUN: llvm-lto -o /dev/null %t -mcpu armv4t

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv4t-unknown-linux"

module asm ".fnstart"
