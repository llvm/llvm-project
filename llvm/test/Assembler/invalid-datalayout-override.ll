; Check that importing this file gives an error due to the broken data layout:
; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; Check that specifying a valid data layout allows to import this file:
; RUN: llvm-as -data-layout "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128" < %s

target datalayout = "A16777216"
; CHECK: Invalid address space, must be a 24-bit integer
