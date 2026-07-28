; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple powerpc-ibm-aix-xcoff < \
; RUN: %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple powerpc64le-unknown-linux < \
; RUN: %s | FileCheck %s

@.str = private unnamed_addr constant [12 x i8] c"MY_METADATA\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [10 x i8] c"my_file.c\00", section "llvm.metadata"
@global.annotations = appending global [3 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @a, ptr @.str, ptr @.str.1, i32 100, ptr null }, { ptr, ptr, ptr, i32, ptr } { ptr @b, ptr @.str, ptr @.str.1, i32 200, ptr null }, { ptr, ptr, ptr, i32, ptr } { ptr @c, ptr @.str, ptr @.str.1, i32 300, ptr null }], section "llvm.metadata"

@a = global i32 1
@b = global i32 2
@c = global i32 3

; CHECK-NOT: metadata
; CHECK-NOT: annotations
