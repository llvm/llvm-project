; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -S | FileCheck %s --check-prefixes=CHECK-NOALIAS,CHECK-NOINDICATOR
; RUN: opt < %s -passes='asan-pipeline' -S | FileCheck %s --check-prefixes=CHECK-NOALIAS,CHECK-NOINDICATOR
; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -asan-use-private-alias=1 -S | FileCheck %s --check-prefixes=CHECK-ALIAS,CHECK-NOINDICATOR
; RUN: opt < %s -passes='asan-pipeline' -asan-use-private-alias=1 -S | FileCheck %s --check-prefixes=CHECK-ALIAS,CHECK-NOINDICATOR
; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -asan-use-odr-indicator=1 -S | FileCheck %s --check-prefixes=CHECK-INDICATOR,CHECK-NOALIAS
; RUN: opt < %s -passes='asan-pipeline' -asan-use-odr-indicator=1 -S | FileCheck %s --check-prefixes=CHECK-INDICATOR,CHECK-NOALIAS
; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -asan-use-private-alias=1 -asan-use-odr-indicator=1 -S | FileCheck %s --check-prefixes=CHECK-ALIAS,CHECK-INDICATOR
; RUN: opt < %s -passes='asan-pipeline' -asan-use-private-alias=1 -asan-use-odr-indicator=1 -S | FileCheck %s --check-prefixes=CHECK-ALIAS,CHECK-INDICATOR

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = dso_local global [2 x i32] zeroinitializer, align 4
@b = private global [2 x i32] zeroinitializer, align 4
@c = internal global [2 x i32] zeroinitializer, align 4
@d = unnamed_addr global [2 x i32] zeroinitializer, align 4

; Check that we generate internal alias and odr indicator symbols for global to be protected.
; CHECK-NOINDICATOR-NOT: __odr_asan_gen_a
; CHECK-NOALIAS-NOT: private alias
; CHECK-INDICATOR: @__odr_asan_gen_a = global i8 0, align 1
; CHECK-ALIAS: @0 = private alias { [2 x i32], [56 x i8] }, { [2 x i32], [56 x i8] }* @a

; CHECK-ALIAS: @1 = private alias { [2 x i32], [56 x i8] }, { [2 x i32], [56 x i8] }* @b
; CHECK-ALIAS: @2 = private alias { [2 x i32], [56 x i8] }, { [2 x i32], [56 x i8] }* @c
; CHECK-ALIAS: @3 = private alias { [2 x i32], [56 x i8] }, { [2 x i32], [56 x i8] }* @d

; Function Attrs: nounwind sanitize_address uwtable
define i32 @foo(i32 %M) #0 {
entry:
  %M.addr = alloca i32, align 4
  store i32 %M, i32* %M.addr, align 4
  store volatile i32 6, i32* getelementptr inbounds ([2 x i32], [2 x i32]* @a, i64 2, i64 0), align 4
  %0 = load i32, i32* %M.addr, align 4
  %idxprom = sext i32 %0 to i64
  %arrayidx = getelementptr inbounds [2 x i32], [2 x i32]* @a, i64 0, i64 %idxprom
  %1 = load volatile i32, i32* %arrayidx, align 4
  ret i32 %1
}
