; Defaults
; RUN: opt < %s -passes=asan -S | FileCheck %s --check-prefixes=CHECK-ALIAS,CHECK-INDICATOR

; {newPM,legacyPM} x {alias0,alias1} x {odr0,odr1}
; RUN: opt < %s -passes=asan             -asan-use-private-alias=0 -asan-use-odr-indicator=0 -S | FileCheck %s --check-prefixes=CHECK-NOALIAS,CHECK-NOINDICATOR
; RUN: opt < %s -passes=asan             -asan-use-private-alias=1 -asan-use-odr-indicator=0 -S | FileCheck %s --check-prefixes=CHECK-ALIAS,CHECK-NOINDICATOR
; RUN: opt < %s -passes=asan             -asan-use-private-alias=0 -asan-use-odr-indicator=1 -S | FileCheck %s --check-prefixes=CHECK-NOALIAS,CHECK-INDICATOR
; RUN: opt < %s -passes=asan             -asan-use-private-alias=1 -asan-use-odr-indicator=1 -S | FileCheck %s --check-prefixes=CHECK-ALIAS,CHECK-INDICATOR

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
; CHECK-ALIAS: @0 = private alias { [2 x i32], [24 x i8] }, ptr @a

; CHECK-ALIAS: @1 = private alias { [2 x i32], [24 x i8] }, ptr @b
; CHECK-ALIAS: @2 = private alias { [2 x i32], [24 x i8] }, ptr @c
; CHECK-ALIAS: @3 = private alias { [2 x i32], [24 x i8] }, ptr @d

; Function Attrs: nounwind sanitize_address uwtable
define i32 @foo(i32 %M) #0 {
entry:
  %M.addr = alloca i32, align 4
  store i32 %M, ptr %M.addr, align 4
  store volatile i32 6, ptr getelementptr inbounds ([2 x i32], ptr @a, i64 2, i64 0), align 4
  %0 = load i32, ptr %M.addr, align 4
  %idxprom = sext i32 %0 to i64
  %arrayidx = getelementptr inbounds [2 x i32], ptr @a, i64 0, i64 %idxprom
  %1 = load volatile i32, ptr %arrayidx, align 4
  ret i32 %1
}
