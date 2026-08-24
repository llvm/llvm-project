; RUN: opt < %s -passes=dasan -S | FileCheck %s
; RUN: opt < %s -passes=dasan -dasan-instrument-globals -S | FileCheck %s --check-prefix=GLOBALS

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@g = global i32 1

; CHECK: @g = global i32 1
; CHECK-NOT: @g.dasan
; CHECK: @__dasan_no_entry = private constant i64 0
; CHECK-NOT: addrspace(

; GLOBALS: @g.dasan = private global { i32, [60 x i8] } { i32 1, [60 x i8] zeroinitializer }, section ".data.dasan.globals", align 64
; GLOBALS: @g = {{.*}}alias i32, ptr @g.dasan

define i32 @load4(ptr %a) sanitize_device_address {
; CHECK-LABEL: define i32 @load4(
; CHECK: [[TOP:%.*]] = lshr i64 {{.*}}, 45
; CHECK-NEXT: [[MINE:%.*]] = icmp eq i64 [[TOP]], 1
; CHECK-NEXT: br i1 [[MINE]], label %[[SLOW:.*]], label %[[GO:.*]], !prof
; CHECK: [[SLOW]]:
; CHECK-NOT: select
; CHECK: load i64, ptr {{.*}}, align 8, !alias.scope
; CHECK: call void @__dasan_report_load(ptr %a, i64 4, i64 {{.*}}, i64 {{.*}})
; CHECK: [[GO]]:
; CHECK: ret i32
  %v = load i32, ptr %a, align 4
  ret i32 %v
}
