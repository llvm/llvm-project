; Test optimization experiments.
; -asan-force-experiment flag turns all memory accesses into experiments.
; RUN: opt < %s -passes=asan -asan-force-experiment=42 -S | FileCheck %s
; RUN: opt < %s -passes=asan -asan-force-experiment=42 -S -mtriple=s390x-unknown-linux | FileCheck %s --check-prefix=EXT
; RUN: opt < %s -passes=asan -asan-force-experiment=42 -S -mtriple=mips-linux-gnu | FileCheck %s --check-prefix=MIPS_EXT
; RUN: opt < %s -passes=asan -asan-force-experiment=42 -S -mtriple=loongarch64-unknown-linux-gnu | FileCheck %s --check-prefix=LA_EXT
; REQUIRES: x86-registered-target, systemz-registered-target, mips-registered-target, loongarch-registered-target

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define void @load1(ptr %p) sanitize_address {
entry:
  %t = load i8, ptr %p, align 1
  ret void
; CHECK-LABEL: define void @load1
; CHECK: __asan_report_exp_load1{{.*}} i32 42
; CHECK: ret void
}

define void @load2(ptr %p) sanitize_address {
entry:
  %t = load i16, ptr %p, align 2
  ret void
; CHECK-LABEL: define void @load2
; CHECK: __asan_report_exp_load2{{.*}} i32 42
; CHECK: ret void
}

define void @load4(ptr %p) sanitize_address {
entry:
  %t = load i32, ptr %p, align 4
  ret void
; CHECK-LABEL: define void @load4
; CHECK: __asan_report_exp_load4{{.*}} i32 42
; CHECK: ret void
}

define void @load8(ptr %p) sanitize_address {
entry:
  %t = load i64, ptr %p, align 8
  ret void
; CHECK-LABEL: define void @load8
; CHECK: __asan_report_exp_load8{{.*}} i32 42
; CHECK: ret void
}

define void @load16(ptr %p) sanitize_address {
entry:
  %t = load i128, ptr %p, align 16
  ret void
; CHECK-LABEL: define void @load16
; CHECK: __asan_report_exp_load16{{.*}} i32 42
; CHECK: ret void
}

define void @loadN(ptr %p) sanitize_address {
entry:
  %t = load i48, ptr %p, align 1
  ret void
; CHECK-LABEL: define void @loadN
; CHECK: __asan_report_exp_load_n{{.*}} i32 42
; CHECK: ret void
}

define void @store1(ptr %p) sanitize_address {
entry:
  store i8 1, ptr %p, align 1
  ret void
; CHECK-LABEL: define void @store1
; CHECK: __asan_report_exp_store1{{.*}} i32 42
; CHECK: ret void
}

define void @store2(ptr %p) sanitize_address {
entry:
  store i16 1, ptr %p, align 2
  ret void
; CHECK-LABEL: define void @store2
; CHECK: __asan_report_exp_store2{{.*}} i32 42
; CHECK: ret void
}

define void @store4(ptr %p) sanitize_address {
entry:
  store i32 1, ptr %p, align 4
  ret void
; CHECK-LABEL: define void @store4
; CHECK: __asan_report_exp_store4{{.*}} i32 42
; CHECK: ret void
}

define void @store8(ptr %p) sanitize_address {
entry:
  store i64 1, ptr %p, align 8
  ret void
; CHECK-LABEL: define void @store8
; CHECK: __asan_report_exp_store8{{.*}} i32 42
; CHECK: ret void
}

define void @store16(ptr %p) sanitize_address {
entry:
  store i128 1, ptr %p, align 16
  ret void
; CHECK-LABEL: define void @store16
; CHECK: __asan_report_exp_store16{{.*}} i32 42
; CHECK: ret void
}

define void @storeN(ptr %p) sanitize_address {
entry:
  store i48 1, ptr %p, align 1
  ret void
; CHECK-LABEL: define void @storeN
; CHECK: __asan_report_exp_store_n{{.*}} i32 42
; CHECK: ret void
}

; CHECK:    declare void @__asan_report_exp_load_n(i64, i64, i32)
; EXT:      declare void @__asan_report_exp_load_n(i64, i64, i32 zeroext)
; MIPS_EXT: declare void @__asan_report_exp_load_n(i64, i64, i32 signext)
; LA_EXT:   declare void @__asan_report_exp_load_n(i64, i64, i32 signext)

; CHECK:    declare void @__asan_exp_loadN(i64, i64, i32)
; EXT:      declare void @__asan_exp_loadN(i64, i64, i32 zeroext)
; MIPS_EXT: declare void @__asan_exp_loadN(i64, i64, i32 signext)
; LA_EXT:   declare void @__asan_exp_loadN(i64, i64, i32 signext)

; CHECK:    declare void @__asan_report_exp_load1(i64, i32)
; EXT:      declare void @__asan_report_exp_load1(i64, i32 zeroext)
; MIPS_EXT: declare void @__asan_report_exp_load1(i64, i32 signext)
; LA_EXT:   declare void @__asan_report_exp_load1(i64, i32 signext)

; CHECK:    declare void @__asan_exp_load1(i64, i32)
; EXT:      declare void @__asan_exp_load1(i64, i32 zeroext)
; MIPS_EXT: declare void @__asan_exp_load1(i64, i32 signext)
; LA_EXT:   declare void @__asan_exp_load1(i64, i32 signext)
