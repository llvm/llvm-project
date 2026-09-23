; RUN: opt -passes='pgo-instr-gen,always-inline,instrprof,verify' -offload-pgo-sampling=0 -S %s | FileCheck %s

; Inlining leaves several profile names in each caller. Each name needs its own
; lane, uniform, and wave counters, even after the callee definition is removed.
source_filename = "wave-inline.c"
target triple = "amdgcn-amd-amdhsa"

; CHECK-DAG: @__profc_caller = {{.*}}[2 x i64] zeroinitializer
; CHECK-DAG: @__profc_other = {{.*}}[2 x i64] zeroinitializer
; CHECK-DAG: @[[CALLEE:__profc_.*callee]] = {{.*}}[2 x i64] zeroinitializer
; CHECK-DAG: @__llvm_prf_unifcnt_{{.*}}callee = {{.*}}[1 x i64] zeroinitializer
; CHECK-DAG: @__profd_{{.*}}callee = {{.*}}i32 2, [3 x i16] zeroinitializer, i16 0, i32 0, i32 1 }

; CHECK-LABEL: define i32 @caller(
; CHECK: call void @__llvm_profile_instrument_gpu({{.*}}i64 1, ptr {{.*}}@__profc_caller{{.*}}i32 1
; CHECK: call void @__llvm_profile_instrument_gpu({{.*}}i64 1, ptr {{.*}}@[[CALLEE]]{{.*}}i32 1
define i32 @caller(i32 %x) {
  %v = call i32 @callee(i32 %x)
  ret i32 %v
}

; CHECK-LABEL: define i32 @other(
; CHECK: call void @__llvm_profile_instrument_gpu({{.*}}i64 1, ptr {{.*}}@__profc_other{{.*}}i32 1
; CHECK: call void @__llvm_profile_instrument_gpu({{.*}}i64 1, ptr {{.*}}@[[CALLEE]]{{.*}}i32 1
; CHECK-NOT: define {{.*}}@callee(
define i32 @other(i32 %x) {
  %v = call i32 @callee(i32 %x)
  ret i32 %v
}

define internal i32 @callee(i32 %x) alwaysinline {
  %v = add i32 %x, 1
  ret i32 %v
}
