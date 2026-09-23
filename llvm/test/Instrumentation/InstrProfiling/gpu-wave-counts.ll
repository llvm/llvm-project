; RUN: opt -passes=instrprof -offload-pgo-sampling=0 -S %s | FileCheck %s
; RUN: opt -passes=instrprof -offload-pgo-sampling=3 -S %s | FileCheck %s --check-prefix=SAMPLE
; RUN: not opt -passes=instrprof -profile-correlate=debug-info -disable-output %s 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not opt -passes=instrprof -sampled-instrumentation -disable-output %s 2>&1 | FileCheck %s --check-prefix=ERROR
; ERROR: wave counts do not support profile correlation or lane-level instrumentation sampling

target triple = "amdgcn-amd-amdhsa"
@__profn_test = private constant [4 x i8] c"test"

; CHECK: @__profc_test = {{.*}}[4 x i64] zeroinitializer
; CHECK: @__llvm_prf_unifcnt_test = {{.*}}[2 x i64] zeroinitializer
; CHECK: @__profd_test = {{.*}}i32 4, [3 x i16] zeroinitializer, i16 0, i32 0, i32 2 }
; CHECK-LABEL: define amdgpu_kernel void @test
; CHECK: call void @__llvm_profile_instrument_gpu(ptr {{.*}}, ptr {{.*}}, i64 1, ptr {{.*}}i32 2
; CHECK-NEXT: br i1 %cond, label %a, label %b
; CHECK: a:
; CHECK-NEXT: call void @__llvm_profile_instrument_gpu(ptr {{.*}}, ptr {{.*}}, i64 1, ptr {{.*}}i32 3
; CHECK-NEXT: br label %exit
; CHECK: b:
; CHECK-NEXT: br label %exit
; CHECK: exit:
; CHECK-NEXT: ret void
; SAMPLE: call i32 @__llvm_profile_sampling_gpu(i32 3)
; SAMPLE: br i1
; SAMPLE: call void @__llvm_profile_instrument_gpu(ptr {{.*}}, ptr {{.*}}, i64 1, ptr {{.*}}i32 2
; SAMPLE: call void @__llvm_profile_instrument_gpu(ptr {{.*}}, ptr {{.*}}, i64 1, ptr {{.*}}i32 3
define amdgpu_kernel void @test(i1 %cond) {
entry:
  call void @llvm.instrprof.increment(ptr @__profn_test, i64 123, i32 2, i32 0)
  br i1 %cond, label %a, label %b
a:
  call void @llvm.instrprof.increment(ptr @__profn_test, i64 123, i32 2, i32 1)
  br label %exit
b:
  br label %exit
exit:
  ret void
}
