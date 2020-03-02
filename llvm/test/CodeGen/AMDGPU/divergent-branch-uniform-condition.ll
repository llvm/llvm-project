; RUN: opt --amdgpu-annotate-uniform -S %s |  FileCheck %s -check-prefix=UNIFORM
; RUN: opt --amdgpu-annotate-uniform --si-annotate-control-flow -S %s |  FileCheck %s -check-prefix=CONTROLFLOW

; This module creates a divergent branch in block Flow2. The branch is
; marked as divergent by the divergence analysis but the condition is
; not. This test ensures that the divergence of the branch is tested,
; not its condition, so that branch is correctly emitted as divergent.

target triple = "amdgcn-mesa-mesa3d"

define amdgpu_ps void @main(i32 %0, float %1) {
start:
  %v0 = call float @llvm.amdgcn.interp.p1(float %1, i32 0, i32 0, i32 %0)
  br label %loop

loop:                                             ; preds = %Flow, %start
  %v1 = phi i32 [ 0, %start ], [ %6, %Flow ]
  %v2 = icmp ugt i32 %v1, 31
  %2 = xor i1 %v2, true
  br i1 %2, label %endif1, label %Flow

Flow1:                                            ; preds = %endif2, %endif1
  %3 = phi i32 [ %v5, %endif2 ], [ undef, %endif1 ]
  %4 = phi i1 [ false, %endif2 ], [ true, %endif1 ]
  br label %Flow

; UNIFORM-LABEL: Flow2:
; UNIFORM-NEXT: br i1 %8, label %if1, label %endloop
; UNIFORM-NOT: !amdgpu.uniform
; UNIFORM: if1:

; CONTROLFLOW-LABEL: Flow2:
; CONTROLFLOW-NEXT:  call void @llvm.amdgcn.end.cf.i64(i64 %{{.*}})
; CONTROLFLOW-NEXT:  [[IF:%.*]] = call { i1, i64 } @llvm.amdgcn.if.i64(i1 %{{.*}})
; CONTROLFLOW-NEXT:  [[COND:%.*]] = extractvalue { i1, i64 } [[IF]], 0
; CONTROLFLOW-NEXT:  %{{.*}} = extractvalue { i1, i64 } [[IF]], 1
; CONTROLFLOW-NEXT:  br i1 [[COND]], label %if1, label %endloop

Flow2:                                            ; preds = %Flow
  br i1 %8, label %if1, label %endloop

if1:                                              ; preds = %Flow2
  %v3 = call float @llvm.sqrt.f32(float %v0)
  br label %endloop

endif1:                                           ; preds = %loop
  %v4 = fcmp ogt float %v0, 0.000000e+00
  %5 = xor i1 %v4, true
  br i1 %5, label %endif2, label %Flow1

Flow:                                             ; preds = %Flow1, %loop
  %6 = phi i32 [ %3, %Flow1 ], [ undef, %loop ]
  %7 = phi i1 [ %4, %Flow1 ], [ true, %loop ]
  %8 = phi i1 [ false, %Flow1 ], [ true, %loop ]
  br i1 %7, label %Flow2, label %loop

endif2:                                           ; preds = %endif1
  %v5 = add i32 %v1, 1
  br label %Flow1

endloop:                                          ; preds = %if1, %Flow2
  %v6 = phi float [ 0.000000e+00, %Flow2 ], [ %v3, %if1 ]
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %v6, float %v6, float %v6, float %v6, i1 true, i1 true)
  ret void
}

; Function Attrs: nounwind readnone speculatable willreturn
declare float @llvm.sqrt.f32(float) #0

; Function Attrs: nounwind readnone speculatable
declare float @llvm.amdgcn.interp.p1(float, i32 immarg, i32 immarg, i32) #1

; Function Attrs: inaccessiblememonly nounwind writeonly
declare void @llvm.amdgcn.exp.f32(i32 immarg, i32 immarg, float, float, float, float, i1 immarg, i1 immarg) #2

attributes #0 = { nounwind readnone speculatable willreturn }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { inaccessiblememonly nounwind writeonly }
