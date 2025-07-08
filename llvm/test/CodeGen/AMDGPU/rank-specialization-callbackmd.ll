; RUN: opt -mtriple=amdgcn-amd-amdhsa -verify-each -passes=amdgpu-rank-specialization,instcombine,simplifycfg -S -o - %s | FileCheck --check-prefixes=CHECK %s

; Check for the required !callback metadata and that protected visibility is
; handled correctly

declare void @dummy_a()
declare void @dummy_b()

define protected amdgpu_kernel void @test() local_unnamed_addr "amdgpu-wavegroup-enable" {
entry:
  %waveId = call i32 @llvm.amdgcn.wave.id.in.wavegroup()
  %is.zero = icmp eq i32 %waveId, 0
  br i1 %is.zero, label %zero, label %other

zero:
  call void @dummy_a()
  ret void

other:
  call void @dummy_b()
  ret void
}

; CHECK: declare !callback !0 void @llvm.amdgcn.wavegroup.rank.p0(i32 immarg, ptr)
; CHECK: !0 = !{!1}
; CHECK: !1 = !{i64 1, i1 false}
