; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #0
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #0

; SI-LABEL: {{^}}test_array_ptr_calc:
; SI-DAG: v_mul_lo_i32
; SI-DAG: v_mul_hi_i32
; SI: s_endpgm
define void @test_array_ptr_calc(i32 addrspace(1)* noalias %out, [1025 x i32] addrspace(1)* noalias %inA, i32 addrspace(1)* noalias %inB) {
  %mbcnt.lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %tid = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %mbcnt.lo)
  %a_ptr = getelementptr [1025 x i32], [1025 x i32] addrspace(1)* %inA, i32 %tid, i32 0
  %b_ptr = getelementptr i32, i32 addrspace(1)* %inB, i32 %tid
  %a = load i32, i32 addrspace(1)* %a_ptr
  %b = load i32, i32 addrspace(1)* %b_ptr
  %result = add i32 %a, %b
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
