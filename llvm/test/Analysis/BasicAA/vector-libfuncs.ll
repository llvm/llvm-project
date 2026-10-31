; RUN: opt -mtriple=x86_64-pc-linux-gnu -vector-library=AMDLIBM -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 %s | FileCheck %s

; CHECK-LABEL: Function: test_vrs4_sincosf
; CHECK:      Just Mod:  Ptr: float* %s	<->  call void @amd_vrs4_sincosf(<4 x float> %x, ptr %s, ptr %c)
; CHECK-NEXT: NoModRef:  Ptr: float* %s.gep.4	<->  call void @amd_vrs4_sincosf(<4 x float> %x, ptr %s, ptr %c)
; CHECK-NEXT: Just Mod:  Ptr: float* %c.gep.3	<->  call void @amd_vrs4_sincosf(<4 x float> %x, ptr %s, ptr %c)
define void @test_vrs4_sincosf(<4 x float> %x, ptr noalias %s, ptr noalias %c) {
entry:
  call void @amd_vrs4_sincosf(<4 x float> %x, ptr %s, ptr %c)
  store float 0.0, ptr %s
  %s.gep.4 = getelementptr float, ptr %s, i32 4
  store float 1.0, ptr %s.gep.4
  %c.gep.3 = getelementptr float, ptr %c, i32 3
  store float 2.0, ptr %c.gep.3
  ret void
}

; The declaration does not match the vectorization factor of the vector
; function, so the size of the written memory is unknown.
; CHECK-LABEL: Function: test_vrs8_sincosf_mismatch
; CHECK:      Just Mod:  Ptr: float* %s	<->  call void @amd_vrs8_sincosf(<4 x float> %x, ptr %s, ptr %c)
; CHECK-NEXT: Just Mod:  Ptr: float* %s.gep.4	<->  call void @amd_vrs8_sincosf(<4 x float> %x, ptr %s, ptr %c)
define void @test_vrs8_sincosf_mismatch(<4 x float> %x, ptr noalias %s, ptr noalias %c) {
entry:
  call void @amd_vrs8_sincosf(<4 x float> %x, ptr %s, ptr %c)
  store float 0.0, ptr %s
  %s.gep.4 = getelementptr float, ptr %s, i32 4
  store float 1.0, ptr %s.gep.4
  ret void
}

declare void @amd_vrs4_sincosf(<4 x float>, ptr, ptr) memory(argmem: write)
declare void @amd_vrs8_sincosf(<4 x float>, ptr, ptr) memory(argmem: write)
