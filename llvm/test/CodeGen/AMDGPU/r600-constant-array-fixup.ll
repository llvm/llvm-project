; RUN: llc -filetype=obj -march=r600 -mcpu=cypress -verify-machineinstrs < %s | llvm-readobj -r --symbols | FileCheck %s

@arr = internal unnamed_addr addrspace(4) constant [4 x i32] [i32 4, i32 5, i32 6, i32 7], align 4

; CHECK: Relocations [
; CHECK: Section (3) .rel.text {
; CHECK: 0x58 R_AMDGPU_ABS32 .text 0x0
; CHECK: }
; CHECK: ]

; CHECK: Symbol {
; CHECK: Name: arr (11)
; CHECK: Value: 0x70
; CHECK: Size: 16
; CHECK: Binding: Local (0x0)
; CHECK: Type: Object (0x1)
; CHECK: Other: 0
; CHECK: Section: .text (0x2)
; CHECK: }
define amdgpu_kernel void @test_constant_array_fixup(i32 addrspace(1)* nocapture %out, i32 %idx) #0 {
entry:
  %arrayidx = getelementptr inbounds [4 x i32], [4 x i32] addrspace(4)* @arr, i32 0, i32 %idx
  %val = load i32, i32 addrspace(4)* %arrayidx
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}

attributes #0 = { nounwind }
