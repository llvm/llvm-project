; RUN: llc -march=r600 -mcpu=cypress < %s | FileCheck %s -check-prefix=R600 -check-prefix=FUNC

declare i32 @llvm.r600.read.tidig.x() nounwind readnone


; Make sure we don't overwrite workitem information with private memory

; FUNC-LABEL: {{^}}work_item_info:
; R600-NOT: MOV T0.X
; Additional check in case the move ends up in the last slot
; R600-NOT: MOV * TO.X

define amdgpu_kernel void @work_item_info(i32 addrspace(1)* %out, i32 %in) {
entry:
  %0 = alloca [2 x i32], addrspace(5)
  %1 = getelementptr [2 x i32], [2 x i32] addrspace(5)* %0, i32 0, i32 0
  %2 = getelementptr [2 x i32], [2 x i32] addrspace(5)* %0, i32 0, i32 1
  store i32 0, i32 addrspace(5)* %1
  store i32 1, i32 addrspace(5)* %2
  %3 = getelementptr [2 x i32], [2 x i32] addrspace(5)* %0, i32 0, i32 %in
  %4 = load i32, i32 addrspace(5)* %3
  %5 = call i32 @llvm.r600.read.tidig.x()
  %6 = add i32 %4, %5
  store i32 %6, i32 addrspace(1)* %out
  ret void
}
