; RUN: llc -show-mc-encoding -mattr=+promote-alloca -verify-machineinstrs -march=amdgcn < %s | FileCheck %s -check-prefix=SI-PROMOTE -check-prefix=SI -check-prefix=FUNC
; RUN: llc -show-mc-encoding -mattr=+promote-alloca -verify-machineinstrs -mtriple=amdgcn--amdhsa -mcpu=kaveri < %s | FileCheck %s -check-prefix=SI-PROMOTE -check-prefix=SI -check-prefix=FUNC -check-prefix=HSA-PROMOTE
; RUN: llc -show-mc-encoding -mattr=-promote-alloca -verify-machineinstrs -march=amdgcn < %s | FileCheck %s -check-prefix=SI-ALLOCA -check-prefix=SI -check-prefix=FUNC
; RUN: llc -show-mc-encoding -mattr=-promote-alloca -verify-machineinstrs -mtriple=amdgcn-amdhsa -mcpu=kaveri < %s | FileCheck %s -check-prefix=SI-ALLOCA -check-prefix=SI -check-prefix=FUNC -check-prefix=HSA-ALLOCA
; RUN: llc -show-mc-encoding -mattr=+promote-alloca -verify-machineinstrs -march=amdgcn -mcpu=tonga < %s | FileCheck %s -check-prefix=SI-PROMOTE -check-prefix=SI -check-prefix=FUNC
; RUN: llc -show-mc-encoding -mattr=-promote-alloca -verify-machineinstrs -march=amdgcn -mcpu=tonga < %s | FileCheck %s -check-prefix=SI-ALLOCA -check-prefix=SI -check-prefix=FUNC


declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone


; Make sure we don't overwrite workitem information with private memory

; FUNC-LABEL: {{^}}work_item_info:

; SI-NOT: v_mov_b32_e{{(32|64)}} v0
define void @work_item_info(i32 addrspace(1)* %out, i32 %in) {
entry:
  %0 = alloca [2 x i32]
  %1 = getelementptr [2 x i32], [2 x i32]* %0, i32 0, i32 0
  %2 = getelementptr [2 x i32], [2 x i32]* %0, i32 0, i32 1
  store i32 0, i32* %1
  store i32 1, i32* %2
  %3 = getelementptr [2 x i32], [2 x i32]* %0, i32 0, i32 %in
  %4 = load i32, i32* %3
  %5 = call i32 @llvm.amdgcn.workitem.id.x()
  %6 = add i32 %4, %5
  store i32 %6, i32 addrspace(1)* %out
  ret void
}
