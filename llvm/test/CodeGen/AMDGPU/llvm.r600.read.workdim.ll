; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG %s

; EG-LABEL: {{^}}read_workdim:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[2].Z
define void @read_workdim(i32 addrspace(1)* %out) {
entry:
  %dim = call i32 @llvm.r600.read.workdim() #0
  store i32 %dim, i32 addrspace(1)* %out
  ret void
}

; EG-LABEL: {{^}}read_workdim_known_bits:
define void @read_workdim_known_bits(i32 addrspace(1)* %out) {
entry:
  %dim = call i32 @llvm.r600.read.workdim() #0
  %shl = shl i32 %dim, 24
  %shr = lshr i32 %shl, 24
  store i32 %shr, i32 addrspace(1)* %out
  ret void
}

; EG-LABEL: {{^}}legacy_read_workdim:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[2].Z
define void @legacy_read_workdim(i32 addrspace(1)* %out) {
entry:
  %dim = call i32 @llvm.AMDGPU.read.workdim() #0
  store i32 %dim, i32 addrspace(1)* %out
  ret void
}

declare i32 @llvm.r600.read.workdim() #0
declare i32 @llvm.AMDGPU.read.workdim() #0

attributes #0 = { nounwind readnone }
