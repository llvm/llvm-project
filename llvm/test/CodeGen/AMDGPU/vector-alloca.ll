; RUN: llc -march=amdgcn -mcpu=verde -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefix=SI-ALLOCA -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=verde -mattr=+promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefix=SI-PROMOTE -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefix=SI-ALLOCA -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=+promote-alloca -verify-machineinstrs < %s | FileCheck -check-prefix=SI-PROMOTE -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck --check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}vector_read:
; EG: MOV
; EG: MOV
; EG: MOV
; EG: MOV
; EG: MOVA_INT
define void @vector_read(i32 addrspace(1)* %out, i32 %index) {
entry:
  %0 = alloca [4 x i32]
  %x = getelementptr [4 x i32], [4 x i32]* %0, i32 0, i32 0
  %y = getelementptr [4 x i32], [4 x i32]* %0, i32 0, i32 1
  %z = getelementptr [4 x i32], [4 x i32]* %0, i32 0, i32 2
  %w = getelementptr [4 x i32], [4 x i32]* %0, i32 0, i32 3
  store i32 0, i32* %x
  store i32 1, i32* %y
  store i32 2, i32* %z
  store i32 3, i32* %w
  %1 = getelementptr [4 x i32], [4 x i32]* %0, i32 0, i32 %index
  %2 = load i32, i32* %1
  store i32 %2, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}vector_write:
; EG: MOV
; EG: MOV
; EG: MOV
; EG: MOV
; EG: MOVA_INT
; EG: MOVA_INT
define void @vector_write(i32 addrspace(1)* %out, i32 %w_index, i32 %r_index) {
entry:
  %0 = alloca [4 x i32]
  %x = getelementptr [4 x i32], [4 x i32]* %0, i32 0, i32 0
  %y = getelementptr [4 x i32], [4 x i32]* %0, i32 0, i32 1
  %z = getelementptr [4 x i32], [4 x i32]* %0, i32 0, i32 2
  %w = getelementptr [4 x i32], [4 x i32]* %0, i32 0, i32 3
  store i32 0, i32* %x
  store i32 0, i32* %y
  store i32 0, i32* %z
  store i32 0, i32* %w
  %1 = getelementptr [4 x i32], [4 x i32]* %0, i32 0, i32 %w_index
  store i32 1, i32* %1
  %2 = getelementptr [4 x i32], [4 x i32]* %0, i32 0, i32 %r_index
  %3 = load i32, i32* %2
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; This test should be optimize to:
; store i32 0, i32 addrspace(1)* %out
; FUNC-LABEL: {{^}}bitcast_gep:
; EG: STORE_RAW
define void @bitcast_gep(i32 addrspace(1)* %out, i32 %w_index, i32 %r_index) {
entry:
  %0 = alloca [4 x i32]
  %x = getelementptr [4 x i32], [4 x i32]* %0, i32 0, i32 0
  %y = getelementptr [4 x i32], [4 x i32]* %0, i32 0, i32 1
  %z = getelementptr [4 x i32], [4 x i32]* %0, i32 0, i32 2
  %w = getelementptr [4 x i32], [4 x i32]* %0, i32 0, i32 3
  store i32 0, i32* %x
  store i32 0, i32* %y
  store i32 0, i32* %z
  store i32 0, i32* %w
  %1 = getelementptr [4 x i32], [4 x i32]* %0, i32 0, i32 1
  %2 = bitcast i32* %1 to [4 x i32]*
  %3 = getelementptr [4 x i32], [4 x i32]* %2, i32 0, i32 0
  %4 = load i32, i32* %3
  store i32 %4, i32 addrspace(1)* %out
  ret void
}
