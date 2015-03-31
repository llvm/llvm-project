; RUN: llc -march=amdgcn -mcpu=SI -mattr=-fp32-denormals,+fp64-denormals < %s | FileCheck -check-prefix=FP64-DENORMAL -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=SI -mattr=+fp32-denormals,-fp64-denormals < %s | FileCheck -check-prefix=FP32-DENORMAL -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=SI -mattr=+fp32-denormals,+fp64-denormals < %s | FileCheck -check-prefix=BOTH-DENORMAL -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=SI -mattr=-fp32-denormals,-fp64-denormals < %s | FileCheck -check-prefix=NO-DENORMAL -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=SI < %s | FileCheck -check-prefix=DEFAULT -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=SI -mattr=-fp32-denormals < %s | FileCheck -check-prefix=DEFAULT -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=SI -mattr=+fp64-denormals < %s | FileCheck -check-prefix=DEFAULT -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-fp32-denormals,+fp64-denormals < %s | FileCheck -check-prefix=FP64-DENORMAL -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=+fp32-denormals,-fp64-denormals < %s | FileCheck -check-prefix=FP32-DENORMAL -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=+fp32-denormals,+fp64-denormals < %s | FileCheck -check-prefix=BOTH-DENORMAL -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-fp32-denormals,-fp64-denormals < %s | FileCheck -check-prefix=NO-DENORMAL -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=DEFAULT -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-fp32-denormals < %s | FileCheck -check-prefix=DEFAULT -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=+fp64-denormals < %s | FileCheck -check-prefix=DEFAULT -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}test_kernel:

; DEFAULT: FloatMode: 192
; DEFAULT: IeeeMode: 0

; FP64-DENORMAL: FloatMode: 192
; FP64-DENORMAL: IeeeMode: 0

; FP32-DENORMAL: FloatMode: 48
; FP32-DENORMAL: IeeeMode: 0

; BOTH-DENORMAL: FloatMode: 240
; BOTH-DENORMAL: IeeeMode: 0

; NO-DENORMAL: FloatMode: 0
; NO-DENORMAL: IeeeMode: 0
define void @test_kernel(float addrspace(1)* %out0, double addrspace(1)* %out1) nounwind {
  store float 0.0, float addrspace(1)* %out0
  store double 0.0, double addrspace(1)* %out1
  ret void
}
