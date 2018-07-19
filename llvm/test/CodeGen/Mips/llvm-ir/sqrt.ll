; RUN: llc < %s -march=mipsel -mcpu=mips32r2 -mattr=+micromips | FileCheck %s
; RUN: llc < %s -march=mips -mcpu=mips32r2 -mattr=+micromips | FileCheck %s
; RUN: llc < %s -march=mips -mcpu=mips32r6 -mattr=+micromips | FileCheck %s

define float @sqrt_fn(float %value) #0 {
entry:
  %sqrtf = tail call float @sqrtf(float %value) #0
  ret float %sqrtf
}

declare float @sqrtf(float)

; CHECK: sqrt.s $f0, $f12
