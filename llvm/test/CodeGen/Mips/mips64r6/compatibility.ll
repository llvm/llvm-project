; RUN: llc -march=mipsel -mcpu=mips64r6 -target-abi n64 < %s | FileCheck %s
; RUN: not llc -march=mipsel -mcpu=mips64r6 -target-abi n64 -mattr=+dsp < %s 2>&1 | FileCheck --check-prefix=DSP %s

; CHECK: foo:
; DSP: MIPS64r6 is not compatible with the DSP ASE

define void @foo() nounwind {
  ret void
}
