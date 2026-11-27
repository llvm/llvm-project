; RUN: llc -mtriple=mipsel -mcpu=mips32r2 -mattr=+micromips -asm-show-inst < %s | FileCheck %s
; RUN: llc -mtriple=mipsel -mcpu=mips32r6 -mattr=+micromips -asm-show-inst < %s | FileCheck %s

; FCSR copies must use the microMIPS cfc1/ctc1 encodings.

define i32 @get_rounding() #0 {
; CHECK-LABEL: get_rounding:
; CHECK: cfc1 ${{[0-9]+}}, $31 # <MCInst #{{[0-9]+}} CFC1_MM
  %r = call i32 @llvm.get.rounding()
  ret i32 %r
}

define void @set_rounding(i32 %m) #0 {
; CHECK-LABEL: set_rounding:
; CHECK: cfc1 ${{[0-9]+}}, $31 # <MCInst #{{[0-9]+}} CFC1_MM
; CHECK: ctc1 ${{[0-9]+}}, $31 # <MCInst #{{[0-9]+}} CTC1_MM
  call void @llvm.set.rounding(i32 %m)
  ret void
}

attributes #0 = { strictfp }
