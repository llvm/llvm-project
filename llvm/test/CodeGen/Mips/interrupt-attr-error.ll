; RUN: not --crash llc -mcpu=mips32 -mtriple=mipsel -relocation-model=static < %s 2>%t
; RUN: FileCheck %s < %t

; CHECK: LLVM ERROR: "interrupt" attribute is not supported on pre-MIPS32R2 or MIPS16 targets.
define i32 @isr_sw0() #0 {
  ret i32 0
}

attributes #0 = { "interrupt"="sw0" }
