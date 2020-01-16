; RUN: not llc -mcpu=mips32r2 -march=mipsel -relocation-model=static < %s 2> %t
; RUN: FileCheck %s < %t

; CHECK: LLVM ERROR: Functions with the interrupt attribute cannot have arguments!
define i32 @isr_sw0(i8 signext %n) #0 {
  ret i32 0
}

attributes #0 = { "interrupt"="sw0" }
