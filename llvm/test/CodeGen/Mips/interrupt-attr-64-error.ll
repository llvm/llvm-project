; RUN: not --crash llc -mcpu=mips64r6 -march=mipsel -target-abi n64 -relocation-model=static < %s 2>%t
; RUN: FileCheck %s < %t

; CHECK: LLVM ERROR: "interrupt" attribute is only supported for the O32 ABI on MIPS32R2+ at the present time.
define i32 @isr_sw0() #0 {
  ret i32 0
}

attributes #0 = { "interrupt"="sw0" }
