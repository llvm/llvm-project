; Check that invalid type for constraint `c` causes an error message.
; RUN: not llc -mtriple=mips -target-abi o32 < %s 2>&1 | FileCheck %s

define i32 @main() #0 {
entry:
  %jmp = alloca float, align 4
  store float 0x4200000000000000, ptr %jmp, align 4
  %0 = load float, ptr %jmp, align 4
  call void asm sideeffect "jr $0", "c,~{$1}"(float %0) #1

; CHECK: error: couldn't allocate input reg for constraint 'c'

  ret i32 0
}

attributes #0 = { noinline nounwind }
attributes #1 = { nounwind }
