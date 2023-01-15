; Check handling of the constraint `c`.
; RUN: llc -march=mips -target-abi o32 < %s | FileCheck %s

define i32 @main() #0 {
entry:
  %jmp = alloca i32, align 4
  store i32 0, ptr %jmp, align 4
  %0 = load i32, ptr %jmp, align 4
  call void asm sideeffect "jr $0", "c,~{$1}"(i32 %0) #1

; CHECK: addiu   $25, $zero, 0
; CHECK: jr      $25

  ret i32 0
}

attributes #0 = { noinline nounwind }
attributes #1 = { nounwind }
