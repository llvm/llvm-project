; RUN: llc -filetype=obj -relax-elf-relocations=true -mtriple=x86_64-linux-gnu -o - %s |  llvm-objdump - -d -r | FileCheck %s

; CHECK: jmpq *(%rip)
; CHECK-NEXT: R_X86_64_GOTPCRELX

define i32 @main() {
entry:
  %call = tail call i32 @foo()
  ret i32 %call
}

; Function Attrs: nonlazybind
declare i32 @foo() #1

attributes #1 = { nonlazybind }
