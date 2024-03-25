; RUN: llc -mtriple=bpfel -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
;
; Source:
;   int test() {
;     asm volatile("r0 = r0" ::);
;     return 0;
;   }
; Compilation flag:
;   clang -target bpf -O2 -S -emit-llvm t.c

; Function Attrs: nounwind
define dso_local i32 @test() local_unnamed_addr {
entry:
  tail call void asm sideeffect "r0 = r0", ""()
  ret i32 0
}
; CHECK-LABEL: test
; CHECK:       r0 = r0
; CHECK:       r0 = 0
