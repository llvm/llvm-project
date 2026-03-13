; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mtriple=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   int test(int arg) { return arg; }
; Compilation flag:
;   clang -target bpf -O2 -S -emit-llvm t.c

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @test(i32 returned) local_unnamed_addr #0 {
  ret i32 %0
}

; CHECK-NOT: BTF

attributes #0 = { norecurse nounwind readnone }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 8.0.20181009 "}
