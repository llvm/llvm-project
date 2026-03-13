; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mtriple=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   int g __attribute__((section("maps"))) = 5;
;   int test() { return g; }
; Compilation flag:
;   clang -target bpf -O2 -S -emit-llvm t.c

@g = dso_local local_unnamed_addr global i32 5, section "maps", align 4

; Function Attrs: norecurse nounwind readonly
define dso_local i32 @test() local_unnamed_addr {
  %1 = load i32, ptr @g, align 4, !tbaa !2
  ret i32 %1
}

; CHECK-NOT:         .section        .BTF
; CHECK-NOT:         .section        .BTF.ext

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 8.0.20181009 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
