; RUN: llc -mtriple=bpfel -filetype=obj -o - %s | llvm-objdump --no-print-imm-hex -d - | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=bpfeb -filetype=obj -o - %s | llvm-objdump --no-print-imm-hex -d - | FileCheck --check-prefix=CHECK %s

; src:
;   static volatile long a = 2;
;   static volatile int b = 3;
;   int test() { return a + b; }
@a = internal global i64 2, align 8
@b = internal global i32 3, align 4

; Function Attrs: norecurse nounwind
define dso_local i32 @test() local_unnamed_addr #0 {
  %1 = load volatile i64, ptr @a, align 8, !tbaa !2
; CHECK: r1 = 0 ll
; CHECK: r1 = *(u64 *)(r1 + 0)
  %2 = load volatile i32, ptr @b, align 4, !tbaa !6
; CHECK: r2 = 8 ll
; CHECK: r0 = *(u32 *)(r2 + 0)
  %3 = trunc i64 %1 to i32
  %4 = add i32 %2, %3
; CHECK: r0 += r1
  ret i32 %4
; CHECK: exit
}

attributes #0 = { norecurse nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 8.0.20181009 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"long", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !4, i64 0}
