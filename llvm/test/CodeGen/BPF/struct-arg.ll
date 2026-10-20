; RUN: llc < %s -mtriple=bpfel | FileCheck %s
; Source:
;   struct t1 {
;     long a;
;   };
;   struct t2 {
;     long a;
;     long b;
;   };
;   long foo2(struct t1 a1, struct t1 a2, struct t1 a3, struct t1 a4, struct t1 a5) {
;     return a1.a + a2.a + a3.a + a4.a + a5.a;
;   }
;   long foo3(struct t2 a1, struct t2 a2, struct t1 a3) {
;     return a1.a + a2.a + a3.a;
;   }
; Compilation flags:
;   clang -target bpf -S -emit-llvm -O2 t.c

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define dso_local i64 @foo2(i64 %a1.coerce, i64 %a2.coerce, i64 %a3.coerce, i64 %a4.coerce, i64 %a5.coerce) local_unnamed_addr #0 {
entry:
  %add = add nsw i64 %a2.coerce, %a1.coerce
  %add8 = add nsw i64 %add, %a3.coerce
  %add10 = add nsw i64 %add8, %a4.coerce
  %add12 = add nsw i64 %add10, %a5.coerce
  ret i64 %add12
; CHECK:        r0 = r2
; CHECK:        r0 += r1
; CHECK:        r0 += r3
; CHECK:        r0 += r4
; CHECK:        r0 += r5
; CHECK-NEXT:   exit
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define dso_local i64 @foo3([2 x i64] %a1.coerce, [2 x i64] %a2.coerce, i64 %a3.coerce) local_unnamed_addr #0 {
entry:
  %a1.coerce.fca.0.extract = extractvalue [2 x i64] %a1.coerce, 0
  %a2.coerce.fca.0.extract = extractvalue [2 x i64] %a2.coerce, 0
  %add = add nsw i64 %a2.coerce.fca.0.extract, %a1.coerce.fca.0.extract
  %add6 = add nsw i64 %add, %a3.coerce
  ret i64 %add6
; CHECK:        r0 = r3
; CHECK-NEXT:   r0 += r1
; CHECK-NEXT:   r0 += r5
; CHECK-NEXT:   exit
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind readnone willreturn "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 16.0.0 (https://github.com/llvm/llvm-project.git 9385660f4ca87d074410a84df89faca313afcb5a)"}
