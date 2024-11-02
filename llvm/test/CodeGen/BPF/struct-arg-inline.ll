; RUN: opt -O2 -mtriple=bpf-pc-linux %s | llvm-dis > %t1
; RUN: llc %t1 -o - | FileCheck %s
; Source:
;   struct t1 {
;     long a;
;   };
;   struct t2 {
;     long a;
;     long b;
;   };
;   __attribute__((always_inline))
;   static long foo1(struct t2 a1, struct t1 a2, struct t1 a3, struct t1 a4,
;                    struct t1 a5, struct t2 a6) {
;     return a1.a + a2.a + a3.a + a4.a + a5.a + a6.a;
;   }
;   long foo2(struct t2 a1, struct t2 a2, struct t1 a3) {
;     return foo1(a1, a3, a3, a3, a3, a2);
;   }
; Compilation flags:
;   clang -target bpf -O2 -S -emit-llvm -Xclang -disable-llvm-passes t.c

%struct.t2 = type { i64, i64 }
%struct.t1 = type { i64 }

; Function Attrs: nounwind
define dso_local i64 @foo2([2 x i64] %a1.coerce, [2 x i64] %a2.coerce, i64 %a3.coerce) #0 {
entry:
  %a1 = alloca %struct.t2, align 8
  %a2 = alloca %struct.t2, align 8
  %a3 = alloca %struct.t1, align 8
  store [2 x i64] %a1.coerce, ptr %a1, align 8
  store [2 x i64] %a2.coerce, ptr %a2, align 8
  %coerce.dive = getelementptr inbounds %struct.t1, ptr %a3, i32 0, i32 0
  store i64 %a3.coerce, ptr %coerce.dive, align 8
  %0 = load [2 x i64], ptr %a1, align 8
  %coerce.dive1 = getelementptr inbounds %struct.t1, ptr %a3, i32 0, i32 0
  %1 = load i64, ptr %coerce.dive1, align 8
  %coerce.dive2 = getelementptr inbounds %struct.t1, ptr %a3, i32 0, i32 0
  %2 = load i64, ptr %coerce.dive2, align 8
  %coerce.dive3 = getelementptr inbounds %struct.t1, ptr %a3, i32 0, i32 0
  %3 = load i64, ptr %coerce.dive3, align 8
  %coerce.dive4 = getelementptr inbounds %struct.t1, ptr %a3, i32 0, i32 0
  %4 = load i64, ptr %coerce.dive4, align 8
  %5 = load [2 x i64], ptr %a2, align 8
  %call = call i64 @foo1([2 x i64] %0, i64 %1, i64 %2, i64 %3, i64 %4, [2 x i64] %5)
  ret i64 %call
; CHECK:             r0 = r3
; CHECK-NEXT:        r0 += r1
; CHECK-NEXT:        r5 <<= 2
; CHECK-NEXT:        r0 += r5
; CHECK-NEXT:        exit
}

; Function Attrs: alwaysinline nounwind
define internal i64 @foo1([2 x i64] %a1.coerce, i64 %a2.coerce, i64 %a3.coerce, i64 %a4.coerce, i64 %a5.coerce, [2 x i64] %a6.coerce) #1 {
entry:
  %a1 = alloca %struct.t2, align 8
  %a2 = alloca %struct.t1, align 8
  %a3 = alloca %struct.t1, align 8
  %a4 = alloca %struct.t1, align 8
  %a5 = alloca %struct.t1, align 8
  %a6 = alloca %struct.t2, align 8
  store [2 x i64] %a1.coerce, ptr %a1, align 8
  %coerce.dive = getelementptr inbounds %struct.t1, ptr %a2, i32 0, i32 0
  store i64 %a2.coerce, ptr %coerce.dive, align 8
  %coerce.dive1 = getelementptr inbounds %struct.t1, ptr %a3, i32 0, i32 0
  store i64 %a3.coerce, ptr %coerce.dive1, align 8
  %coerce.dive2 = getelementptr inbounds %struct.t1, ptr %a4, i32 0, i32 0
  store i64 %a4.coerce, ptr %coerce.dive2, align 8
  %coerce.dive3 = getelementptr inbounds %struct.t1, ptr %a5, i32 0, i32 0
  store i64 %a5.coerce, ptr %coerce.dive3, align 8
  store [2 x i64] %a6.coerce, ptr %a6, align 8
  %a = getelementptr inbounds %struct.t2, ptr %a1, i32 0, i32 0
  %0 = load i64, ptr %a, align 8, !tbaa !3
  %a7 = getelementptr inbounds %struct.t1, ptr %a2, i32 0, i32 0
  %1 = load i64, ptr %a7, align 8, !tbaa !8
  %add = add nsw i64 %0, %1
  %a8 = getelementptr inbounds %struct.t1, ptr %a3, i32 0, i32 0
  %2 = load i64, ptr %a8, align 8, !tbaa !8
  %add9 = add nsw i64 %add, %2
  %a10 = getelementptr inbounds %struct.t1, ptr %a4, i32 0, i32 0
  %3 = load i64, ptr %a10, align 8, !tbaa !8
  %add11 = add nsw i64 %add9, %3
  %a12 = getelementptr inbounds %struct.t1, ptr %a5, i32 0, i32 0
  %4 = load i64, ptr %a12, align 8, !tbaa !8
  %add13 = add nsw i64 %add11, %4
  %a14 = getelementptr inbounds %struct.t2, ptr %a6, i32 0, i32 0
  %5 = load i64, ptr %a14, align 8, !tbaa !3
  %add15 = add nsw i64 %add13, %5
  ret i64 %add15
}

attributes #0 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { alwaysinline nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 16.0.0 (https://github.com/llvm/llvm-project.git 9385660f4ca87d074410a84df89faca313afcb5a)"}
!3 = !{!4, !5, i64 0}
!4 = !{!"t2", !5, i64 0, !5, i64 8}
!5 = !{!"long", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{!9, !5, i64 0}
!9 = !{!"t1", !5, i64 0}
