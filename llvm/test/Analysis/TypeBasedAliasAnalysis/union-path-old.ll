; RUN: opt < %s -aa-pipeline=tbaa -passes=aa-eval -evaluate-aa-metadata \
; RUN:     -print-no-aliases -print-may-aliases -disable-output 2>&1 | \
; RUN:     FileCheck %s
; RUN: opt < %s -aa-pipeline=tbaa -passes=gvn -S | FileCheck %s --check-prefix=OPT
;
; Check various union use cases with old struct path TBAA.

; IR generated from following C code:
;
; // Case 1:  Union with array field.
;
; union A{
;  int a[5];
;  double g;
; };
;
; // MayAlias.
; double f1(union A* a) {
;   a->g = 2.0;
;   a->a[1] = 5;
;   return a->g;
; }
;
; // Case 2: Union with struct and primitive type.
;
; struct S1{
;   int a;
;   float b;
;   int c;
; };
; 
; union S2{
;   struct S1 c;
;   double d;
; };
; 
; // MayAlias.
; double f2(union S2* u) {
;   u->d = 2.0;
;   u->c.b = 3.0;
;   return u->d;
; }
; // MayAlias.
; // Old struct path is conservative here.
; // (For reference see union-path-new.ll)
; double f3(union S2* u) {
;   u->d = 2.0;
;   u->c.c = 3;
;   return u->d;
; }
; 
; // Case 3: Union of two structs.
;
; struct FloatS{
;   float a;
;   float b;
; };
; 
; struct IntS{
;  short a;
;  short b;
;  char c;
; };
; 
; union SU {
;   struct FloatS a;
;   struct IntS b;
; };
; 
; // NoAlias. 
; float f4(union SU* u) {
;   u->a.a = 3.0;
;   u->b.c = 5;
;   return u->a.a;
; }
;
; // MayAlias.
; float f5(union SU* u) {
;   u->a.a = 3.0;
;   u->b.b = 5;
;   return u->a.a;
; }



define double @f1(ptr %0) {
entry:
; CHECK-LABEL: f1
; CHECK: MayAlias:   store i32 5, {{.*}} <-> store double 2.0
; OPT-LABEL: f1
; OPT: store double 2.0
; OPT: store i32 5,
; OPT: load double
; OPT: ret double
  store double 2.000000e+00, ptr %0, align 8, !tbaa !5
  %2 = getelementptr inbounds i8, ptr %0, i64 4
  store i32 5, ptr %2, align 4, !tbaa !11
  %3 = load double, ptr %0, align 8, !tbaa !5
  ret double %3
}

define double @f2(ptr %0) {
entry:
; CHECK-LABEL: f2
; CHECK: MayAlias:   store float 3.0{{.*}} <-> store double 2.0
; OPT-LABEL: f2
; OPT: store double 2.0
; OPT: store float 3.0
; OPT: load double
; OPT: ret double
  store double 2.000000e+00, ptr %0, align 8, !tbaa !12
  %2 = getelementptr inbounds i8, ptr %0, i64 4
  store float 3.000000e+00, ptr %2, align 4, !tbaa !16
  %3 = load double, ptr %0, align 8, !tbaa !12
  ret double %3
}

define double @f3(ptr %0) {
entry:
; CHECK-LABEL: f3
; CHECK: MayAlias:   store i32 3,{{.*}} <-> store double 2.0
; OPT-LABEL: f3
; OPT: store double 2.0
; OPT: store i32 3
; OPT: load double
; OPT: ret double
  store double 2.000000e+00, ptr %0, align 8, !tbaa !12
  %2 = getelementptr inbounds i8, ptr %0, i64 8
  store i32 3, ptr %2, align 8, !tbaa !17
  %3 = load double, ptr %0, align 8, !tbaa !12
  ret double %3
}

define float @f4(ptr %0) {
; CHECK-LABEL: f4
; CHECK: NoAlias:   store i8 5{{.*}} <-> store float 3.0
; OPT-LABEL: f4
; OPT: store float 3.0
; OPT: store i8 5
; OPT-NOT: load
; OPT: ret float 3.0
  store float 3.000000e+00, ptr %0, align 4, !tbaa !18
  %2 = getelementptr inbounds i8, ptr %0, i64 4
  store i8 5, ptr %2, align 4, !tbaa !23
  %3 = load float, ptr %0, align 4, !tbaa !18
  ret float %3
}

define float @f5(ptr %0) {
entry:
; CHECK-LABEL: f5
; CHECK: MayAlias: store i16 5, {{.*}} <-> store float 3.0
; OPT-LABEL: f5
; OPT: store float 3.0
; OPT: store i16 5
; OPT: load float
; OPT: ret float
  store float 3.000000e+00, ptr %0, align 4, !tbaa !18
  %2 = getelementptr inbounds i8, ptr %0, i64 2
  store i16 5, ptr %2, align 2, !tbaa !24
  %3 = load float, ptr %0, align 4, !tbaa !18
  ret float %3
}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!""}
!5 = !{!6, !10, i64 0}
!6 = !{!"A", !7, i64 0, !10, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!"double", !8, i64 0}
!11 = !{!6, !7, i64 0}
!12 = !{!13, !10, i64 0}
!13 = !{!"S2", !14, i64 0, !10, i64 0}
!14 = !{!"S1", !7, i64 0, !15, i64 4, !7, i64 8}
!15 = !{!"float", !8, i64 0}
!16 = !{!13, !15, i64 4}
!17 = !{!13, !7, i64 8}
!18 = !{!19, !15, i64 0}
!19 = !{!"SU", !20, i64 0, !21, i64 0}
!20 = !{!"FloatS", !15, i64 0, !15, i64 4}
!21 = !{!"IntS", !22, i64 0, !22, i64 2, !8, i64 4}
!22 = !{!"short", !8, i64 0}
!23 = !{!19, !8, i64 4}
!24 = !{!19, !22, i64 2}

