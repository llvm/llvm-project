; RUN: opt < %s -aa-pipeline=tbaa,basic-aa -passes=aa-eval -evaluate-aa-metadata -print-no-aliases -print-may-aliases -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -aa-pipeline=tbaa,basic-aa -passes=gvn -S | FileCheck %s --check-prefix=OPT
; Generated from clang/test/CodeGen/tbaa.cpp with "-O1 -new-struct-path-tbaa".

%struct.StructA = type { i16, i32, i16, i32 }
%struct.StructB = type { i16, %struct.StructA, i32 }
%struct.StructS = type { i16, i32 }
%struct.StructS2 = type { i16, i32 }
%struct.StructC = type { i16, %struct.StructB, i32 }
%struct.StructD = type { i16, %struct.StructB, i32, i8 }

; uint32_t g(uint32_t *s, StructA *A, uint64_t count) {
;   *s = 1;
;   A->f32 = 4;
;   return *s;
; }
;
define i32 @_Z1gPjP7StructAy(ptr nocapture %s, ptr nocapture %A, i64 %count) {
entry:
; CHECK-LABEL: Z1gPjP7StructAy
; CHECK: MayAlias: store i32 4, {{.*}} <-> store i32 1,
; OPT-LABEL: _Z1gPjP7StructAy
; OPT: store i32 1,
; OPT: store i32 4,
; OPT: %[[RET:.*]] = load i32,
; OPT: ret i32 %[[RET]]
  store i32 1, ptr %s, align 4, !tbaa !2
  %f32 = getelementptr inbounds %struct.StructA, ptr %A, i64 0, i32 1
  store i32 4, ptr %f32, align 4, !tbaa !6
  %0 = load i32, ptr %s, align 4, !tbaa !2
  ret i32 %0
}

; uint32_t g2(uint32_t *s, StructA *A, uint64_t count) {
;   *s = 1;
;   A->f16 = 4;
;   return *s;
; }
;
define i32 @_Z2g2PjP7StructAy(ptr nocapture %s, ptr nocapture %A, i64 %count) {
entry:
; CHECK-LABEL: _Z2g2PjP7StructAy
; CHECK: NoAlias: store i16 4, {{.*}} <-> store i32 1,
; OPT-LABEL: _Z2g2PjP7StructAy
; OPT: store i32 1,
; OPT: store i16 4,
; Remove a load and propagate the value from store.
; OPT: ret i32 1
  store i32 1, ptr %s, align 4, !tbaa !2
  store i16 4, ptr %A, align 4, !tbaa !9
  ret i32 1
}

; uint32_t g3(StructA *A, StructB *B, uint64_t count) {
;   A->f32 = 1;
;   B->a.f32 = 4;
;   return A->f32;
; }
;
define i32 @_Z2g3P7StructAP7StructBy(ptr nocapture %A, ptr nocapture %B, i64 %count) {
entry:
; CHECK-LABEL: _Z2g3P7StructAP7StructBy
; CHECK: MayAlias: store i32 4, {{.*}} <-> store i32 1,
; OPT-LABEL: _Z2g3P7StructAP7StructBy
; OPT: store i32 1
; OPT: store i32 4
; OPT: %[[RET:.*]] = load i32,
; OPT: ret i32 %[[RET]]
  %f32 = getelementptr inbounds %struct.StructA, ptr %A, i64 0, i32 1
  store i32 1, ptr %f32, align 4, !tbaa !6
  %f321 = getelementptr inbounds %struct.StructB, ptr %B, i64 0, i32 1, i32 1
  store i32 4, ptr %f321, align 4, !tbaa !10
  %0 = load i32, ptr %f32, align 4, !tbaa !6
  ret i32 %0
}

; uint32_t g4(StructA *A, StructB *B, uint64_t count) {
;   A->f32 = 1;
;   B->a.f16 = 4;
;   return A->f32;
; }
;
define i32 @_Z2g4P7StructAP7StructBy(ptr nocapture %A, ptr nocapture %B, i64 %count) {
entry:
; CHECK-LABEL: _Z2g4P7StructAP7StructBy
; CHECK: NoAlias: store i16 4, {{.*}} <-> store i32 1,
; OPT-LABEL: _Z2g4P7StructAP7StructBy
; OPT: store i32 1,
; OPT: store i16 4,
; Remove a load and propagate the value from store.
; OPT: ret i32 1
  %f32 = getelementptr inbounds %struct.StructA, ptr %A, i64 0, i32 1
  store i32 1, ptr %f32, align 4, !tbaa !6
  %f16 = getelementptr inbounds %struct.StructB, ptr %B, i64 0, i32 1, i32 0
  store i16 4, ptr %f16, align 4, !tbaa !12
  ret i32 1
}

; uint32_t g5(StructA *A, StructB *B, uint64_t count) {
;   A->f32 = 1;
;   B->f32 = 4;
;   return A->f32;
; }
;
define i32 @_Z2g5P7StructAP7StructBy(ptr nocapture %A, ptr nocapture %B, i64 %count) {
entry:
; CHECK-LABEL: _Z2g5P7StructAP7StructBy
; CHECK: NoAlias: store i32 4, {{.*}} <-> store i32 1,
; OPT-LABEL: _Z2g5P7StructAP7StructBy
; OPT: store i32 1,
; OPT: store i32 4,
; Remove a load and propagate the value from store.
; OPT: ret i32 1
  %f32 = getelementptr inbounds %struct.StructA, ptr %A, i64 0, i32 1
  store i32 1, ptr %f32, align 4, !tbaa !6
  %f321 = getelementptr inbounds %struct.StructB, ptr %B, i64 0, i32 2
  store i32 4, ptr %f321, align 4, !tbaa !13
  ret i32 1
}

; uint32_t g6(StructA *A, StructB *B, uint64_t count) {
;   A->f32 = 1;
;   B->a.f32_2 = 4;
;   return A->f32;
; }
;
define i32 @_Z2g6P7StructAP7StructBy(ptr nocapture %A, ptr nocapture %B, i64 %count) {
entry:
; CHECK-LABEL: _Z2g6P7StructAP7StructBy
; CHECK: NoAlias: store i32 4, {{.*}} <-> store i32 1,
; OPT-LABEL: _Z2g6P7StructAP7StructBy
; OPT: store i32 1,
; OPT: store i32 4,
; Remove a load and propagate the value from store.
; OPT: ret i32 1
  %f32 = getelementptr inbounds %struct.StructA, ptr %A, i64 0, i32 1
  store i32 1, ptr %f32, align 4, !tbaa !6
  %f32_2 = getelementptr inbounds %struct.StructB, ptr %B, i64 0, i32 1, i32 3
  store i32 4, ptr %f32_2, align 4, !tbaa !14
  ret i32 1
}

; uint32_t g7(StructA *A, StructS *S, uint64_t count) {
;   A->f32 = 1;
;   S->f32 = 4;
;   return A->f32;
; }
;
define i32 @_Z2g7P7StructAP7StructSy(ptr nocapture %A, ptr nocapture %S, i64 %count) {
entry:
; CHECK-LABEL: _Z2g7P7StructAP7StructSy
; CHECK: NoAlias: store i32 4, {{.*}} <-> store i32 1,
; OPT-LABEL: _Z2g7P7StructAP7StructSy
; OPT: store i32 1,
; OPT: store i32 4,
; Remove a load and propagate the value from store.
; OPT: ret i32 1
  %f32 = getelementptr inbounds %struct.StructA, ptr %A, i64 0, i32 1
  store i32 1, ptr %f32, align 4, !tbaa !6
  %f321 = getelementptr inbounds %struct.StructS, ptr %S, i64 0, i32 1
  store i32 4, ptr %f321, align 4, !tbaa !15
  ret i32 1
}

; uint32_t g8(StructA *A, StructS *S, uint64_t count) {
;   A->f32 = 1;
;   S->f16 = 4;
;   return A->f32;
; }
;
define i32 @_Z2g8P7StructAP7StructSy(ptr nocapture %A, ptr nocapture %S, i64 %count) {
entry:
; CHECK-LABEL: _Z2g8P7StructAP7StructSy
; CHECK: NoAlias: store i16 4, {{.*}} <-> store i32 1,
; OPT-LABEL: _Z2g8P7StructAP7StructSy
; OPT: store i32 1,
; OPT: store i16 4,
; Remove a load and propagate the value from store.
; OPT: ret i32 1
  %f32 = getelementptr inbounds %struct.StructA, ptr %A, i64 0, i32 1
  store i32 1, ptr %f32, align 4, !tbaa !6
  store i16 4, ptr %S, align 4, !tbaa !17
  ret i32 1
}

; uint32_t g9(StructS *S, StructS2 *S2, uint64_t count) {
;   S->f32 = 1;
;   S2->f32 = 4;
;   return S->f32;
; }
;
define i32 @_Z2g9P7StructSP8StructS2y(ptr nocapture %S, ptr nocapture %S2, i64 %count) {
entry:
; CHECK-LABEL: _Z2g9P7StructSP8StructS2y
; CHECK: NoAlias: store i32 4, {{.*}} <-> store i32 1,
; OPT-LABEL: _Z2g9P7StructSP8StructS2y
; OPT: store i32 1,
; OPT: store i32 4,
; Remove a load and propagate the value from store.
; OPT: ret i32 1
  %f32 = getelementptr inbounds %struct.StructS, ptr %S, i64 0, i32 1
  store i32 1, ptr %f32, align 4, !tbaa !15
  %f321 = getelementptr inbounds %struct.StructS2, ptr %S2, i64 0, i32 1
  store i32 4, ptr %f321, align 4, !tbaa !18
  ret i32 1
}

; uint32_t g10(StructS *S, StructS2 *S2, uint64_t count) {
;   S->f32 = 1;
;   S2->f16 = 4;
;   return S->f32;
; }
;
define i32 @_Z3g10P7StructSP8StructS2y(ptr nocapture %S, ptr nocapture %S2, i64 %count) {
entry:
; CHECK-LABEL: _Z3g10P7StructSP8StructS2y
; CHECK: NoAlias: store i16 4, {{.*}} <-> store i32 1,
; OPT-LABEL: _Z3g10P7StructSP8StructS2y
; OPT: store i32 1,
; OPT: store i16 4,
; Remove a load and propagate the value from store.
; OPT: ret i32 1
  %f32 = getelementptr inbounds %struct.StructS, ptr %S, i64 0, i32 1
  store i32 1, ptr %f32, align 4, !tbaa !15
  store i16 4, ptr %S2, align 4, !tbaa !20
  ret i32 1
}

; uint32_t g11(StructC *C, StructD *D, uint64_t count) {
;   C->b.a.f32 = 1;
;   D->b.a.f32 = 4;
;   return C->b.a.f32;
; }
;
define i32 @_Z3g11P7StructCP7StructDy(ptr nocapture %C, ptr nocapture %D, i64 %count) {
entry:
; CHECK-LABEL: _Z3g11P7StructCP7StructDy
; CHECK: NoAlias: store i32 4, {{.*}} <-> store i32 1,
; OPT-LABEL: _Z3g11P7StructCP7StructDy
; OPT: store i32 1,
; OPT: store i32 4,
; Remove a load and propagate the value from store.
; OPT: ret i32 1
  %f32 = getelementptr inbounds %struct.StructC, ptr %C, i64 0, i32 1, i32 1, i32 1
  store i32 1, ptr %f32, align 4, !tbaa !21
  %f323 = getelementptr inbounds %struct.StructD, ptr %D, i64 0, i32 1, i32 1, i32 1
  store i32 4, ptr %f323, align 4, !tbaa !23
  ret i32 1
}

; uint32_t g12(StructC *C, StructD *D, uint64_t count) {
;   StructB *b1 = &(C->b);
;   StructB *b2 = &(D->b);
;   // b1, b2 have different context.
;   b1->a.f32 = 1;
;   b2->a.f32 = 4;
;   return b1->a.f32;
; }
;
define i32 @_Z3g12P7StructCP7StructDy(ptr nocapture %C, ptr nocapture %D, i64 %count) {
entry:
; CHECK-LABEL: _Z3g12P7StructCP7StructDy
; CHECK: MayAlias: store i32 4, {{.*}} <-> store i32 1,
; OPT-LABEL: _Z3g12P7StructCP7StructDy
; OPT: store i32 1,
; OPT: store i32 4,
; OPT: %[[RET:.*]] = load i32,
; OPT: ret i32 %[[RET]]
  %f32 = getelementptr inbounds %struct.StructC, ptr %C, i64 0, i32 1, i32 1, i32 1
  store i32 1, ptr %f32, align 4, !tbaa !10
  %f325 = getelementptr inbounds %struct.StructD, ptr %D, i64 0, i32 1, i32 1, i32 1
  store i32 4, ptr %f325, align 4, !tbaa !10
  %0 = load i32, ptr %f32, align 4, !tbaa !10
  ret i32 %0
}

!2 = !{!3, !3, i64 0, i64 4}
!3 = !{!4, i64 4, !"int"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !3, i64 4, i64 4}
!7 = !{!4, i64 16, !"_ZTS7StructA", !8, i64 0, i64 2, !3, i64 4, i64 4, !8, i64 8, i64 2, !3, i64 12, i64 4}
!8 = !{!4, i64 2, !"short"}
!9 = !{!7, !8, i64 0, i64 2}
!10 = !{!11, !3, i64 8, i64 4}
!11 = !{!4, i64 24, !"_ZTS7StructB", !8, i64 0, i64 2, !7, i64 4, i64 16, !3, i64 20, i64 4}
!12 = !{!11, !8, i64 4, i64 2}
!13 = !{!11, !3, i64 20, i64 4}
!14 = !{!11, !3, i64 16, i64 4}
!15 = !{!16, !3, i64 4, i64 4}
!16 = !{!4, i64 8, !"_ZTS7StructS", !8, i64 0, i64 2, !3, i64 4, i64 4}
!17 = !{!16, !8, i64 0, i64 2}
!18 = !{!19, !3, i64 4, i64 4}
!19 = !{!4, i64 8, !"_ZTS8StructS2", !8, i64 0, i64 2, !3, i64 4, i64 4}
!20 = !{!19, !8, i64 0, i64 2}
!21 = !{!22, !3, i64 12, i64 4}
!22 = !{!4, i64 32, !"_ZTS7StructC", !8, i64 0, i64 2, !11, i64 4, i64 24, !3, i64 28, i64 4}
!23 = !{!24, !3, i64 12, i64 4}
!24 = !{!4, i64 36, !"_ZTS7StructD", !8, i64 0, i64 2, !11, i64 4, i64 24, !3, i64 28, i64 4, !4, i64 32, i64 1}
!25 = !{!26, !4, i64 1, i64 1}
!26 = !{!4, i64 3, !"_ZTS4five", !4, i64 0, i64 1, !3, i64 1, i64 4, !4, i64 1, i64 1, !4, i64 2, i64 1}
!27 = !{!28, !4, i64 4, i64 1}
!28 = !{!4, i64 6, !"_ZTS3six", !4, i64 0, i64 1, !3, i64 4, i64 4, !4, i64 4, i64 1, !4, i64 5, i64 1}
