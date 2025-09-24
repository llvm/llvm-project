; RUN: opt < %s -S | FileCheck %s

define void @store_unsignedptr(ptr %ptr) {
entry:
  store ptr null, ptr %ptr, align 8, !tbaa !0
  ret void
}

define void @store_char(ptr %ptr) {
entry:
  store i8 0, ptr %ptr, align 1, !tbaa !5
  ret void
}

define float @ptr_to_float(ptr %ptr) {
entry:
  store float 0.000000e+00, ptr %ptr, align 4, !tbaa !6
  call void @opaque(ptr %ptr)
  %val = load float, ptr %ptr, align 4, !tbaa !6
  ret float %val
}

define i64 @ptr_to_longlong(ptr %ptr) {
entry:
  %val = load i64, ptr %ptr, align 8, !tbaa !8
  store i64 0, ptr %ptr, align 8, !tbaa !8
  ret i64 %val
}

; struct STRUCT1 {
;   int x;
;   int y;
; };

define void @store_struct1ptr(ptr %ptr) {
entry:
  ; *(struct STRUCT1 **)ptr = 0;
  store ptr null, ptr %ptr, align 8, !tbaa !10
  ret void
}

; struct STRUCT2 {
;   struct STRUCT1 *s;
; };

define void @store_struct2(ptr %ptr) {
entry:
  ; ptr->s = 0;
  store ptr null, ptr %ptr, align 8, !tbaa !12
  ret void
}

define double @access_matrix(ptr %ptr) {
entry:
  %alloca.ptr = alloca ptr, align 8
  store ptr %ptr, ptr %alloca.ptr, align 8, !tbaa !14
  %ptr.idx = load ptr, ptr %alloca.ptr, align 8, !tbaa !14
  %add.ptr = getelementptr inbounds ptr, ptr %ptr.idx, i64 4
  %ptr.idx.1 = load ptr, ptr %add.ptr, align 8, !tbaa !16
  %add.ptr1 = getelementptr inbounds [6 x double], ptr %ptr.idx.1, i64 6
  %ptr.idx.2 = load <6 x double>, ptr %add.ptr1, align 8, !tbaa !5
  %matrixext = extractelement <6 x double> %ptr.idx.2, i64 5
  ret double %matrixext
}

declare void @opaque(ptr)

!0 = !{!1, !1, i64 0}
!1 = !{!"p1 int", !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!3, !3, i64 0}
!6 = !{!7, !7, i64 0}
!7 = !{!"float", !3, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"long long", !3, i64 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"p1 _ZTS7STRUCT1", !2, i64 0}
!12 = !{!13, !11, i64 0}
!13 = !{!"STRUCT2", !11, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"any p2 pointer", !2, i64 0}
!16 = !{!2, !2, i64 0}
