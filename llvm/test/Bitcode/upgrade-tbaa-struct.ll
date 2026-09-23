; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder < %s

; Old-style scalar tags used as !tbaa.struct field tags are auto-upgraded to
; the struct-path aware format, like top-level !tbaa. Null and already
; struct-path field tags are left unchanged; an immutability flag is kept.

define void @copy(ptr %a, ptr %b) {
  call void @llvm.memcpy.p0.p0.i64(ptr %a, ptr %b, i64 12, i1 false), !tbaa.struct !0
; CHECK: !tbaa.struct [[TS:![0-9]+]]
  ret void
}

define void @copy_flag_and_structpath(ptr %a, ptr %b) {
  call void @llvm.memcpy.p0.p0.i64(ptr %a, ptr %b, i64 8, i1 false), !tbaa.struct !5
; CHECK: !tbaa.struct [[TS2:![0-9]+]]
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

; Old-style 2-operand scalar field tags and a null field tag.
!0 = !{i64 0, i64 4, !1, i64 4, i64 4, !3, i64 8, i64 4, null}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !4}
!3 = !{!"float", !2}
!4 = !{!"Simple C/C++ TBAA"}

; Old-style 3-operand (immutable) field tag and an already struct-path field tag.
!5 = !{i64 0, i64 4, !6, i64 4, i64 4, !7}
!6 = !{!"const int", !2, i64 1}
!7 = !{!8, !8, i64 0}
!8 = !{!"double", !2}

; CHECK: [[TS]] = !{i64 0, i64 4, [[TAG_INT:![0-9]+]], i64 4, i64 4, [[TAG_FLOAT:![0-9]+]], i64 8, i64 4, null}
; CHECK: [[TAG_INT]] = !{[[TYPE_INT:![0-9]+]], [[TYPE_INT]], i64 0}
; CHECK: [[TYPE_INT]] = !{!"int", [[TYPE_CHAR:![0-9]+]]}
; CHECK: [[TYPE_CHAR]] = !{!"omnipotent char", !{{[0-9]+}}}
; CHECK: [[TAG_FLOAT]] = !{[[TYPE_FLOAT:![0-9]+]], [[TYPE_FLOAT]], i64 0}
; CHECK: [[TYPE_FLOAT]] = !{!"float", [[TYPE_CHAR]]}
; The immutability flag (i64 1) is preserved; the already struct-path tag is unchanged.
; CHECK: [[TS2]] = !{i64 0, i64 4, [[TAG_CONST:![0-9]+]], i64 4, i64 4, [[TAG_DOUBLE:![0-9]+]]}
; CHECK: [[TAG_CONST]] = !{[[TYPE_CONST:![0-9]+]], [[TYPE_CONST]], i64 0, i64 1}
; CHECK: [[TYPE_CONST]] = !{!"const int", [[TYPE_CHAR]]}
; CHECK: [[TAG_DOUBLE]] = !{[[TYPE_DOUBLE:![0-9]+]], [[TYPE_DOUBLE]], i64 0}
; CHECK: [[TYPE_DOUBLE]] = !{!"double", [[TYPE_CHAR]]}
