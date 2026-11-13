; RUN: mlir-translate -import-llvm -emit-expensive-warnings %s 2>&1 | FileCheck %s

; !tbaa.struct describes which fields of an aggregate a copy touches, so the
; copy is not treated as one opaque access and the gaps are known to be padding.
; The node is a flat list of (offset, size, tag) triples.

; CHECK-NOT: warning

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memmove.p0.p0.i64(ptr, ptr, i64, i1)
declare void @llvm.memcpy.inline.p0.p0.i64(ptr, ptr, i64, i1)

; The two fields start eight bytes apart with a four byte hole between them;
; that hole is padding the copy need not preserve.
; CHECK-DAG: #[[INT:.*]] = #llvm.tbaa_type_desc<id = "int"
; CHECK-DAG: #[[FLT:.*]] = #llvm.tbaa_type_desc<id = "float"
; CHECK-DAG: #[[TI:.*]] = #llvm.tbaa_tag<base_type = #[[INT]], access_type = #[[INT]], offset = 0>
; CHECK-DAG: #[[TF:.*]] = #llvm.tbaa_tag<base_type = #[[FLT]], access_type = #[[FLT]], offset = 0>
; CHECK-DAG: #[[F0:.*]] = #llvm.tbaa_struct_field<offset = 0, size = 4, tag = #[[TI]]>
; CHECK-DAG: #[[F1:.*]] = #llvm.tbaa_struct_field<offset = 8, size = 4, tag = #[[TF]]>
; CHECK-DAG: #[[F2:.*]] = #llvm.tbaa_struct_field<offset = 2147483648, size = 4, tag = #[[TI]]>
; CHECK-DAG: #[[F3:.*]] = #llvm.tbaa_struct_field<offset = 18446744073709551615, size = 4, tag = #[[TI]]>
; CHECK-DAG: #[[F4:.*]] = #llvm.tbaa_struct_field<offset = 0, size = 2147483648, tag = #[[TI]]>
; CHECK: llvm.intr.memcpy
; CHECK-SAME: tbaa_struct = [#[[F0]], #[[F1]]]
; CHECK: llvm.intr.memmove
; CHECK-SAME: tbaa_struct = [#[[F0]], #[[F1]]]
; CHECK: llvm.intr.memcpy.inline
; CHECK-SAME: tbaa_struct = [#[[F0]], #[[F1]]]
define void @cp(ptr %d, ptr %s) {
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 12, i1 false), !tbaa.struct !4
  call void @llvm.memmove.p0.p0.i64(ptr %d, ptr %s, i64 12, i1 false), !tbaa.struct !4
  call void @llvm.memcpy.inline.p0.p0.i64(ptr %d, ptr %s, i64 12, i1 false), !tbaa.struct !4
  ret void
}

!0 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!1 = !{!"int", !0, i64 0}
!2 = !{!"float", !0, i64 0}
!6 = !{!1, !1, i64 0}
!7 = !{!2, !2, i64 0}
!4 = !{ i64 0, i64 4, !6, i64 8, i64 4, !7 }

; A narrow offset constant with its high bit set keeps its unsigned value:
; LLVM compares offsets zero-extended, so i32 0x80000000 is 2147483648, not
; a negative offset.
; CHECK: llvm.intr.memcpy
; CHECK-SAME: tbaa_struct = [#[[F2]]]
define void @narrow_offset(ptr %d, ptr %s) {
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 12, i1 false), !tbaa.struct !8
  ret void
}

!8 = !{ i32 2147483648, i64 4, !6 }

; A high-bit i64 offset keeps its full unsigned value; it is not dropped
; as unrepresentable.
; CHECK: llvm.intr.memcpy
; CHECK-SAME: tbaa_struct = [#[[F3]]]
define void @high_bit_offset(ptr %d, ptr %s) {
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 12, i1 false), !tbaa.struct !9
  ret void
}

!9 = !{ i64 -1, i64 4, !6 }

; A narrow size constant with its high bit set keeps its unsigned value too.
; CHECK: llvm.intr.memcpy
; CHECK-SAME: tbaa_struct = [#[[F4]]]
define void @narrow_size(ptr %d, ptr %s) {
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 12, i1 false), !tbaa.struct !10
  ret void
}

!10 = !{ i64 0, i32 2147483648, !6 }
