; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; RUN: llvm-as -disable-verify < %s | opt -verify -S | FileCheck %s  --check-prefix=STRIP

; STRIP-NOT: tbaa
; STRIP: @f_0
; STRIP: Do no strip this
define void @f_0(i32* %ptr) {
; This part checks for the easy syntactic verifier rules.

; CHECK: Struct tag metadata must have either 3 or 4 operands
; CHECK-NEXT:  store i32 0, i32* %ptr, !tbaa !{{[0-9]+}}

; CHECK: Immutability tag on struct tag metadata must be a constant
; CHECK-NEXT:  store i32 1, i32* %ptr, !tbaa !{{[0-9]+}}

; CHECK: Immutability part of the struct tag metadata must be either 0 or 1
; CHECK-NEXT:  store i32 2, i32* %ptr, !tbaa !{{[0-9]+}}

; CHECK: Offset must be constant integer
; CHECK-NEXT:  store i32 3, i32* %ptr, !tbaa !{{[0-9]+}}

; CHECK: Malformed struct tag metadata:  base and access-type should be non-null and point to Metadata nodes
; CHECK-NEXT:  store i32 4, i32* %ptr, !tbaa !{{[0-9]+}}

; CHECK: Access type node must be a valid scalar type
; CHECK-NEXT:  store i32 5, i32* %ptr, !tbaa !{{[0-9]+}}

; CHECK: Access bit-width not the same as description bit-width
; CHECK-NEXT:  store i32 6, i32* %ptr, !tbaa !{{[0-9]+}}

; CHECK: Access type node must be a valid scalar type
; CHECK-NEXT:  store i32 7, i32* %ptr, !tbaa !{{[0-9]+}}

; CHECK: Struct tag nodes have a string as their first operand
; CHECK-NEXT:  !{{[0-9]+}} = !{!{{[0-9]+}}, !{{[0-9]+}}, i64 0}

; CHECK: Access type node must be a valid scalar type
; CHECK-NEXT:  store i32 9, i32* %ptr, !tbaa !{{[0-9]+}}

  store i32 0, i32* %ptr, !tbaa !{!3, !2, i64 40, i64 0, i64 1, i64 2}
  store i32 1, i32* %ptr, !tbaa !{!3, !2, i64 40, !"immutable"}
  store i32 2, i32* %ptr, !tbaa !{!3, !2, i64 40, i64 4}
  store i32 3, i32* %ptr, !tbaa !{!3, !2, !"40", i64 0}
  store i32 4, i32* %ptr, !tbaa !{!3, null, !"40", i64 0}
  store i32 5, i32* %ptr, !tbaa !{!3, !3, !"40", i64 0}
  store i32 6, i32* %ptr, !tbaa !{!3, !2, i32 40, i64 0}
  store i32 7, i32* %ptr, !tbaa !{!3, !12, i32 40, i64 0}, !metadata !42
  store i32 8, i32* %ptr, !tbaa !{!13, !1, i64 0}
  store i32 9, i32* %ptr, !tbaa !{!14, !14, i64 0}
  ret void
}
!42 = !{!"Do no strip this!"}

define void @f_1(i32* %ptr) {
; This part checks for more semantic verifier rules.

; CHECK: Cycle detected in struct path
; CHECK-NEXT:  store i32 0, i32* %ptr, !tbaa !{{[0-9]+}}

; CHECK: Offset not zero at the point of scalar access
; CHECK-NEXT:  store i32 1, i32* %ptr, !tbaa !{{[0-9]+}}

; CHECK: Offset not zero at the point of scalar access
; CHECK-NEXT:  store i32 2, i32* %ptr, !tbaa !{{[0-9]+}}

; CHECK: Could not find TBAA parent in struct type node
; CHECK-NEXT:  store i32 3, i32* %ptr, !tbaa !{{[0-9]+}}

; CHECK: Did not see access type in access path!
; CHECK-NEXT:  store i32 3, i32* %ptr, !tbaa !{{[0-9]+}}

; CHECK: Access type node must be a valid scalar type
; CHECK-NEXT:  store i32 4, i32* %ptr, !tbaa !{{[0-9]+}}

; CHECK: Access type node must be a valid scalar type
; CHECK-NEXT:  store i32 5, i32* %ptr, !tbaa !{{[0-9]+}}

; CHECK: Access type node must be a valid scalar type
; CHECK-NEXT:  store i32 6, i32* %ptr, !tbaa !{{[0-9]+}}

; CHECK: Struct tag nodes must have an odd number of operands!
; CHECK-NEXT:!{{[0-9]+}} = !{!"bad-struct-type-0", !{{[0-9]+}}, i64 40, !{{[0-9]+}}}

; CHECK: Incorrect field entry in struct type node!
; CHECK-NEXT:  store i32 8, i32* %ptr, !tbaa !{{[0-9]+}}

; CHECK: Bitwidth between the offsets and struct type entries must match
; CHECK-NEXT:  store i32 9, i32* %ptr, !tbaa !{{[0-9]+}}

; CHECK: Offsets must be increasing!
; CHECK-NEXT:  store i32 10, i32* %ptr, !tbaa !{{[0-9]+}}

  store i32 0, i32* %ptr, !tbaa !{!4, !2, i64 40}
  store i32 1, i32* %ptr, !tbaa !{!3, !2, i64 45}
  store i32 2, i32* %ptr, !tbaa !{!3, !2, i64 45}
  store i32 3, i32* %ptr, !tbaa !{!3, !2, i64 10}
  store i32 4, i32* %ptr, !tbaa !{!5, !5, i64 0}
  store i32 5, i32* %ptr, !tbaa !{!6, !6, i64 0}
  store i32 6, i32* %ptr, !tbaa !{!7, !7, i64 0}
  store i32 7, i32* %ptr, !tbaa !{!8, !1, i64 40}
  store i32 8, i32* %ptr, !tbaa !{!9, !1, i64 40}
  store i32 9, i32* %ptr, !tbaa !{!10, !1, i64 40}
  store i32 10, i32* %ptr, !tbaa !{!11, !1, i64 40}
  ret void
}



!0 = !{!"root"}
!1 = !{!"scalar-a", !0}
!2 = !{!"scalar-b", !0}
!3 = !{!"struct-a", !2, i64 20, !1, i64 40}
!4 = distinct !{!"self-recursive-struct", !2, i64 20, !4, i64 40}
!5 = !{!"bad-scalar-0", i64 40}
!6 = !{i64 42, !0}
!7 = !{!"bad-scalar-1", null}
!8 = !{!"bad-struct-type-0", !1, i64 40, !1}
!9 = !{!"bad-struct-type-1", !1, i64 40, i64 56, !1}
!10 = !{!"bad-struct-type-2", !1, i64 40, !1, i32 56}
!11 = !{!"bad-struct-type-2", !1, i64 80, !1, i64 56}
!12 = !{!"bad-scalar-2", !3, i64 0}
!13 = !{!1, !1, i64 0}
!14 = !{!"bad-scalar-2", !13}
