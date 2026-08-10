; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12}

!0 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!1 = !{null}
!2 = !{null, !0}
!3 = !{!0, !0, !0}


; CHECK: !4 = !DISubroutineType(types: !1)
; CHECK: !5 = !DISubroutineType(types: !2)
; CHECK: !6 = !DISubroutineType(types: !3)
; CHECK: !7 = !DISubroutineType(flags: DIFlagLValueReference, types: !3)
!4 = !DISubroutineType(types: !1)
!5 = !DISubroutineType(types: !2)
!6 = !DISubroutineType(types: !3)
!7 = !DISubroutineType(flags: DIFlagLValueReference, types: !3)

; CHECK: !8 = !DISubroutineType(types: null)
!8 = !DISubroutineType(types: null)

; CHECK: !9 = !DISubroutineType(cc: DW_CC_normal, types: null)
; CHECK: !10 = !DISubroutineType(cc: DW_CC_BORLAND_msfastcall, types: null)
; CHECK: !11 = !DISubroutineType(cc: DW_CC_BORLAND_stdcall, types: null)
; CHECK: !12 = !DISubroutineType(cc: DW_CC_LLVM_vectorcall, types: null)
!9 = !DISubroutineType(cc: DW_CC_normal, types: null)
!10 = !DISubroutineType(cc: DW_CC_BORLAND_msfastcall, types: null)
!11 = !DISubroutineType(cc: DW_CC_BORLAND_stdcall, types: null)
!12 = !DISubroutineType(cc: DW_CC_LLVM_vectorcall, types: null)
