; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{
; CHECK-SAME: !DIExpr(),
; CHECK-SAME: !DIExpr(DIOpReferrer(i32)),
; CHECK-SAME: !DIExpr(DIOpArg(0, i32), DIOpConvert(float)),
; CHECK-SAME: !DIExpr(DIOpTypeObject(double)),
; CHECK-SAME: !DIExpr(DIOpConstant(i8 1)),
; CHECK-SAME: !DIExpr(DIOpConvert(i16)),
; CHECK-SAME: !DIExpr(DIOpReinterpret(i64)),
; CHECK-SAME: !DIExpr(DIOpBitOffset(i1)),
; CHECK-SAME: !DIExpr(DIOpByteOffset(i16)),
; CHECK-SAME: !DIExpr(DIOpComposite(4, i8)),
; CHECK-SAME: !DIExpr(DIOpExtend(6)),
; CHECK-SAME: !DIExpr(DIOpSelect()),
; CHECK-SAME: !DIExpr(DIOpAddrOf(1)),
; CHECK-SAME: !DIExpr(DIOpDeref(i32)),
; CHECK-SAME: !DIExpr(DIOpRead()),
; CHECK-SAME: !DIExpr(DIOpAdd()),
; CHECK-SAME: !DIExpr(DIOpSub()),
; CHECK-SAME: !DIExpr(DIOpMul()),
; CHECK-SAME: !DIExpr(DIOpDiv()),
; CHECK-SAME: !DIExpr(DIOpShr()),
; CHECK-SAME: !DIExpr(DIOpShl()),
; CHECK-SAME: !DIExpr(DIOpPushLane(i32)),
; CHECK-SAME: !DIExpr()}

!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22}

!0 = !DIExpr()
!1 = !DIExpr(DIOpReferrer(i32))
!2 = !DIExpr(DIOpArg(0, i32), DIOpConvert(float))
!3 = !DIExpr(DIOpTypeObject(double))
!4 = !DIExpr(DIOpConstant(i8 1))
!5 = !DIExpr(DIOpConvert(i16))
!6 = !DIExpr(DIOpReinterpret(i64))
!7 = !DIExpr(DIOpBitOffset(i1))
!8 = !DIExpr(DIOpByteOffset(i16))
!9 = !DIExpr(DIOpComposite(4, i8))
!10 = !DIExpr(DIOpExtend(6))
!11 = !DIExpr(DIOpSelect())
!12 = !DIExpr(DIOpAddrOf(1))
!13 = !DIExpr(DIOpDeref(i32))
!14 = !DIExpr(DIOpRead())
!15 = !DIExpr(DIOpAdd())
!16 = !DIExpr(DIOpSub())
!17 = !DIExpr(DIOpMul())
!18 = !DIExpr(DIOpDiv())
!19 = !DIExpr(DIOpShr())
!20 = !DIExpr(DIOpShl())
!21 = !DIExpr(DIOpPushLane(i32))
!22 = !DIExpr()
