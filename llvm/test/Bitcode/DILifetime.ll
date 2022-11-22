; RUN: llvm-dis -o - %s.bc | FileCheck %s

; CHECK: !named = !{!0, !2, !3, !8, !11}
!named = !{!0, !2, !3, !8, !11}

!llvm.dbg.cu = !{!6}
!llvm.module.flags = !{!12}

; CHECK: !0 = distinct !DILifetime(object: !1, location: !DIExpr())
!0 = distinct !DILifetime(object: !1, location: !DIExpr())
!1 = distinct !DIFragment()
; CHECK: !2 = distinct !DILifetime(object: !1, location: !DIExpr())
!2 = distinct !DILifetime(object: !1, location: !DIExpr(), argObjects: {})

; CHECK: !3 = distinct !DILifetime(object: !4, location: !DIExpr())
!3 = distinct !DILifetime(object: !4, location: !DIExpr())
!4 = !DILocalVariable(scope: !5)
!5 = distinct !DISubprogram(unit: !6)
!6 = distinct !DICompileUnit(language: DW_LANG_C99, file: !7)
!7 = !DIFile(filename: "<stdin>", directory: "/")

; CHECK: !8 = distinct !DILifetime(object: !9, location: !DIExpr())
!8 = distinct !DILifetime(object: !9, location: !DIExpr())
!9 = !DIGlobalVariable(name: "G", type: !10)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

; CHECK: !11 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpArg(0, i32)), argObjects: {!9})
!11 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpArg(0, i32)), argObjects: {!9})

!12 = !{i32 2, !"Debug Info Version", i32 4}
