; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

; CHECK: assembly parsed, but does not verify

!named = !{!4, !6, !7, !8, !9}

!0 = distinct !DIFragment()
!1 = distinct !DIFragment()
!2 = distinct !DIFragment()
!3 = distinct !DIFragment()

; CHECK: object must be a DIObject
!4 = distinct !DILifetime(object: !5, location: !5)
!5 = !{}

; CHECK: location expression must be a DIExpr
!6 = distinct !DILifetime(object: !0, location: !5)

; CHECK: each argObject must be a DIObject
!7 = distinct !DILifetime(object: !0, location: !DIExpr(DIOpArg(0, i32)), argObjects: {!5})

; CHECK: each argObject must be a DIObject
!8 = distinct !DILifetime(object: !0, location: !DIExpr(DIOpArg(0, i32), DIOpArg(1, i32), DIOpArg(2, i32)), argObjects: {!1, !5, !2})

; CHECK: debug location expression cannot reference an out-of-bounds argObjects index
!9 = distinct !DILifetime(object: !0, location: !DIExpr(DIOpArg(0, i32), DIOpArg(1, i32), DIOpArg(2, i32)), argObjects: {!1, !2})
