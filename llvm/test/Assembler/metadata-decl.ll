; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: llvm-as < %s | llvm-dis -materialize-metadata | FileCheck %s

; CHECK: @foo = external global i32, !foo [[M0:![0-9]+]]
@foo = external global i32, !foo !0

; CHECK: declare !bar [[M1:![0-9]+]] void @bar()
declare !bar !1 void @bar()

; CHECK: declare void @test1(i32 noundef !foo [[M0]] !bar [[M1]], i32 !range [[M2:![0-9]+]])
declare void @test1(i32 noundef !foo !0 !bar !1, i32 !range !2)

; CHECK: [[M0]] = distinct !{}
; CHECK: [[M1]] = distinct !{}
; CHECK: [[M2]] = !{i32 1, i32 0}

!0 = distinct !{}
!1 = distinct !{}
!2 = !{ i32 1, i32 0 }
