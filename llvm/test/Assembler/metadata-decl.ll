; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: llvm-as < %s | llvm-dis -materialize-metadata | FileCheck %s

; CHECK: @foo = external global i32, !foo !0
@foo = external global i32, !foo !0

; CHECK: declare !bar !1 void @bar()
declare !bar !1 void @bar()

!0 = distinct !{}
!1 = distinct !{}
