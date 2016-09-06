; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -disable-ppc-sco=false --enable-shrink-wrap=false | FileCheck %s -check-prefix=CHECK-SCO-ONLY
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -disable-ppc-sco=false --enable-shrink-wrap=true | FileCheck %s -check-prefix=CHECK-SCO-SHRK
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu -disable-ppc-sco=false --enable-shrink-wrap=false | FileCheck %s -check-prefix=CHECK-SCO-ONLY
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu -disable-ppc-sco=false --enable-shrink-wrap=true | FileCheck %s -check-prefix=CHECK-SCO-SHRK

%"class.clang::NamedDecl" = type { i32 }
declare void @__assert_fail();

define i8 @_ZNK5clang9NamedDecl23getLinkageAndVisibilityEv(
    %"class.clang::NamedDecl"* %this) {
entry:
  %tobool = icmp eq %"class.clang::NamedDecl"* %this, null
  br i1 %tobool, label %cond.false, label %exit

cond.false:
  tail call void @__assert_fail()
  unreachable

exit:
  %DeclKind = getelementptr inbounds
                            %"class.clang::NamedDecl",
                            %"class.clang::NamedDecl"* %this, i64 0, i32 0
  %bf.load = load i32, i32* %DeclKind, align 4
  %call.i = tail call i8 @LVComputationKind(
    %"class.clang::NamedDecl"* %this,
    i32 %bf.load)
  ret i8 %call.i

; CHECK-SCO-SHRK-LABEL: _ZNK5clang9NamedDecl23getLinkageAndVisibilityEv:
; CHECK-SCO-SHRK: b LVComputationKind
; CHECK-SCO-SHRK: #TC_RETURNd8
; CHECK-SCO-SHRK: stdu 1, -{{[0-9]+}}(1)
; CHECK-SCO-SHRK: bl __assert_fail
;
; CHECK-SCO-ONLY-LABEL: _ZNK5clang9NamedDecl23getLinkageAndVisibilityEv:
; CHECK-SCO-ONLY: stdu 1, -{{[0-9]+}}(1)
; CHECK-SCO-ONLY: b LVComputationKind
; CHECK-SCO-ONLY: #TC_RETURNd8
; CHECK-SCO-ONLY: bl __assert_fail
}

define fastcc i8 @LVComputationKind(
    %"class.clang::NamedDecl"* %D,
    i32 %computation) {
  ret i8 0
}
