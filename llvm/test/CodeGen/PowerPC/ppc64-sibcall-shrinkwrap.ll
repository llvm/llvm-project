; RUN: llc -relocation-model=static -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -disable-ppc-sco=false --enable-shrink-wrap=false | FileCheck %s -check-prefix=CHECK-SCO-ONLY
; RUN: llc -relocation-model=static -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -disable-ppc-sco=false --enable-shrink-wrap=true | FileCheck %s -check-prefix=CHECK-SCO-SR
; RUN: llc -relocation-model=static -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu -disable-ppc-sco=false --enable-shrink-wrap=false | FileCheck %s -check-prefix=CHECK-SCO-ONLY
; RUN: llc -relocation-model=static -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu -disable-ppc-sco=false --enable-shrink-wrap=true | FileCheck %s -check-prefix=CHECK-SCO-SR


; RUN: llc -relocation-model=pic -verify-machineinstrs < %s -mtriple=powerpc64-ibm-aix-xcoff -disable-ppc-sco=false --enable-shrink-wrap=false | FileCheck %s -check-prefixes=CHECK-SCO-ONLY-AIX,CHECK-SCO-ONLY-AIX64
; RUN: llc -relocation-model=pic -verify-machineinstrs < %s -mtriple=powerpc-ibm-aix-xcoff -disable-ppc-sco=false --enable-shrink-wrap=false | FileCheck %s -check-prefixes=CHECK-SCO-ONLY-AIX,CHECK-SCO-ONLY-AIX32
; RUN: llc -relocation-model=pic -verify-machineinstrs < %s -mtriple=powerpc64-ibm-aix-xcoff -disable-ppc-sco=false --enable-shrink-wrap=true | FileCheck %s -check-prefixes=CHECK-SCO-SR-AIX64,CHECK-SCO-SR-AIX
; RUN: llc -relocation-model=pic -verify-machineinstrs < %s -mtriple=powerpc-ibm-aix-xcoff -disable-ppc-sco=false --enable-shrink-wrap=true | FileCheck %s -check-prefixes=CHECK-SCO-SR-AIX32,CHECK-SCO-SR-AIX

%"class.clang::NamedDecl" = type { i32 }
declare void @__assert_fail();

define dso_local i8 @_ZNK5clang9NamedDecl23getLinkageAndVisibilityEv(
    ptr %this) {
entry:
  %tobool = icmp eq ptr %this, null
  br i1 %tobool, label %cond.false, label %exit

cond.false:
  tail call void @__assert_fail()
  unreachable

exit:
  %bf.load = load i32, ptr %this, align 4
  %call.i = tail call i8 @LVComputationKind(
    ptr %this,
    i32 %bf.load)
  ret i8 %call.i

; CHECK-SCO-ONLY-LABEL: _ZNK5clang9NamedDecl23getLinkageAndVisibilityEv:
; CHECK-SCO-ONLY: stdu 1, -{{[0-9]+}}(1)
; CHECK-SCO-ONLY: b LVComputationKind
; CHECK-SCO-ONLY: #TC_RETURNd8
; CHECK-SCO-ONLY: bl __assert_fail

; CHECK-SCO-ONLY-AIXLABEL: _ZNK5clang9NamedDecl23getLinkageAndVisibilityEv:
; CHECK-SCO-ONLY-AIX64: stdu 1, -{{[0-9]+}}(1)
; CHECK-SCO-ONLY-AIX32: stwu 1, -{{[0-9]+}}(1)
; CHECK-SCO-ONLY-AIX: b .LVComputationKind
; CHECK-SCO-ONLY-AiX64: #TC_RETURNd8
; CHECK-SCO-ONLY-AiX32: #TC_RETURNd
; CHECK-SCO-ONLY-AIX: bl .__assert_fail

; CHECK-SCO-SR-LABEL: _ZNK5clang9NamedDecl23getLinkageAndVisibilityEv:
; CHECK-SCO-SR: b LVComputationKind
; CHECK-SCO-SR: #TC_RETURNd8
; CHECK-SCO-SR: stdu 1, -{{[0-9]+}}(1)
; CHECK-SCO-SR: bl __assert_fail

; CHECK-SCO-SR-AIX-LABEL: _ZNK5clang9NamedDecl23getLinkageAndVisibilityEv:
; CHECK-SCO-SR-AIX: b .LVComputationKind
; CHECK-SCO-SR-AIX64: #TC_RETURNd8
; CHECK-SCO-SR-AIX64: stdu 1, -{{[0-9]+}}(1)
; CHECK-SCO-SR-AIX32: #TC_RETURNd
; CHECK-SCO-SR-AIX32: stwu 1, -{{[0-9]+}}(1)
; CHECK-SCO-SR-AIX: bl .__assert_fail

}

define dso_local fastcc i8 @LVComputationKind(
    ptr %D,
    i32 %computation) {
  ret i8 0
}
