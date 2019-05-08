; In PR41658, argpromotion put an inalloca in a position that per the
; calling convention is passed in a register. This test verifies that
; we don't do that anymore. It also verifies that the combination of
; globalopt and argpromotion is able to optimize the call safely.
;
; RUN: opt -S -argpromotion %s | FileCheck --check-prefix=THIS %s
; RUN: opt -S -globalopt -argpromotion %s | FileCheck --check-prefix=OPT %s
; THIS: define internal x86_thiscallcc void @internalfun(%struct.a* %this, <{ %struct.a
; OPT: define internal fastcc void @internalfun(<{ %struct.a }>*)

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc19.11.0"

%struct.a = type { i8 }

define internal x86_thiscallcc void @internalfun(%struct.a* %this, <{ %struct.a }>* inalloca) {
entry:
  %a = getelementptr inbounds <{ %struct.a }>, <{ %struct.a }>* %0, i32 0, i32 0
  %argmem = alloca inalloca <{ %struct.a }>, align 4
  %1 = getelementptr inbounds <{ %struct.a }>, <{ %struct.a }>* %argmem, i32 0, i32 0
  %call = call x86_thiscallcc %struct.a* @copy_ctor(%struct.a* %1, %struct.a* dereferenceable(1) %a)
  call void @ext(<{ %struct.a }>* inalloca %argmem)
  ret void
}

; This is here to ensure @internalfun is live.
define void @exportedfun(%struct.a* %a) {
  %inalloca.save = tail call i8* @llvm.stacksave()
  %argmem = alloca inalloca <{ %struct.a }>, align 4
  call x86_thiscallcc void @internalfun(%struct.a* %a, <{ %struct.a }>* inalloca %argmem)
  call void @llvm.stackrestore(i8* %inalloca.save)
  ret void
}

declare x86_thiscallcc %struct.a* @copy_ctor(%struct.a* returned, %struct.a* dereferenceable(1))
declare void @ext(<{ %struct.a }>* inalloca)
declare i8* @llvm.stacksave()
declare void @llvm.stackrestore(i8*)
