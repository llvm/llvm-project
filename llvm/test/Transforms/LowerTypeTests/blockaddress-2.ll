; RUN: opt -S %s -passes=lowertypetests | FileCheck %s

; CHECK: @badfileops = internal global %struct.f { void ()* @bad_f, void ()* @bad_f }
; CHECK: @bad_f = internal alias void (), void ()* @.cfi.jumptable
; CHECK: define internal void @bad_f.cfi() !type !0 {
; CHECK-NEXT:  ret void

target triple = "x86_64-unknown-linux"

%struct.f = type { void ()*, void ()* }
@badfileops = internal global %struct.f { void ()* @bad_f, void ()* @bad_f }, align 8

declare i1 @llvm.type.test(i8*, metadata)

define internal void @bad_f() !type !1 {
  ret void
}

define internal fastcc void @do_f() unnamed_addr !type !2 {
  %1 = tail call i1 @llvm.type.test(i8* undef, metadata !"_ZTSFiP4fileP3uioP5ucrediP6threadE"), !nosanitize !3
  ret void
}

!1 = !{i64 0, !"_ZTSFiP4fileP3uioP5ucrediP6threadE"}
!2 = !{i64 0, !"_ZTSFiP6threadiP4fileP3uioliE"}
!3 = !{}
