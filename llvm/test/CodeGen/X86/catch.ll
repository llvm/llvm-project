; RUN: llc < %s -mtriple=x86_64-pc-linux -relocation-model=pic | FileCheck %s

; PR18390
; We used to assert creating this label. The name itself is not critical. It
; just needs to be a unique local symbol.
; PR36885
; The stub symbol should have pointer-size (8 byte) alignment.
; CHECK:      .data
; CHECK-NEXT: .p2align 3
; CHECK-NEXT: .L.Lstr.DW.stub:
; CHECK-NEXT: .quad   .Lstr

@str = private unnamed_addr constant [12 x i8] c"NSException\00"
define void @f() personality ptr @h {
  invoke void @g()
          to label %invoke.cont unwind label %lpad
invoke.cont:
  ret void
lpad:
  %tmp14 = landingpad { ptr, i32 }
           catch ptr @str
  ret void
}
declare void @g()
declare void @h()
