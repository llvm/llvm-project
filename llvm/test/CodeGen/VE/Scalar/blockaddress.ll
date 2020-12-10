; RUN: llc < %s -mtriple=ve | FileCheck %s

@addr = global i8* null, align 8

; Function Attrs: nofree norecurse nounwind writeonly
define void @test() {
; CHECK-LABEL: test:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:  .Ltmp0: # Block address taken
; CHECK-NEXT:  # %bb.1: # %test1
; CHECK-NEXT:    lea %s0, addr@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, addr@hi(, %s0)
; CHECK-NEXT:    lea %s1, .Ltmp0@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, .Ltmp0@hi(, %s1)
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
entry:
  br label %test1

test1:
  store i8* blockaddress(@test, %test1), i8** @addr, align 8
  ret void
}
