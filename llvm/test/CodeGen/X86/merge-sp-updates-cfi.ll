; RUN: llc < %s -mtriple=i686-pc-linux | FileCheck %s


; Function Attrs: optsize
declare void @foo(i32, i32)
declare x86_stdcallcc void @stdfoo(i32, i32)

; CHECK-LABEL: testNoUnwind:
; CHECK:       subl $20, %esp
; CHECK-NOT:   subl $12, %esp
; CHECK-NOT:   subl $8, %esp
; CHECK:       calll foo
; CHECK:       addl $8, %esp
; CHECK-NOT:   addl $16, %esp
; CHECK-NOT:   subl $8, %esp
; CHECK:       calll stdfoo
; CHECK:       addl $20, %esp
; CHECK-NOT:   addl $8, %esp
; CHECK-NOT:   addl $12, %esp
define void @testNoUnwind() nounwind {
entry:
  tail call void @foo(i32 1, i32 2)
  tail call x86_stdcallcc void @stdfoo(i32 3, i32 4)
  ret void
}

; CHECK-LABEL: testWithUnwind:
; CHECK:       subl $20, %esp
; CHECK-NEXT: .cfi_adjust_cfa_offset 20
; CHECK-NOT:   subl $12, %esp
; CHECK-NOT:   subl $8, %esp
; CHECK:       calll foo
; CHECK:       addl $8, %esp
; CHECK-NEXT: .cfi_adjust_cfa_offset -8
; CHECK-NOT:   addl $16, %esp
; CHECK-NOT:   subl $8, %esp
; CHECK:       calll stdfoo
; CHECK:       addl $20, %esp
; CHECK-NEXT: .cfi_adjust_cfa_offset -20
; CHECK-NOT:   addl $8, %esp
; CHECK-NOT:   addl $12, %esp
define void @testWithUnwind() {
entry:
  tail call void @foo(i32 1, i32 2)
  tail call x86_stdcallcc void @stdfoo(i32 3, i32 4)
  ret void
}
