; RUN: llc -mtriple=bpfel -mcpu=v4 -filetype=asm -o - %s | FileCheck %s

; A program that defines _Unwind_RaiseException() gets what it defines: the call
; is an ordinary one.

define i32 @_Unwind_RaiseException(ptr %exn) nounwind uwtable {
entry:
  ret i32 5
}

define i32 @caller() nounwind uwtable {
entry:
  %r = call i32 @_Unwind_RaiseException(ptr null)
  ret i32 %r
}

; CHECK-LABEL: caller:
; CHECK:         call _Unwind_RaiseException
; CHECK-NOT:     unwind_exit
; CHECK:         exit
