; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s

declare dso_local swifttailcc i32 @tailcallee(i32 %a1, i32 %a2, i32 %a3, i32 %a4)

define dso_local swifttailcc i32 @tailcaller(i32 %in1, i32 %in2) nounwind {
; CHECK-LABEL: tailcaller:
; CHECK-NOT: subq
; CHECK-NOT: addq
; CHECK: jmp tailcallee
entry:
  %tmp11 = musttail call swifttailcc i32 @tailcallee(i32 %in1, i32 %in2, i32 %in1, i32 %in2)
  ret i32 %tmp11
}

declare dso_local swifttailcc i8* @alias_callee()

define swifttailcc noalias i8* @noalias_caller() nounwind {
; CHECK-LABEL: noalias_caller:
; CHECK:    jmp alias_callee
  %p = musttail call swifttailcc i8* @alias_callee()
  ret i8* %p
}

declare dso_local swifttailcc noalias i8* @noalias_callee()

define dso_local swifttailcc i8* @alias_caller() nounwind {
; CHECK-LABEL: alias_caller:
; CHECK:    jmp noalias_callee # TAILCALL
  %p = musttail call swifttailcc noalias i8* @noalias_callee()
  ret i8* %p
}

declare dso_local swifttailcc i32 @i32_callee()

define dso_local swifttailcc i32 @ret_undef() nounwind {
; CHECK-LABEL: ret_undef:
; CHECK:    jmp i32_callee # TAILCALL
  %p = musttail call swifttailcc i32 @i32_callee()
  ret i32 undef
}

declare dso_local swifttailcc void @does_not_return()

define dso_local swifttailcc i32 @noret() nounwind {
; CHECK-LABEL: noret:
; CHECK:    jmp does_not_return
  tail call swifttailcc void @does_not_return()
  unreachable
}

define dso_local swifttailcc void @void_test(i32, i32, i32, i32) {
; CHECK-LABEL: void_test:
; CHECK:    jmp void_test
  entry:
   musttail call swifttailcc void @void_test( i32 %0, i32 %1, i32 %2, i32 %3)
   ret void
}

define dso_local swifttailcc i1 @i1test(i32, i32, i32, i32) {
; CHECK-LABEL: i1test:
; CHECK:    jmp i1test
  entry:
  %4 = musttail call swifttailcc i1 @i1test( i32 %0, i32 %1, i32 %2, i32 %3)
  ret i1 %4
}
