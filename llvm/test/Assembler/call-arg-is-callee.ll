; RUN: llvm-as < %s -disable-output 2>&1 | FileCheck %s -allow-empty
; CHECK-NOT: error
; CHECK-NOT: warning
; RUN: verify-uselistorder < %s

; Check ordering of callee operand versus the argument operand.
define void @call(void (...)* %p) {
  call void (...) %p(void (...)* %p)
  ret void
}

; Check ordering of callee operand versus the argument operand.
declare void @personality(i8*)
define void @invoke(void (...)* %p) personality void(i8*)* @personality {
entry:
  invoke void (...) %p(void (...)* %p)
  to label %normal unwind label %exception
normal:
  ret void
exception:
  landingpad { i8*, i32 } cleanup
  ret void
}

; Check order for callbr instruction. Cannot reuse labels in the test since the
; verifier prevents duplicating callbr destinations.
define void @callbr() {
entry:
  callbr i32 asm "", "=r,r,i,i"(i32 0,
                                i8 *blockaddress(@callbr, %two),
                                i8 *blockaddress(@callbr, %three))
              to label %one [label %two, label %three]
one:
  ret void
two:
  ret void
three:
  ret void
}
