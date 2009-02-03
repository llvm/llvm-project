; RUN: llvm-as < %s | llvm-dis | not grep {void@}
; PR2894
declare void @g()
define void @f() {
  invoke void @g() to label %c unwind label %c
  c: ret void
}
