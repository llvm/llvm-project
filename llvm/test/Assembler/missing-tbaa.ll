; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; Check that !tbaa upgrade doesn't crash on undefined metadata (it should give
; an error).

define void @foo() {
entry:
  store i8 undef, i8* undef,
; CHECK: :[[@LINE+1]]:10: error: use of undefined metadata '!1'
  !tbaa !1
  unreachable
}
