; Bugzilla: https://bugs.llvm.org/show_bug.cgi?id=33623
; RUN: llvm-diff %s %s

%A = type { i64, i64 }
@_gm_ = global <2 x %A*> zeroinitializer

define void @f() {
entry:
  store <2 x %A*> zeroinitializer, <2 x %A*>* @_gm_
  ret void
}
