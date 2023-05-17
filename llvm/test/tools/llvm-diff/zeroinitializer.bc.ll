; Bugzilla: https://bugs.llvm.org/show_bug.cgi?id=33623
; RUN: llvm-diff %s %s

%A = type { i64, i64 }
@_gm_ = global <2 x ptr> zeroinitializer

define void @f() {
entry:
  store <2 x ptr> zeroinitializer, ptr @_gm_
  ret void
}
