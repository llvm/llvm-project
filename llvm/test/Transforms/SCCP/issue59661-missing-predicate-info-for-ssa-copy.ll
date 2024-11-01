; REQUIRES: asserts
; XFAIL: *
; RUN: opt -S -passes=ipsccp < %s

; https://github.com/llvm/llvm-project/issues/59661

define i32 @bar() {
entry:
  %call = call i32 @foo()
  ret i32 0
}

define internal i32 @foo() {
entry:
  %arst = call ptr @llvm.ssa.copy.p0(ptr @foo)
  ret i32 0
}

declare ptr @llvm.ssa.copy.p0(ptr) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
