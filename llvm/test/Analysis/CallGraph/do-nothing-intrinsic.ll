; RUN: opt < %s -basiccg
; PR13903

define void @main() {
  invoke void @llvm.donothing()
          to label %ret unwind label %unw
unw:
  %tmp = landingpad i8 personality i8 0 cleanup
  br label %ret
ret:
  ret void
}
declare void @llvm.donothing() nounwind readnone
