; RUN: opt < %s -basiccg -enable-new-pm=0
; PR13903

define void @main() personality i8 0 {
  invoke void @llvm.donothing()
          to label %ret unwind label %unw
unw:
  %tmp = landingpad i8 cleanup
  br label %ret
ret:
  ret void
}
declare void @llvm.donothing() nounwind readnone
