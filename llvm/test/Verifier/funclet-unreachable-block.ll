; RUN: llvm-as -disable-output %s

; colorEHFunclets() leaves unreachable blocks colorless, so there is no funclet
; color to look up for a call in one. The verifier must not crash on that.

declare i32 @__CxxFrameHandler3(...)
declare ptr @llvm.objc.retain(ptr returned) nounwind

define void @f(ptr %p) personality ptr @__CxxFrameHandler3 {
entry:
  ret void

unreachable.funclet:                              ; No predecessors!
  %pad = cleanuppad within none []
  %call = call ptr @llvm.objc.retain(ptr %p) [ "funclet"(token %pad) ]
  cleanupret from %pad unwind to caller
}
