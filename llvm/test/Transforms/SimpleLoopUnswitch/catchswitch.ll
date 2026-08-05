; RUN: opt -passes='simple-loop-unswitch<nontrivial>' < %s -S | FileCheck %s

; CHECK: if.end{{.*}}:
; CHECK-NOT: if.end{{.*}}:
declare i32 @__gxx_wasm_personality_v0(...)

declare void @foo()

define void @test(i1 %arg) personality ptr @__gxx_wasm_personality_v0 {
entry:
  br label %while.body

while.body:                                       ; preds = %cleanup, %entry
  br i1 %arg, label %if.end, label %if.then

if.then:                                          ; preds = %while.body
  br label %if.end

if.end:                                           ; preds = %if.then, %while.body
  invoke void @foo()
          to label %cleanup unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %invoke.cont, %if.end
  %0 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr null]
  unreachable

cleanup:                                          ; preds = %invoke.cont
  br label %while.body
}

