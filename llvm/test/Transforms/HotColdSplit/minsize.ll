; RUN: opt -hotcoldsplit -S < %s | FileCheck %s

; CHECK-LABEL: @fun
; CHECK: codeRepl:
; CHECK-NEXT: call void @fun.cold.1

define void @fun() {
entry:
  br i1 undef, label %if.then, label %if.else

if.then:
  ret void

if.else:
  br label %if.then4

if.then4:
  br i1 undef, label %if.then5, label %if.end

if.then5:
  br label %cleanup

if.end:
  br label %cleanup

cleanup:
  %cleanup.dest.slot.0 = phi i32 [ 1, %if.then5 ], [ 0, %if.end ]
  unreachable
}

; CHECK: define {{.*}} @fun.cold.1{{.*}}#[[outlined_func_attr:[0-9]+]]
; CHECK: attributes #[[outlined_func_attr]] = { {{.*}}minsize
