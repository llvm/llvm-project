; RUN: opt -hotcoldsplit -S < %s | FileCheck %s

; CHECK-LABEL: @fun
; CHECK: call void @fun.cold.1

define void @fun() {
entry:
  br i1 undef, label %if.then, label %if.else

if.then:
  ret void

if.else:
  call void @sink()
  call void @sink()
  call void @sink()
  ret void
}

declare void @sink() cold

; CHECK: define {{.*}} @fun.cold.1{{.*}}#[[outlined_func_attr:[0-9]+]]
; CHECK: attributes #[[outlined_func_attr]] = { {{.*}}minsize
