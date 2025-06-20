; RUN: opt -passes=hotcoldsplit -hotcoldsplit-threshold=-1 -pass-remarks=hotcoldsplit -S < %s 2>&1 | FileCheck %s

; Make sure this compiles. This test used to fail with an invalid phi node: the
; two predecessors were outlined and the SSA representation was invalid.

; CHECK: remark: <unknown>:0:0: fun split cold code into fun.cold.1
; CHECK-LABEL: @fun
; CHECK: codeRepl:
; CHECK-NEXT: call void @fun.cold.1

; CHECK: define internal {{.*}}@fun.cold.1{{.*}} [[cold_attr:#[0-9]+]]
; CHECK: attributes [[cold_attr]] = { {{.*}}noreturn

define void @fun(i1 %arg) {
entry:
  br i1 %arg, label %if.then, label %if.else

if.then:
  ret void

if.else:
  br label %if.then4

if.then4:
  br i1 %arg, label %if.then5, label %if.end

if.then5:
  br label %cleanup

if.end:
  br label %cleanup

cleanup:
  %cleanup.dest.slot.0 = phi i32 [ 1, %if.then5 ], [ 0, %if.end ]
  unreachable
}
