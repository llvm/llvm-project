; RUN: opt -analyze -branch-prob < %s | FileCheck %s
; RUN: opt < %s -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s

declare i32 @llvm.experimental.deoptimize.i32(...)

define i32 @test1(i32 %a, i32 %b) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test1':
entry:
  %cond = icmp eq i32 %a, 42
  br i1 %cond, label %exit, label %deopt

; CHECK:  edge entry -> exit probability is 0x7fffffff / 0x80000000 = 100.00% [HOT edge]
; CHECK:  edge entry -> deopt probability is 0x00000001 / 0x80000000 = 0.00%

deopt:
  %rval = call i32(...) @llvm.experimental.deoptimize.i32() [ "deopt"() ]
  ret i32 %rval

exit:
  ret i32 %b
}
