; RUN: llc -mtriple=hexagon -hexagon-initial-cfg-cleanup=0 -O2 < %s \
; RUN:   | FileCheck %s

; Test coverage for HexagonCFGOptimizer: exercise the J2_jumptnewpt/J2_jumpfnewpt
; branch inversion path. The CFG optimizer should invert a conditional branch
; to eliminate an unconditional jump block.

; CHECK-LABEL: test_cfgopt_newpt:
; CHECK: if ({{!?}}p0
define i32 @test_cfgopt_newpt(i32 %a, i32 %b, i32 %c) #0 {
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %add = add nsw i32 %a, %c
  br label %return

if.else:
  %sub = sub nsw i32 %b, %c
  %mul = mul nsw i32 %sub, 3
  br label %return

return:
  %retval = phi i32 [ %add, %if.then ], [ %mul, %if.else ]
  ret i32 %retval
}

; Exercise the case2 path of CFG optimizer where JumpAroundTarget has a single
; predecessor and a single successor with an unconditional jump.
; CHECK-LABEL: test_cfgopt_case2:
; CHECK: if ({{!?}}p0
define i32 @test_cfgopt_case2(i32 %a, i32 %b, ptr %p) #0 {
entry:
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  store i32 %b, ptr %p, align 4
  br label %merge

if.else:
  %add = add i32 %a, %b
  store i32 %add, ptr %p, align 4
  br label %merge

merge:
  %val = load i32, ptr %p, align 4
  ret i32 %val
}
