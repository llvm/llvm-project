; RUN: opt < %s -analyze -branch-prob | FileCheck %s
; RUN: opt < %s -analyze -lazy-branch-prob | FileCheck %s
; RUN: opt < %s -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s

define i32 @test1(i32 %i, i32* %a) {
; CHECK: Printing analysis {{.*}} for function 'test1'
entry:
  br label %body
; CHECK: edge entry -> body probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

body:
  %iv = phi i32 [ 0, %entry ], [ %next, %body ]
  %base = phi i32 [ 0, %entry ], [ %sum, %body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i32 %iv
  %0 = load i32, i32* %arrayidx
  %sum = add nsw i32 %0, %base
  %next = add i32 %iv, 1
  %exitcond = icmp eq i32 %next, %i
  br i1 %exitcond, label %exit, label %body
; CHECK: edge body -> exit probability is 0x04000000 / 0x80000000 = 3.12%
; CHECK: edge body -> body probability is 0x7c000000 / 0x80000000 = 96.88% [HOT edge]

exit:
  ret i32 %sum
}

define i32 @test2(i32 %i, i32 %a, i32 %b) {
; CHECK: Printing analysis {{.*}} for function 'test2'
entry:
  %cond = icmp ult i32 %i, 42
  br i1 %cond, label %then, label %else, !prof !0
; CHECK: edge entry -> then probability is 0x78787878 / 0x80000000 = 94.12% [HOT edge]
; CHECK: edge entry -> else probability is 0x07878788 / 0x80000000 = 5.88%

then:
  br label %exit
; CHECK: edge then -> exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

else:
  br label %exit
; CHECK: edge else -> exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

exit:
  %result = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %result
}

!0 = !{!"branch_weights", i32 64, i32 4}

define i32 @test3(i32 %i, i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
; CHECK: Printing analysis {{.*}} for function 'test3'
entry:
  switch i32 %i, label %case_a [ i32 1, label %case_b
                                 i32 2, label %case_c
                                 i32 3, label %case_d
                                 i32 4, label %case_e ], !prof !1
; CHECK: edge entry -> case_a probability is 0x06666666 / 0x80000000 = 5.00%
; CHECK: edge entry -> case_b probability is 0x06666666 / 0x80000000 = 5.00%
; CHECK: edge entry -> case_c probability is 0x66666666 / 0x80000000 = 80.00%
; CHECK: edge entry -> case_d probability is 0x06666666 / 0x80000000 = 5.00%
; CHECK: edge entry -> case_e probability is 0x06666666 / 0x80000000 = 5.00%

case_a:
  br label %exit
; CHECK: edge case_a -> exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

case_b:
  br label %exit
; CHECK: edge case_b -> exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

case_c:
  br label %exit
; CHECK: edge case_c -> exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

case_d:
  br label %exit
; CHECK: edge case_d -> exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

case_e:
  br label %exit
; CHECK: edge case_e -> exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

exit:
  %result = phi i32 [ %a, %case_a ],
                    [ %b, %case_b ],
                    [ %c, %case_c ],
                    [ %d, %case_d ],
                    [ %e, %case_e ]
  ret i32 %result
}

!1 = !{!"branch_weights", i32 4, i32 4, i32 64, i32 4, i32 4}

define i32 @test4(i32 %x) nounwind uwtable readnone ssp {
; CHECK: Printing analysis {{.*}} for function 'test4'
entry:
  %conv = sext i32 %x to i64
  switch i64 %conv, label %return [
    i64 0, label %sw.bb
    i64 1, label %sw.bb
    i64 2, label %sw.bb
    i64 5, label %sw.bb1
  ], !prof !2
; CHECK: edge entry -> return probability is 0x0a8a8a8b / 0x80000000 = 8.24%
; CHECK: edge entry -> sw.bb probability is 0x15151515 / 0x80000000 = 16.47%
; CHECK: edge entry -> sw.bb1 probability is 0x60606060 / 0x80000000 = 75.29%

sw.bb:
  br label %return

sw.bb1:
  br label %return

return:
  %retval.0 = phi i32 [ 5, %sw.bb1 ], [ 1, %sw.bb ], [ 0, %entry ]
  ret i32 %retval.0
}

!2 = !{!"branch_weights", i32 7, i32 6, i32 4, i32 4, i32 64}

declare void @coldfunc() cold

define i32 @test5(i32 %a, i32 %b, i1 %flag) {
; CHECK: Printing analysis {{.*}} for function 'test5'
entry:
  br i1 %flag, label %then, label %else
; CHECK: edge entry -> then probability is 0x07878788 / 0x80000000 = 5.88%
; CHECK: edge entry -> else probability is 0x78787878 / 0x80000000 = 94.12% [HOT edge]

then:
  call void @coldfunc()
  br label %exit
; CHECK: edge then -> exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

else:
  br label %exit
; CHECK: edge else -> exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

exit:
  %result = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %result
}

declare i32 @regular_function(i32 %i)

define i32 @test_cold_call_sites_with_prof(i32 %a, i32 %b, i1 %flag, i1 %flag2) {
; CHECK: Printing analysis {{.*}} for function 'test_cold_call_sites_with_prof'
entry:
  br i1 %flag, label %then, label %else
; CHECK: edge entry -> then probability is 0x07878788 / 0x80000000 = 5.88%
; CHECK: edge entry -> else probability is 0x78787878 / 0x80000000 = 94.12% [HOT edge]

then:
  br i1 %flag2, label %then2, label %else2, !prof !3
; CHECK: edge then -> then2 probability is 0x7ebb907a / 0x80000000 = 99.01% [HOT edge]
; CHECK: edge then -> else2 probability is 0x01446f86 / 0x80000000 = 0.99%

then2:
  br label %join
; CHECK: edge then2 -> join probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

else2:
  br label %join
; CHECK: edge else2 -> join probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

join:
  %joinresult = phi i32 [ %a, %then2 ], [ %b, %else2 ]
  call void @coldfunc()
  br label %exit
; CHECK: edge join -> exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

else:
  br label %exit
; CHECK: edge else -> exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

exit:
  %result = phi i32 [ %joinresult, %join ], [ %b, %else ]
  ret i32 %result
}

!3 = !{!"branch_weights", i32 100, i32 1}

define i32 @test_cold_call_sites(i32* %a) {
; Test that edges to blocks post-dominated by cold call sites
; are marked as not expected to be taken.
; TODO(dnovillo) The calls to regular_function should not be merged, but
; they are currently being merged. Convert this into a code generation test
; after that is fixed.

; CHECK: Printing analysis {{.*}} for function 'test_cold_call_sites'
; CHECK: edge entry -> then probability is 0x07878788 / 0x80000000 = 5.88%
; CHECK: edge entry -> else probability is 0x78787878 / 0x80000000 = 94.12% [HOT edge]

entry:
  %gep1 = getelementptr i32, i32* %a, i32 1
  %val1 = load i32, i32* %gep1
  %cond1 = icmp ugt i32 %val1, 1
  br i1 %cond1, label %then, label %else

then:
  ; This function is not declared cold, but this call site is.
  %val4 = call i32 @regular_function(i32 %val1) cold
  br label %exit

else:
  %gep2 = getelementptr i32, i32* %a, i32 2
  %val2 = load i32, i32* %gep2
  %val3 = call i32 @regular_function(i32 %val2)
  br label %exit

exit:
  %ret = phi i32 [ %val4, %then ], [ %val3, %else ]
  ret i32 %ret
}

; CHECK-LABEL: test_invoke_code_callsite1
define i32 @test_invoke_code_callsite1(i1 %c) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  br i1 %c, label %if.then, label %if.end
; Edge "entry->if.end" should have higher probability based on the cold call
; heuristic which treat %if.then as a cold block because the normal destination
; of the invoke instruction in %if.then is post-dominated by ColdFunc().
; CHECK:  edge entry -> if.then probability is 0x07878788 / 0x80000000 = 5.88%
; CHECK:  edge entry -> if.end probability is 0x78787878 / 0x80000000 = 94.12% [HOT edge]

if.then:
  invoke i32 @InvokeCall()
          to label %invoke.cont unwind label %lpad
; CHECK:  edge if.then -> invoke.cont probability is 0x7ffff800 / 0x80000000 = 100.00% [HOT edge]
; CHECK:  edge if.then -> lpad probability is 0x00000800 / 0x80000000 = 0.00%

invoke.cont:
  call void @ColdFunc() #0
  br label %if.end

lpad:
  %ll = landingpad { i8*, i32 }
          cleanup
  br label %if.end

if.end:
  ret i32 0
}

; CHECK-LABEL: test_invoke_code_callsite2
define i32 @test_invoke_code_callsite2(i1 %c) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  br i1 %c, label %if.then, label %if.end

; CHECK:  edge entry -> if.then probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK:  edge entry -> if.end probability is 0x40000000 / 0x80000000 = 50.00%

if.then:
  invoke i32 @InvokeCall()
          to label %invoke.cont unwind label %lpad
; The cold call heuristic should not kick in when the cold callsite is in EH path.
; CHECK:  edge if.then -> invoke.cont probability is 0x7ffff800 / 0x80000000 = 100.00% [HOT edge]
; CHECK:  edge if.then -> lpad probability is 0x00000800 / 0x80000000 = 0.00%

invoke.cont:
  br label %if.end

lpad:
  %ll = landingpad { i8*, i32 }
          cleanup
  call void @ColdFunc() #0
  br label %if.end

if.end:
  ret i32 0
}

; CHECK-LABEL: test_invoke_code_callsite3
define i32 @test_invoke_code_callsite3(i1 %c) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  br i1 %c, label %if.then, label %if.end
; CHECK: edge entry -> if.then probability is 0x07878788 / 0x80000000 = 5.88%
; CHECK: edge entry -> if.end probability is 0x78787878 / 0x80000000 = 94.12% [HOT edge]

if.then:
  invoke i32 @InvokeCall()
          to label %invoke.cont unwind label %lpad
; Regardless of cold calls, edge weights from a invoke instruction should be
; determined by the invoke heuristic.
; CHECK: edge if.then -> invoke.cont probability is 0x7ffff800 / 0x80000000 = 100.00% [HOT edge]
; CHECK: edge if.then -> lpad probability is 0x00000800 / 0x80000000 = 0.00%

invoke.cont:
  call void @ColdFunc() #0
  br label %if.end

lpad:
  %ll = landingpad { i8*, i32 }
          cleanup
  call void @ColdFunc() #0
  br label %if.end

if.end:
  ret i32 0
}

declare i32 @__gxx_personality_v0(...)
declare void  @ColdFunc()
declare i32 @InvokeCall()

attributes #0 = { cold }


define i32 @zero1(i32 %i, i32 %a, i32 %b) {
; CHECK: Printing analysis {{.*}} for function 'zero1'
entry:
  %cond = icmp eq i32 %i, 0
  br i1 %cond, label %then, label %else
; CHECK: edge entry -> then probability is 0x30000000 / 0x80000000 = 37.50%
; CHECK: edge entry -> else probability is 0x50000000 / 0x80000000 = 62.50%

then:
  br label %exit

else:
  br label %exit

exit:
  %result = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %result
}

define i32 @zero2(i32 %i, i32 %a, i32 %b) {
; CHECK: Printing analysis {{.*}} for function 'zero2'
entry:
  %cond = icmp ne i32 %i, -1
  br i1 %cond, label %then, label %else
; CHECK: edge entry -> then probability is 0x50000000 / 0x80000000 = 62.50%
; CHECK: edge entry -> else probability is 0x30000000 / 0x80000000 = 37.50%

then:
  br label %exit

else:
  br label %exit

exit:
  %result = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %result
}

define i32 @zero3(i32 %i, i32 %a, i32 %b) {
; CHECK: Printing analysis {{.*}} for function 'zero3'
entry:
; AND'ing with a single bit bitmask essentially leads to a bool comparison,
; meaning we don't have probability information.
  %and = and i32 %i, 2
  %tobool = icmp eq i32 %and, 0
  br i1 %tobool, label %then, label %else
; CHECK: edge entry -> then probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK: edge entry -> else probability is 0x40000000 / 0x80000000 = 50.00%

then:
; AND'ing with other bitmask might be something else, so we still assume the
; usual probabilities.
  %and2 = and i32 %i, 5
  %tobool2 = icmp eq i32 %and2, 0
  br i1 %tobool2, label %else, label %exit
; CHECK: edge then -> else probability is 0x30000000 / 0x80000000 = 37.50%
; CHECK: edge then -> exit probability is 0x50000000 / 0x80000000 = 62.50%

else:
  br label %exit

exit:
  %result = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %result
}

