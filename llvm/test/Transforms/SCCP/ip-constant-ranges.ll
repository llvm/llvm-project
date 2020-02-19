; RUN: opt < %s -ipsccp -S | FileCheck %s

; Constant range for %a is [1, 48) and for %b is [301, 1000)
; CHECK-LABEL: f1
; CHECK: ret i32 undef
define internal i32 @f1(i32 %a, i32 %b) {
entry:
  %cmp.a = icmp sgt i32 %a, 300
  %cmp.b = icmp sgt i32 %b, 300
  %cmp.a2 = icmp ugt i32 %a, 300
  %cmp.b2 = icmp ugt i32 %b, 300

  %a.1 = select i1 %cmp.a, i32 1, i32 2
  %b.1 = select i1 %cmp.b, i32 1, i32 2
  %a.2 = select i1 %cmp.a2, i32 1, i32 2
  %b.2 = select i1 %cmp.b2, i32 1, i32 2
  %res1 = add i32 %a.1, %b.1
  %res2 = add i32 %a.2, %b.2
  %res3 = add i32 %res1, %res2
  ret i32 %res3
}

; Constant range for %x is [47, 302)
; CHECK-LABEL: f2
; CHECK: %cmp = icmp sgt i32 %x, 300
; CHECK: %res1 = select i1 %cmp, i32 1, i32 2
; CHECK-NEXT: %res4 = select i1 %cmp4, i32 3, i32 4
; CHECK-NEXT: %res6 = add i32 %res1, 3
; CHECK-NEXT: %res7 = add i32 5, %res4
; CHECK-NEXT: %res = add i32 %res6, 5
; CHECK-NEXT: ret i32 %res
define internal i32 @f2(i32 %x) {
entry:
  %cmp = icmp sgt i32 %x, 300
  %cmp2 = icmp ne i32 %x, 10
  %cmp3 = icmp sge i32 %x, 47
  %cmp4 = icmp ugt i32 %x, 300
  %cmp5 = icmp uge i32 %x, 47
  %res1 = select i1 %cmp, i32 1, i32 2
  %res2 = select i1 %cmp2, i32 3, i32 4
  %res3 = select i1 %cmp3, i32 5, i32 6
  %res4 = select i1 %cmp4, i32 3, i32 4
  %res5 = select i1 %cmp5, i32 5, i32 6

  %res6 = add i32 %res1, %res2
  %res7 = add i32 %res3, %res4
  %res = add i32 %res6, %res5
  ret i32 %res
}

define i32 @caller1() {
entry:
  %call1 = tail call i32 @f1(i32 1, i32 301)
  %call2 = tail call i32 @f1(i32 47, i32 999)
  %call3 = tail call i32 @f2(i32 47)
  %call4 = tail call i32 @f2(i32 301)
  %res.1 = add nsw i32 12, %call3
  %res.2 = add nsw i32 %res.1, %call4
  ret i32 %res.2
}

; x is overdefined, because constant ranges are only used for parameter
; values.
; CHECK-LABEL: f3
; CHECK: %cmp = icmp sgt i32 %x, 300
; CHECK: %res = select i1 %cmp, i32 1, i32 2
; CHECK: ret i32 %res
define internal i32 @f3(i32 %x) {
entry:
  %cmp = icmp sgt i32 %x, 300
  %res = select i1 %cmp, i32 1, i32 2
  ret i32 %res
}

; The phi node could be converted in a ConstantRange.
define i32 @caller2(i1 %cmp) {
entry:
  br i1 %cmp, label %if.true, label %end

if.true:
  br label %end

end:
  %res = phi i32 [ 0, %entry], [ 1, %if.true ]
  %call1 = tail call i32 @f3(i32 %res)
  ret i32 %call1
}

; CHECK-LABEL: f4
; CHECK: %cmp = icmp sgt i32 %x, 300
; CHECK: %res = select i1 %cmp, i32 1, i32 2
; CHECK: ret i32 %res
define internal i32 @f4(i32 %x) {
entry:
  %cmp = icmp sgt i32 %x, 300
  %res = select i1 %cmp, i32 1, i32 2
  ret i32 %res
}

; ICmp could introduce bounds on ConstantRanges.
define i32 @caller3(i32 %x) {
entry:
  %cmp = icmp sgt i32 %x, 300
  br i1 %cmp, label %if.true, label %end

if.true:
  %x.1 = tail call i32 @f4(i32 %x)
  br label %end

end:
  %res = phi i32 [ 0, %entry], [ %x.1, %if.true ]
  ret i32 %res
}

; Check to make sure we do not attempt to access lattice values in unreachable
; blocks.
define i32 @test_unreachable() {
entry:
  call i1 @test_unreachable_callee(i32 1)
  call i1 @test_unreachable_callee(i32 2)
  ret i32 1
}

define internal i1 @test_unreachable_callee(i32 %a) {
entry:
  ret i1 true

unreachablebb:
  %cmp = icmp eq i32 undef, %a
  unreachable
}

; Check that we do not attempt to get range info for non-integer types and
; crash.
define double @test_struct({ double, double } %test) {
    %v = extractvalue { double, double } %test, 0
    %r = fmul double %v, %v
    ret double %r
}

; Constant range for %x is [47, 302)
; CHECK-LABEL: @f5
; CHECK-NEXT: entry:
; CHECK-NEXT:   %cmp = icmp sgt i32 %x, undef
; CHECK-NEXT:   %cmp2 = icmp ne i32 undef, %x
; CHECK-NEXT:   %res1 = select i1 %cmp, i32 1, i32 2
; CHECK-NEXT:   %res2 = select i1 %cmp2, i32 3, i32 4
; CHECK-NEXT:   %res = add i32 %res1, %res2
; CHECK-NEXT:   ret i32 %res
define internal i32 @f5(i32 %x) {
entry:
  %cmp = icmp sgt i32 %x, undef
  %cmp2 = icmp ne i32 undef, %x
  %res1 = select i1 %cmp, i32 1, i32 2
  %res2 = select i1 %cmp2, i32 3, i32 4

  %res = add i32 %res1, %res2
  ret i32 %res
}

define i32 @caller4() {
entry:
  %call1 = tail call i32 @f5(i32 47)
  %call2 = tail call i32 @f5(i32 301)
  %res = add nsw i32 %call1, %call2
  ret i32 %res
}

; Make sure we do re-evaluate the function after ParamState changes.
; CHECK-LABEL: @recursive_f
; CHECK-LABEL: entry:
; CHECK:  %cmp = icmp eq i32 %i, 0
; CHECK-NEXT: br i1 %cmp, label %if.then, label %if.else
define internal i32 @recursive_f(i32 %i) {
entry:
  %cmp = icmp eq i32 %i, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  br label %return

if.else:                                          ; preds = %entry
  %sub = sub nsw i32 %i, 1
  %call = call i32 @recursive_f(i32 %sub)
  %add = add i32 %i, %call
  br label %return

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi i32 [ 0, %if.then ], [ %add, %if.else ]
  ret i32 %retval.0
}

; CHECK-LABEL: @caller5
; CHECK: %call = call i32 @recursive_f(i32 42)
; CHECK-NEXT: ret i32 %call
define i32 @caller5() {
entry:
  %call = call i32 @recursive_f(i32 42)
  ret i32 %call
}

define internal i32 @callee6.1(i32 %i) {
; CHECK-LABEL: define internal i32 @callee6.1(
; CHECK-NEXT:    %res = call i32 @callee6.2(i32 %i)
; CHECK-NEXT:    ret i32 undef
;
  %res = call i32 @callee6.2(i32 %i)
  ret i32 %res
}

define internal i32 @callee6.2(i32 %i) {
; CHECK-LABEL: define internal i32 @callee6.2(i32 %i) {
; CHECK-NEXT:    br label %if.then

; CHECK-LABEL: if.then:
; CHECK-NEXT:    ret i32 undef
;
  %cmp = icmp ne i32 %i, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  ret i32 1

if.else:                                          ; preds = %entry
  ret i32 2
}

define i32 @caller6() {
; CHECK-LABEL: define i32 @caller6() {
; CHECK-NEXT:    %call.1 = call i32 @callee6.1(i32 30)
; CHECK-NEXT:    %call.2 = call i32 @callee6.1(i32 43)
; CHECK-NEXT:    ret i32 2

  %call.1 = call i32 @callee6.1(i32 30)
  %call.2 = call i32 @callee6.1(i32 43)
  %res = add i32 %call.1, %call.2
  ret i32 %res
}
