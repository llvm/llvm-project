; RUN: opt < %s -correlated-propagation -S | FileCheck %s

; CHECK-LABEL: @simple(
define i8 @simple(i1) {
entry:
  %s = select i1 %0, i8 0, i8 1
  br i1 %0, label %then, label %else

then:
; CHECK: ret i8 0
  %a = phi i8 [ %s, %entry ]
  ret i8 %a

else:
; CHECK: ret i8 1
  %b = phi i8 [ %s, %entry ]
  ret i8 %b
}

; CHECK-LABEL: @loop(
define void @loop(i32) {
entry:
  br label %loop

loop:
  %idx = phi i32 [ %0, %entry ], [ %sel, %loop ]
; CHECK: %idx = phi i32 [ %0, %entry ], [ %2, %loop ]
  %1 = icmp eq i32 %idx, 0
  %2 = add i32 %idx, -1
  %sel = select i1 %1, i32 0, i32 %2
  br i1 %1, label %out, label %loop

out:
  ret void
}

; CHECK-LABEL: @not_correlated(
define i8 @not_correlated(i1, i1) {
entry:
  %s = select i1 %0, i8 0, i8 1
  br i1 %1, label %then, label %else

then:
; CHECK: ret i8 %s
  %a = phi i8 [ %s, %entry ]
  ret i8 %a

else:
; CHECK: ret i8 %s
  %b = phi i8 [ %s, %entry ]
  ret i8 %b
}

@c = global i32 0, align 4
@b = global i32 0, align 4

; CHECK-LABEL: @PR23752(
define i32 @PR23752() {
entry:
  br label %for.body

for.body:
  %phi = phi i32 [ 0, %entry ], [ %sel, %for.body ]
  %sel = select i1 icmp sgt (i32* @b, i32* @c), i32 %phi, i32 1
  %cmp = icmp ne i32 %sel, 1
  br i1 %cmp, label %for.body, label %if.end

; CHECK:      %[[sel:.*]] = select i1 icmp sgt (i32* @b, i32* @c), i32 0, i32 1
; CHECK-NEXT: %[[cmp:.*]] = icmp ne i32 %[[sel]], 1
; CHECK-NEXT: br i1 %[[cmp]]

if.end:
  ret i32 %sel
; CHECK: ret i32 1
}

define i1 @test1(i32* %p, i1 %unknown) {
; CHECK-LABEL: @test1
  %pval = load i32, i32* %p
  %cmp1 = icmp slt i32 %pval, 255
  br i1 %cmp1, label %next, label %exit

next:
  %min = select i1 %unknown, i32 %pval, i32 5
  ;; TODO: This pointless branch shouldn't be neccessary
  br label %next2
next2:
; CHECK-LABEL: next2:
; CHECK: ret i1 false
  %res = icmp eq i32 %min, 255
  ret i1 %res

exit:
; CHECK-LABEL: exit:
; CHECK: ret i1 true
  ret i1 true
}

; Check that we take a conservative meet
define i1 @test2(i32* %p, i32 %qval, i1 %unknown) {
; CHECK-LABEL: test2
  %pval = load i32, i32* %p
  %cmp1 = icmp slt i32 %pval, 255
  br i1 %cmp1, label %next, label %exit

next:
  %min = select i1 %unknown, i32 %pval, i32 %qval
  ;; TODO: This pointless branch shouldn't be neccessary
  br label %next2
next2:
; CHECK-LABEL: next2
; CHECK: ret i1 %res
  %res = icmp eq i32 %min, 255
  ret i1 %res

exit:
; CHECK-LABEL: exit:
; CHECK: ret i1 true
  ret i1 true
}

; Same as @test2, but for the opposite select input
define i1 @test3(i32* %p, i32 %qval, i1 %unknown) {
; CHECK-LABEL: test3
  %pval = load i32, i32* %p
  %cmp1 = icmp slt i32 %pval, 255
  br i1 %cmp1, label %next, label %exit

next:
  %min = select i1 %unknown, i32 %qval, i32 %pval
  ;; TODO: This pointless branch shouldn't be neccessary
  br label %next2
next2:
; CHECK-LABEL: next2
; CHECK: ret i1 %res
  %res = icmp eq i32 %min, 255
  ret i1 %res

exit:
; CHECK-LABEL: exit:
; CHECK: ret i1 true
  ret i1 true
}

; Conflicting constants (i.e. isOverdefined result)
; NOTE: Using doubles in this version is a bit of a hack.  This
; is to get around the fact that all integers (including constants
; and non-constants) are actually represented as constant-ranges.
define i1 @test4(i32* %p, i32 %qval, i1 %unknown) {
; CHECK-LABEL: test4
  %pval = load i32, i32* %p
  %cmp1 = icmp slt i32 %pval, 255
  br i1 %cmp1, label %next, label %exit

next:
  %min = select i1 %unknown, double 1.0, double 0.0
  ;; TODO: This pointless branch shouldn't be neccessary
  br label %next2
next2:
; CHECK-LABEL: next2
; CHECK: ret i1 %res
  %res = fcmp oeq double %min, 300.0
  ret i1 %res

exit:
; CHECK-LABEL: exit:
; CHECK: ret i1 true
  ret i1 true
}

;; Using the condition to clamp the result
;; 

define i1 @test5(i32* %p, i1 %unknown) {
; CHECK-LABEL: @test5
  %pval = load i32, i32* %p
  %cmp1 = icmp slt i32 %pval, 255
  br i1 %cmp1, label %next, label %exit

next:
  %cond = icmp sgt i32 %pval, 0
  %min = select i1 %cond, i32 %pval, i32 5
  ;; TODO: This pointless branch shouldn't be neccessary
  br label %next2
next2:
; CHECK-LABEL: next2:
; CHECK: ret i1 false
  %res = icmp eq i32 %min, -1
  ret i1 %res

exit:
; CHECK-LABEL: exit:
; CHECK: ret i1 true
  ret i1 true
}

define i1 @test6(i32* %p, i1 %unknown) {
; CHECK-LABEL: @test6
  %pval = load i32, i32* %p
  %cmp1 = icmp ult i32 %pval, 255
  br i1 %cmp1, label %next, label %exit

next:
  %cond = icmp ne i32 %pval, 254
  %sel = select i1 %cond, i32 %pval, i32 1
  ;; TODO: This pointless branch shouldn't be neccessary
  br label %next2
next2:
; CHECK-LABEL: next2:
; CHECK: ret i1 true
  %res = icmp slt i32 %sel, 254
  ret i1 %res

exit:
; CHECK-LABEL: exit:
; CHECK: ret i1 true
  ret i1 true
}
