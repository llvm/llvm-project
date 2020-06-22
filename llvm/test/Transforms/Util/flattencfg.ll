; RUN: opt -flattencfg -S < %s | FileCheck %s


; This test checks whether the pass completes without a crash.
; The code is not transformed in any way
;
; CHECK-LABEL: @test_not_crash
define void @test_not_crash(i32 %in_a) #0 {
entry:
  %cmp0 = icmp eq i32 %in_a, -1
  %cmp1 = icmp ne i32 %in_a, 0
  %cond0 = and i1 %cmp0, %cmp1
  br i1 %cond0, label %b0, label %b1

b0:                                ; preds = %entry
  %cmp2 = icmp eq i32 %in_a, 0
  %cmp3 = icmp ne i32 %in_a, 1
  %cond1 = or i1 %cmp2, %cmp3
  br i1 %cond1, label %exit, label %b1

b1:                                       ; preds = %entry, %b0
  br label %exit

exit:                               ; preds = %entry, %b0, %b1
  ret void
}

; CHECK-LABEL: @test_not_crash2
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %0 = fcmp ult float %a
; CHECK-NEXT:    %1 = fcmp ult float %b
; CHECK-NEXT:    [[COND:%[a-z0-9]+]] = and i1 %0, %1
; CHECK-NEXT:    br i1 [[COND]], label %bb4, label %bb3
; CHECK:       bb3:
; CHECK-NEXT:    br label %bb4
; CHECK:       bb4:
; CHECK-NEXT:    ret void
define void @test_not_crash2(float %a, float %b) #0 {
entry:
  %0 = fcmp ult float %a, 1.000000e+00
  br i1 %0, label %bb0, label %bb1

bb3:                                               ; preds = %bb0
  br label %bb4

bb4:                                               ; preds = %bb0, %bb3
  ret void

bb1:                                               ; preds = %entry
  br label %bb0

bb0:                                               ; preds = %bb1, %entry
  %1 = fcmp ult float %b, 1.000000e+00
  br i1 %1, label %bb4, label %bb3
}

; CHECK-LABEL: @test_not_crash3
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %a_eq_0 = icmp eq i32 %a, 0
; CHECK-NEXT:    %a_eq_1 = icmp eq i32 %a, 1
; CHECK-NEXT:    [[COND:%[a-z0-9]+]] = or i1 %a_eq_0, %a_eq_1
; CHECK-NEXT:    br i1 [[COND]], label %bb2, label %bb3
; CHECK:       bb2:
; CHECK-NEXT:    br label %bb3
; CHECK:       bb3:
; CHECK-NEXT:    %check_badref = phi i32 [ 17, %entry ], [ 11, %bb2 ]
; CHECK-NEXT:    ret void
define void @test_not_crash3(i32 %a) #0 {
entry:
  %a_eq_0 = icmp eq i32 %a, 0
  br i1 %a_eq_0, label %bb0, label %bb1

bb0:                                              ; preds = %entry
  br label %bb1

bb1:                                              ; preds = %bb0, %entry
  %a_eq_1 = icmp eq i32 %a, 1
  br i1 %a_eq_1, label %bb2, label %bb3

bb2:                                              ; preds = %bb1
  br label %bb3

bb3:                                              ; preds = %bb2, %bb1
  %check_badref = phi i32 [ 17, %bb1 ], [ 11, %bb2 ]
  ret void
}


@g = global i32 0, align 4

; CHECK-LABEL: @test_then
; CHECK-NEXT:  entry.x:
; CHECK-NEXT:    %cmp.x = icmp ne i32 %x, 0
; CHECK-NEXT:    %cmp.y = icmp ne i32 %y, 0
; CHECK-NEXT:    [[COND:%[a-z0-9]+]] = or i1 %cmp.x, %cmp.y
; CHECK-NEXT:    br i1 [[COND]], label %if.then.y, label %exit
; CHECK:       if.then.y:
; CHECK-NEXT:    store i32 %z, i32* @g, align 4
; CHECK-NEXT:    br label %exit
; CHECK:       exit:
; CHECK-NEXT:    ret void
define void @test_then(i32 %x, i32 %y, i32 %z) {
entry.x:
  %cmp.x = icmp ne i32 %x, 0
  br i1 %cmp.x, label %if.then.x, label %entry.y

if.then.x:
  store i32 %z, i32* @g, align 4
  br label %entry.y

entry.y:
  %cmp.y = icmp ne i32 %y, 0
  br i1 %cmp.y, label %if.then.y, label %exit

if.then.y:
  store i32 %z, i32* @g, align 4
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test_else
; CHECK-NEXT:  entry.x:
; CHECK-NEXT:    %cmp.x = icmp eq i32 %x, 0
; CHECK-NEXT:    %cmp.y = icmp eq i32 %y, 0
; CHECK-NEXT:    [[COND:%[a-z0-9]+]] = and i1 %cmp.x, %cmp.y
; CHECK-NEXT:    br i1 [[COND]], label %exit, label %if.else.y
; CHECK:       if.else.y:
; CHECK-NEXT:    store i32 %z, i32* @g, align 4
; CHECK-NEXT:    br label %exit
; CHECK:       exit:
; CHECK-NEXT:    ret void
define void @test_else(i32 %x, i32 %y, i32 %z) {
entry.x:
  %cmp.x = icmp eq i32 %x, 0
  br i1 %cmp.x, label %entry.y, label %if.else.x

if.else.x:
  store i32 %z, i32* @g, align 4
  br label %entry.y

entry.y:
  %cmp.y = icmp eq i32 %y, 0
  br i1 %cmp.y, label %exit, label %if.else.y

if.else.y:
  store i32 %z, i32* @g, align 4
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test_combine_and
; CHECK-NEXT:  entry.x:
; CHECK-NEXT:    %cmp.x = icmp eq i32 %x, 0
; CHECK-NEXT:    %cmp.y = icmp eq i32 %y, 0
; CHECK-NEXT:    [[COND:%[a-z0-9]+]] = and i1 %cmp.x, %cmp.y
; CHECK-NEXT:    br i1 [[COND]], label %exit, label %if.then.y
; CHECK:       if.then.y:
; CHECK-NEXT:    store i32 %z, i32* @g, align 4
; CHECK-NEXT:    br label %exit
; CHECK:       exit:
; CHECK-NEXT:    ret void
define void @test_combine_and(i32 %x, i32 %y, i32 %z) {
entry.x:
  %cmp.x = icmp eq i32 %x, 0
  br i1 %cmp.x, label %entry.y, label %if.else.x

if.else.x:
  store i32 %z, i32* @g, align 4
  br label %entry.y

entry.y:
  %cmp.y = icmp ne i32 %y, 0
  br i1 %cmp.y, label %if.then.y, label %exit

if.then.y:
  store i32 %z, i32* @g, align 4
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test_combine_or
; CHECK-NEXT:  entry.x:
; CHECK-NEXT:    %cmp.x = icmp ne i32 %x, 0
; CHECK-NEXT:    %cmp.y = icmp ne i32 %y, 0
; CHECK-NEXT:    [[COND:%[a-z0-9]+]] = or i1 %cmp.x, %cmp.y
; CHECK-NEXT:    br i1 [[COND]], label %if.else.y, label %exit
; CHECK:       if.else.y:
; CHECK-NEXT:    store i32 %z, i32* @g, align 4
; CHECK-NEXT:    br label %exit
; CHECK:       exit:
; CHECK-NEXT:    ret void
define void @test_combine_or(i32 %x, i32 %y, i32 %z) {
entry.x:
  %cmp.x = icmp ne i32 %x, 0
  br i1 %cmp.x, label %if.then.x, label %entry.y

if.then.x:
  store i32 %z, i32* @g, align 4
  br label %entry.y

entry.y:
  %cmp.y = icmp eq i32 %y, 0
  br i1 %cmp.y, label %exit, label %if.else.y

if.else.y:
  store i32 %z, i32* @g, align 4
  br label %exit

exit:
  ret void
}
