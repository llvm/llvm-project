; RUN: opt < %s -simplifycfg -S | FileCheck %s

; a, b;
; fn1() {
;   if (b)
;     if (a == 0 || a == 5)
;       return a;
;   return 0;
; }

; Checking that we handle correctly the case when we have a switch
; branching multiple times to the same block

@b = common global i32 0, align 4
@a = common global i32 0, align 4

; Function Attrs: nounwind
define i32 @fn1() {
; CHECK-LABEL: @fn1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = load i32, i32* @b, align 4
; CHECK-NEXT:    [[TOBOOL:%.*]] = icmp eq i32 [[TMP0]], 0
; CHECK-NEXT:    br i1 [[TOBOOL]], label %return, label %if.then
; CHECK:       if.then:
; CHECK-NEXT:    [[TMP1:%.*]] = load i32, i32* @a, align 4
; CHECK-NEXT:    [[SWITCH_SELECTCMP:%.*]] = icmp eq i32 [[TMP1]], 0
; CHECK-NEXT:    [[SWITCH_SELECT:%.*]] = select i1 [[SWITCH_SELECTCMP:%.*]], i32 0, i32 0
; CHECK-NEXT:    [[SWITCH_SELECTCMP1:%.*]] = icmp eq i32 [[TMP1]], 5
; CHECK-NEXT:    [[SWITCH_SELECT2:%.*]] = select i1 [[SWITCH_SELECTCMP1]], i32 5, i32 [[SWITCH_SELECT]]
; CHECK-NEXT:    br label %return
; CHECK:       return:
; CHECK-NEXT:    [[RETVAL_0:%.*]] = phi i32 [ [[SWITCH_SELECT2]], %if.then ], [ 0, %entry ]
; CHECK-NEXT:    ret i32 [[RETVAL_0]]
;
entry:
  %0 = load i32, i32* @b, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %if.end3, label %if.then

if.then:
  %1 = load i32, i32* @a, align 4
  switch i32 %1, label %if.end3 [
  i32 5, label %return
  i32 0, label %return
  ]

if.end3:
  br label %return

return:
  %retval.0 = phi i32 [ 0, %if.end3 ], [ %1, %if.then ], [ %1, %if.then ]
  ret i32 %retval.0
}

