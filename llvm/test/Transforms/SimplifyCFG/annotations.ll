; RUN: opt -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S %s | FileCheck --match-full-lines %s

; The branch in %cont has !annotation metadata. Make sure generated AND
; has !annotation metadata.
define i32 @test_preserve_and(i8* %a, i8* %b, i8* %c, i8* %d) {
; CHECK-LABEL: define {{.*}} @test_preserve_and({{.*}}
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[C_1:%.*]] = icmp ult i8* [[A:%.*]], [[B:%.*]], !annotation !0
; CHECK-NEXT:    [[C_2:%.*]] = icmp uge i8* [[C:%.*]], [[D:%.*]], !annotation !0
; CHECK-NEXT:    [[OR_COND:%.*]] = and i1 [[C_1]], [[C_2]], !annotation !0
; CHECK-NEXT:    br i1 [[OR_COND]], label [[CONT1:%.*]], label [[TRAP:%.*]], !annotation !0
; CHECK:       trap: ; preds = %entry
; CHECK-NEXT:    call void @fn1()
; CHECK-NEXT:    unreachable
; CHECK:       cont1: ; preds = %entry
; CHECK-NEXT:    call void @fn2()
; CHECK-NEXT:    ret i32 0
;
entry:
  %c.1 = icmp ult i8* %a, %b, !annotation !0
  br i1 %c.1, label %cont, label %trap, !annotation !0

cont:                                             ; preds = %entry
  %c.2 = icmp uge i8* %c, %d, !annotation !0
  br i1 %c.2, label %cont1, label %trap, !annotation !0

trap:                                             ; preds = %cont, %entry
  call void @fn1()
  unreachable

cont1:                                            ; preds = %cont
  call void @fn2()
  ret i32 0
}

; The branch in %cont has !annotation metadata. Make sure generated OR
; has !annotation metadata.
define i32 @test_preserve_or(i8* %a, i8* %b, i8* %c, i8* %d) {
; CHECK-LABEL: define {{.*}} @test_preserve_or({{.*}}
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[C_1:%.*]] = icmp uge i8* [[A:%.*]], [[B:%.*]], !annotation !0
; CHECK-NEXT:    [[C_2:%.*]] = icmp uge i8* [[C:%.*]], [[D:%.*]], !annotation !0
; CHECK-NEXT:    [[OR_COND:%.*]] = or i1 [[C_1]], [[C_2]], !annotation !0
; CHECK-NEXT:    br i1 [[OR_COND]], label [[TRAP:%.*]], label [[CONT1:%.*]], !annotation !0
; CHECK:       trap: ; preds = %entry
; CHECK-NEXT:    call void @fn1()
; CHECK-NEXT:    unreachable
; CHECK:       cont1:  ; preds = %entry
; CHECK-NEXT:    call void @fn2()
; CHECK-NEXT:    ret i32 0
;
entry:
  %c.1 = icmp ult i8* %a, %b, !annotation !0
  br i1 %c.1, label %cont, label %trap, !annotation !0

cont:                                             ; preds = %entry
  %c.2 = icmp uge i8* %c, %d, !annotation !0
  br i1 %c.2, label %trap, label %cont1, !annotation !0

trap:                                             ; preds = %cont, %entry
  call void @fn1()
  unreachable

cont1:                                            ; preds = %cont
  call void @fn2()
  ret i32 0
}

; The branch in %cont has !annotation metadata. Make sure generated negation
; and OR have !annotation metadata.
define i32 @test_preserve_or_not(i8* %a, i8* %b, i8* %c, i8* %d) {
; CHECK-LABEL: define {{.*}} @test_preserve_or_not({{.*}}
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[C_1:%.*]] = icmp ult i8* [[A:%.*]], [[B:%.*]], !annotation !0
; CHECK-NEXT:    [[C_2:%.*]] = xor i1 [[C_1]], true
; CHECK-NEXT:    [[C_2_NOT:%.*]] = xor i1 [[C_2]], true, !annotation !0
; CHECK-NEXT:    [[C_3:%.*]] = icmp uge i8* [[C:%.*]], [[D:%.*]], !annotation !0
; CHECK-NEXT:    [[OR_COND:%.*]] = or i1 [[C_2_NOT]], [[C_3]], !annotation !0
; CHECK-NEXT:    br i1 [[OR_COND]], label [[TRAP:%.*]], label [[CONT1:%.*]], !annotation !0
; CHECK:       trap: ; preds = %entry
; CHECK-NEXT:    call void @fn1()
; CHECK-NEXT:    unreachable
; CHECK:       cont1:  ; preds = %entry
; CHECK-NEXT:    call void @fn2()
; CHECK-NEXT:    ret i32 0
;
entry:
  %c.1 = icmp ult i8* %a, %b, !annotation !0
  %c.2 = xor i1 %c.1, true
  br i1 %c.2, label %cont, label %trap, !annotation !0

cont:                                             ; preds = %entry
  %c.3 = icmp uge i8* %c, %d, !annotation !0
  br i1 %c.3, label %trap, label %cont1, !annotation !0

trap:                                             ; preds = %cont, %entry
  call void @fn1()
  unreachable

cont1:                                            ; preds = %cont
  call void @fn2()
  ret i32 0
}


; The branch in %cont has no !annotation metadata. Make sure generated negation
; and OR do not have !annotation metadata.
define i32 @test_or_not_no_annotation(i8* %a, i8* %b, i8* %c, i8* %d) {
; CHECK-LABEL: define {{.*}} @test_or_not_no_annotation({{.*}}
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[C_1:%.*]] = icmp ult i8* [[A:%.*]], [[B:%.*]], !annotation !0
; CHECK-NEXT:    [[C_2:%.*]] = xor i1 [[C_1]], true
; CHECK-NEXT:    [[C_2_NOT:%.*]] = xor i1 [[C_2]], true
; CHECK-NEXT:    [[C_3:%.*]] = icmp uge i8* [[C:%.*]], [[D:%.*]], !annotation !0
; CHECK-NEXT:    [[OR_COND:%.*]] = or i1 [[C_2_NOT]], [[C_3]]
; CHECK-NEXT:    br i1 [[OR_COND]], label [[TRAP:%.*]], label [[CONT1:%.*]], !annotation !0
; CHECK:       trap: ; preds = %entry
; CHECK-NEXT:    call void @fn1()
; CHECK-NEXT:    unreachable
; CHECK:       cont1:  ; preds = %entry
; CHECK-NEXT:    call void @fn2()
; CHECK-NEXT:    ret i32 0
;
entry:
  %c.1 = icmp ult i8* %a, %b, !annotation !0
  %c.2 = xor i1 %c.1, true
  br i1 %c.2, label %cont, label %trap, !annotation !0

cont:                                             ; preds = %entry
  %c.3 = icmp uge i8* %c, %d, !annotation !0
  br i1 %c.3, label %trap, label %cont1

trap:                                             ; preds = %cont, %entry
  call void @fn1()
  unreachable

cont1:                                            ; preds = %cont
  call void @fn2()
  ret i32 0
}

declare void @fn1()
declare void @fn2()

!0 = !{!"foo"}
