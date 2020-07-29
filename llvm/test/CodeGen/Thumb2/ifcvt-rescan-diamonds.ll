; RUN: llc -O2 -o - %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8-unknown-linux-gnueabihf"

; This is a tricky test case.
; The point of it is to create a diamond where both the true block and the
; false block clobber the predicate when viewed as a whole, but only one of them
; clobbers the predicate when considering the instructions they share.

; Function Attrs: nounwind
define void @BN_kronecker(i1 %a, i32 %b) #0 {
entry:
  br label %while.cond38

while.cond38:                                     ; preds = %if.end111, %entry
  %cmp79 = icmp eq i32 0, 0
  br i1 %a, label %cond.true77, label %cond.false87

; CHECK: %cond.true77
; CHECK-NEXT: @ in Loop
; CHECK-NEXT: cmp.w {{r[0-9]+}}, #0
; CHECK-NEXT: it eq
; CHECK-NEXT: ldreq
; CHECK-NEXT: it ne
  ; N.b. 16-bit mov instruction in IT block does not set flags.
; CHECK-NEXT: movne
; CHECK-NEXT: mvns
; CHECK-NEXT: b
cond.true77:                                      ; preds = %while.cond38
  br i1 %cmp79, label %cond.end84, label %cond.false81

cond.false81:                                     ; preds = %cond.true77
  %0 = load i32, i32* null, align 4
  br label %cond.end84

cond.end84:                                       ; preds = %cond.false81, %cond.true77
  %cond85 = phi i32 [ %0, %cond.false81 ], [ 0, %cond.true77 ]
  %neg86 = xor i32 %cond85, -1
  br label %cond.false101

cond.false87:                                     ; preds = %while.cond38
  br i1 %cmp79, label %cond.false101, label %cond.false91

cond.false91:                                     ; preds = %cond.false87
  br label %cond.false101

cond.false101:                                    ; preds = %cond.false91, %cond.false87, %cond.end84
  %cond97 = phi i32 [ %neg86, %cond.end84 ], [ %b, %cond.false91 ], [ 0, %cond.false87 ]
  %1 = load i32, i32* null, align 4
  %and106 = and i32 %cond97, %1
  %and107 = and i32 %and106, 2
  %tobool108 = icmp ne i32 %and107, 0
  br i1 %tobool108, label %if.then109, label %if.end111

if.then109:                                       ; preds = %cond.false101
  store i32 0, i32* undef, align 4
  br label %if.end111

if.end111:                                        ; preds = %if.then109, %cond.false101
  %tobool113 = icmp ne i32 0, 0
  br i1 %tobool113, label %while.cond38, label %end

end:                                              ; preds = %if.end111
  ret void
}

attributes #0 = { nounwind }
