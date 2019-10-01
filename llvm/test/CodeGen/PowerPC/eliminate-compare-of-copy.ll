; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr8 -ppc-asm-full-reg-names < %s | FileCheck %s

define dso_local signext i32 @func(i32 zeroext %x, i32 zeroext %y) local_unnamed_addr {
; CHECK-LABEL: func
; CHECK: or. {{r[0-9]+}}, r4, r3
; CHECK-NOT: cmplwi
; CHECK: blr
entry:
  %or = or i32 %y, %x
  %tobool = icmp eq i32 %or, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %call = tail call signext i32 bitcast (i32 (...)* @f1 to i32 ()*)()
  br label %return

if.else:                                          ; preds = %entry
  %call1 = tail call signext i32 bitcast (i32 (...)* @f2 to i32 ()*)()
  br label %return

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi i32 [ %call, %if.then ], [ %call1, %if.else ]
  ret i32 %retval.0
}

declare signext i32 @f1(...) local_unnamed_addr

declare signext i32 @f2(...) local_unnamed_addr
