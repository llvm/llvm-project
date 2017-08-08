; RUN: llc < %s -march=avr | FileCheck %s

declare void @f1(i8)
declare void @f2(i8)
define void @cmp8(i8 %a, i8 %b) {
; CHECK-LABEL: cmp8:
; CHECK: cp
; CHECK-NOT: cpc
  %cmp = icmp eq i8 %a, %b
  br i1 %cmp, label %if.then, label %if.else
if.then:
  tail call void @f1(i8 %a)
  br label %if.end
if.else:
  tail call void @f2(i8 %b)
  br label %if.end
if.end:
  ret void
}

declare void @f3(i16)
declare void @f4(i16)
define void @cmp16(i16 %a, i16 %b) {
; CHECK-LABEL: cmp16:
; CHECK: cp
; CHECK-NEXT: cpc
  %cmp = icmp eq i16 %a, %b
  br i1 %cmp, label %if.then, label %if.else
if.then:
  tail call void @f3(i16 %a)
  br label %if.end
if.else:
  tail call void @f4(i16 %b)
  br label %if.end
if.end:
  ret void
}

declare void @f5(i32)
declare void @f6(i32)
define void @cmp32(i32 %a, i32 %b) {
; CHECK-LABEL: cmp32:
; CHECK: cp
; CHECK-NEXT: cpc
; CHECK-NEXT: cpc
; CHECK-NEXT: cpc
  %cmp = icmp eq i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else
if.then:
  tail call void @f5(i32 %a)
  br label %if.end
if.else:
  tail call void @f6(i32 %b)
  br label %if.end
if.end:
  ret void
}

declare void @f7(i64)
declare void @f8(i64)
define void @cmp64(i64 %a, i64 %b) {
; CHECK-LABEL: cmp64:
; CHECK: cp
; CHECK-NEXT: cpc
; CHECK-NEXT: cpc
; CHECK-NEXT: cpc
; CHECK-NEXT: cpc
; CHECK-NEXT: cpc
; CHECK-NEXT: cpc
; CHECK-NEXT: cpc
  %cmp = icmp eq i64 %a, %b
  br i1 %cmp, label %if.then, label %if.else
if.then:
  tail call void @f7(i64 %a)
  br label %if.end
if.else:
  tail call void @f8(i64 %b)
  br label %if.end
if.end:
  ret void
}

declare void @f9()
declare void @f10()

define void @tst8(i8 %a) {
; CHECK-LABEL: tst8:
; CHECK: tst r24
; CHECK-NEXT: brmi
  %cmp = icmp sgt i8 %a, -1
  br i1 %cmp, label %if.then, label %if.else
if.then:
  tail call void @f9()
  br label %if.end
if.else:
  tail call void @f10()
  br label %if.end
if.end:
  ret void
}

define void @tst16(i16 %a) {
; CHECK-LABEL: tst16:
; CHECK: tst r25
; CHECK-NEXT: brmi
  %cmp = icmp sgt i16 %a, -1
  br i1 %cmp, label %if.then, label %if.else
if.then:
  tail call void @f9()
  br label %if.end
if.else:
  tail call void @f10()
  br label %if.end
if.end:
  ret void
}

define void @tst32(i32 %a) {
; CHECK-LABEL: tst32:
; CHECK: tst r25
; CHECK-NEXT: brmi
  %cmp = icmp sgt i32 %a, -1
  br i1 %cmp, label %if.then, label %if.else
if.then:
  tail call void @f9()
  br label %if.end
if.else:
  tail call void @f10()
  br label %if.end
if.end:
  ret void
}

define void @tst64(i64 %a) {
; CHECK-LABEL: tst64:
; CHECK: tst r25
; CHECK-NEXT: brmi
  %cmp = icmp sgt i64 %a, -1
  br i1 %cmp, label %if.then, label %if.else
if.then:
  tail call void @f9()
  br label %if.end
if.else:
  tail call void @f10()
  br label %if.end
if.end:
  ret void
}
