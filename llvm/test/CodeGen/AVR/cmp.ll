; RUN: llc < %s -mtriple=avr | FileCheck %s
; RUN: llc < %s -mtriple=avr -mcpu=attiny10 | FileCheck --check-prefix=TINY %s

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

define void @cmpimm16(i16 %a) {
; CHECK-LABEL: cmpimm16:
; CHECK: cpi
; CHECK-NEXT: cpc
  %cmp = icmp eq i16 %a, 4567
  br i1 %cmp, label %if.then, label %if.else
if.then:
  tail call void @f3(i16 %a)
  br label %if.end
if.else:
  tail call void @f4(i16 %a)
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

define void @cmpimm32(i32 %a) {
; CHECK-LABEL: cmpimm32:
; CHECK: cpi
; CHECK-NEXT: cpc
; CHECK-NEXT: cpc
; CHECK-NEXT: cpc
  %cmp = icmp eq i32 %a, 6789343
  br i1 %cmp, label %if.then, label %if.else
if.then:
  tail call void @f5(i32 %a)
  br label %if.end
if.else:
  tail call void @f6(i32 %a)
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

define void @cmpimm64(i64 %a) {
; CHECK-LABEL: cmpimm64:
; CHECK: cpi
; CHECK-NEXT: cpc
; CHECK-NEXT: cpc
; CHECK-NEXT: cpc
; CHECK-NEXT: cpc
; CHECK-NEXT: cpc
; CHECK-NEXT: cpc
; CHECK-NEXT: cpc
  %cmp = icmp eq i64 %a, 234566452
  br i1 %cmp, label %if.then, label %if.else
if.then:
  tail call void @f7(i64 %a)
  br label %if.end
if.else:
  tail call void @f8(i64 %a)
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

define i16 @cmp_i16_gt_0(i16 %0) {
; CHECK-LABEL: cmp_i16_gt_0:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    ldi r18, 1
; CHECK-NEXT:    cp r1, r24
; CHECK-NEXT:    cpc r1, r25
; CHECK-NEXT:    brlt .LBB11_2
; CHECK-NEXT:  ; %bb.1:
; CHECK-NEXT:    mov r18, r1
; CHECK-NEXT:  .LBB11_2:
; CHECK-NEXT:    mov r24, r18
; CHECK-NEXT:    clr r25
; CHECK-NEXT:    ret
;
; TINY-LABEL: cmp_i16_gt_0:
; TINY:       ; %bb.0:
; TINY-NEXT:    ldi r20, 1
; TINY-NEXT:    cp r17, r24
; TINY-NEXT:    cpc r17, r25
; TINY-NEXT:    brlt .LBB11_2
; TINY-NEXT:  ; %bb.1:
; TINY-NEXT:    mov r20, r17
; TINY-NEXT:  .LBB11_2:
; TINY-NEXT:    mov r24, r20
; TINY-NEXT:    clr r25
; TINY-NEXT:    ret
  %2 = icmp sgt i16 %0, 0
  %3 = zext i1 %2 to i16
  ret i16 %3
}

define i16 @cmp_i16_gt_126(i16 %0) {
; CHECK-LABEL: cmp_i16_gt_126:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    ldi r18, 1
; CHECK-NEXT:    cpi r24, 127
; CHECK-NEXT:    cpc r25, r1
; CHECK-NEXT:    brge .LBB12_2
; CHECK-NEXT:  ; %bb.1:
; CHECK-NEXT:    mov r18, r1
; CHECK-NEXT:  .LBB12_2:
; CHECK-NEXT:    mov r24, r18
; CHECK-NEXT:    clr r25
; CHECK-NEXT:    ret
;
; TINY-LABEL: cmp_i16_gt_126:
; TINY:       ; %bb.0:
; TINY-NEXT:    ldi r20, 1
; TINY-NEXT:    cpi r24, 127
; TINY-NEXT:    cpc r25, r17
; TINY-NEXT:    brge .LBB12_2
; TINY-NEXT:  ; %bb.1:
; TINY-NEXT:    mov r20, r17
; TINY-NEXT:  .LBB12_2:
; TINY-NEXT:    mov r24, r20
; TINY-NEXT:    clr r25
; TINY-NEXT:    ret
  %2 = icmp sgt i16 %0, 126
  %3 = zext i1 %2 to i16
  ret i16 %3
}

define i16 @cmp_i16_gt_1023(i16 %0) {
; CHECK-LABEL: cmp_i16_gt_1023:
; CHECK:       ; %bb.0:
; CHECK-NEXT:    ldi r19, 4
; CHECK-NEXT:    ldi r18, 1
; CHECK-NEXT:    cp r24, r1
; CHECK-NEXT:    cpc r25, r19
; CHECK-NEXT:    brge .LBB13_2
; CHECK-NEXT:  ; %bb.1:
; CHECK-NEXT:    mov r18, r1
; CHECK-NEXT:  .LBB13_2:
; CHECK-NEXT:    mov r24, r18
; CHECK-NEXT:    clr r25
; CHECK-NEXT:    ret
;
; TINY-LABEL: cmp_i16_gt_1023:
; TINY:       ; %bb.0:
; TINY-NEXT:    ldi r21, 4
; TINY-NEXT:    ldi r20, 1
; TINY-NEXT:    cp r24, r17
; TINY-NEXT:    cpc r25, r21
; TINY-NEXT:    brge .LBB13_2
; TINY-NEXT:  ; %bb.1:
; TINY-NEXT:    mov r20, r17
; TINY-NEXT:  .LBB13_2:
; TINY-NEXT:    mov r24, r20
; TINY-NEXT:    clr r25
; TINY-NEXT:    ret
  %2 = icmp sgt i16 %0, 1023
  %3 = zext i1 %2 to i16
  ret i16 %3
}

define void @cmp_issue152097(i16 %a) addrspace(1) {
; See: https://github.com/llvm/llvm-project/issues/152097
; CHECK-LABEL: cmp_issue152097
; CHECK:      ldi r18, -1
; CHECK-NEXT: cpi r24, -2
; CHECK-NEXT: cpc r25, r18
; CHECK-NEXT: ret
  %cmp = icmp ugt i16 -2, %a
  br i1 %cmp, label %if.then, label %if.else
if.then:
  ret void
if.else:
  ret void
}
