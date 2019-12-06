; RUN: llc < %s -O1 -mtriple=aarch64-eabi -aarch64-enable-cond-br-tune=false | FileCheck %s

declare void @t()

define void @test1(i32 %a) {
; CHECK-LABEL: @test1
entry:
  %sub = add nsw i32 %a, -12
  %cmp = icmp slt i32 %sub, 0
  br i1 %cmp, label %if.then, label %if.end

; CHECK: sub [[CMP:w[0-9]+]], w0, #12
; CHECK: tbnz [[CMP]], #31

if.then:
  call void @t()
  br label %if.end

if.end:
  ret void
}

define void @test2(i64 %a) {
; CHECK-LABEL: @test2
entry:
  %sub = add nsw i64 %a, -12
  %cmp = icmp slt i64 %sub, 0
  br i1 %cmp, label %if.then, label %if.end

; CHECK: sub [[CMP:x[0-9]+]], x0, #12
; CHECK: tbnz [[CMP]], #63

if.then:
  call void @t()
  br label %if.end

if.end:
  ret void
}

define void @test3(i32 %a) {
; CHECK-LABEL: @test3
entry:
  %sub = add nsw i32 %a, -12
  %cmp = icmp sgt i32 %sub, -1
  br i1 %cmp, label %if.then, label %if.end

; CHECK: sub [[CMP:w[0-9]+]], w0, #12
; CHECK: tbnz [[CMP]], #31

if.then:
  call void @t()
  br label %if.end

if.end:
  ret void
}

define void @test4(i64 %a) {
; CHECK-LABEL: @test4
entry:
  %sub = add nsw i64 %a, -12
  %cmp = icmp sgt i64 %sub, -1
  br i1 %cmp, label %if.then, label %if.end

; CHECK: sub [[CMP:x[0-9]+]], x0, #12
; CHECK: tbnz [[CMP]], #63

if.then:
  call void @t()
  br label %if.end

if.end:
  ret void
}

define void @test5(i32 %a) {
; CHECK-LABEL: @test5
entry:
  %sub = add nsw i32 %a, -12
  %cmp = icmp sge i32 %sub, 0
  br i1 %cmp, label %if.then, label %if.end

; CHECK: sub [[CMP:w[0-9]+]], w0, #12
; CHECK: tbnz [[CMP]], #31

if.then:
  call void @t()
  br label %if.end

if.end:
  ret void
}

define void @test6(i64 %a) {
; CHECK-LABEL: @test6
entry:
  %sub = add nsw i64 %a, -12
  %cmp = icmp sge i64 %sub, 0
  br i1 %cmp, label %if.then, label %if.end

; CHECK: sub [[CMP:x[0-9]+]], x0, #12
; CHECK: tbnz [[CMP]], #63

if.then:
  call void @t()
  br label %if.end

if.end:
  ret void
}

define void @test7(i32 %a) {
; CHECK-LABEL: @test7
entry:
  %sub = sub nsw i32 %a, 12
  %cmp = icmp slt i32 %sub, 0
  br i1 %cmp, label %if.then, label %if.end

; CHECK: sub [[CMP:w[0-9]+]], w0, #12
; CHECK: tbnz [[CMP]], #31

if.then:
  call void @t()
  br label %if.end

if.end:
  ret void
}

define void @test8(i64 %val1, i64 %val2, i64 %val3) {
; CHECK-LABEL: @test8
  %and1 = and i64 %val1, %val2
  %tst1 = icmp slt i64 %and1, 0
  br i1 %tst1, label %if.then1, label %if.end

; CHECK: tst x0, x1
; CHECK-NEXT: b.ge

if.then1:
  %and2 = and i64 %val2, %val3
  %tst2 = icmp sge i64 %and2, 0
  br i1 %tst2, label %if.then2, label %if.end

; CHECK: and [[CMP:x[0-9]+]], x1, x2
; CHECK-NOT: cmp
; CHECK: tbnz [[CMP]], #63

if.then2:
  %shifted_op1 = shl i64 %val2, 63
  %shifted_and1 = and i64 %val1, %shifted_op1
  %tst3 = icmp slt i64 %shifted_and1, 0
  br i1 %tst3, label %if.then3, label %if.end

; CHECK: tst x0, x1, lsl #63
; CHECK: b.lt

if.then3:
  %shifted_op2 = shl i64 %val2, 62
  %shifted_and2 = and i64 %val1, %shifted_op2
  %tst4 = icmp sge i64 %shifted_and2, 0
  br i1 %tst4, label %if.then4, label %if.end

; CHECK: tst x0, x1, lsl #62
; CHECK: b.lt

if.then4:
  call void @t()
  br label %if.end

if.end:
  ret void
}

define void @test9(i64 %val1) {
; CHECK-LABEL: @test9
  %tst = icmp slt i64 %val1, 0
  br i1 %tst, label %if.then, label %if.end

; CHECK-NOT: cmp
; CHECK: tbnz x0, #63

if.then:
  call void @t()
  br label %if.end

if.end:
  ret void
}

define void @test10(i64 %val1) {
; CHECK-LABEL: @test10
  %tst = icmp slt i64 %val1, 0
  br i1 %tst, label %if.then, label %if.end

; CHECK-NOT: cmp
; CHECK: tbnz x0, #63

if.then:
  call void @t()
  br label %if.end

if.end:
  ret void
}

define void @test11(i64 %val1, i64* %ptr) {
; CHECK-LABEL: @test11

; CHECK: ldr [[CMP:x[0-9]+]], [x1]
; CHECK-NOT: cmp
; CHECK: tbnz [[CMP]], #63

  %val = load i64, i64* %ptr
  %tst = icmp slt i64 %val, 0
  br i1 %tst, label %if.then, label %if.end

if.then:
  call void @t()
  br label %if.end

if.end:
  ret void
}

define void @test12(i64 %val1) {
; CHECK-LABEL: @test12
  %tst = icmp slt i64 %val1, 0
  br i1 %tst, label %if.then, label %if.end

; CHECK-NOT: cmp
; CHECK: tbnz x0, #63

if.then:
  call void @t()
  br label %if.end

if.end:
  ret void
}

define void @test13(i64 %val1, i64 %val2) {
; CHECK-LABEL: @test13
  %or = or i64 %val1, %val2
  %tst = icmp slt i64 %or, 0
  br i1 %tst, label %if.then, label %if.end

; CHECK: orr [[CMP:x[0-9]+]], x0, x1
; CHECK-NOT: cmp
; CHECK: tbnz [[CMP]], #63

if.then:
  call void @t()
  br label %if.end

if.end:
  ret void
}

define void @test14(i1 %cond) {
; CHECK-LABEL: @test14
  br i1 %cond, label %if.end, label %if.then

; CHECK-NOT: and
; CHECK: tbnz w0, #0

if.then:
  call void @t()
  br label %if.end

if.end:
  ret void
}

define void @test15(i1 %cond) {
; CHECK-LABEL: @test15
  %cond1 = xor i1 %cond, -1
  br i1 %cond1, label %if.then, label %if.end

; CHECK-NOT: movn
; CHECK: tbnz w0, #0

if.then:
  call void @t()
  br label %if.end

if.end:
  ret void
}

define void @test16(i64 %in) {
; CHECK-LABEL: @test16
  %shl = shl i64 %in, 3
  %and = and i64 %shl, 32
  %cond = icmp eq i64 %and, 0
  br i1 %cond, label %then, label %end

; CHECK-NOT: lsl
; CHECK: tbnz w0, #2

then:
  call void @t()
  br label %end

end:
  ret void
}

define void @test17(i64 %in) {
; CHECK-LABEL: @test17
  %shr = ashr i64 %in, 3
  %and = and i64 %shr, 1
  %cond = icmp eq i64 %and, 0
  br i1 %cond, label %then, label %end

; CHECK-NOT: lsr
; CHECK: tbnz w0, #3

then:
  call void @t()
  br label %end

end:
  ret void
}

define void @test18(i32 %in) {
; CHECK-LABEL: @test18
  %shr = ashr i32 %in, 2
  %cond = icmp sge i32 %shr, 0
  br i1 %cond, label %then, label %end

; CHECK-NOT: asr
; CHECK: tbnz w0, #31

then:
  call void @t()
  br label %end

end:
  ret void
}

define void @test19(i64 %in) {
; CHECK-LABEL: @test19
  %shl = lshr i64 %in, 3
  %trunc = trunc i64 %shl to i32
  %and = and i32 %trunc, 1
  %cond = icmp eq i32 %and, 0
  br i1 %cond, label %then, label %end

; CHECK-NOT: ubfx
; CHECK: tbnz w0, #3

then:
  call void @t()
  br label %end

end:
  ret void
}

define void @test20(i32 %in) nounwind {
; CHECK-LABEL: test20:
; CHECK:       // %bb.0:
; CHECK-NEXT:    tbnz w0, #2, .LBB19_2
; CHECK-NEXT:  // %bb.1: // %then
; CHECK-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; CHECK-NEXT:    bl t
; CHECK-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; CHECK-NEXT:  .LBB19_2: // %end
; CHECK-NEXT:    ret
  %shl = shl i32 %in, 3
  %zext = zext i32 %shl to i64
  %and = and i64 %zext, 32
  %cond = icmp eq i64 %and, 0
  br i1 %cond, label %then, label %end


then:
  call void @t()
  br label %end

end:
  ret void
}

