; RUN: llc -O3 < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32:64"
target triple = "arm64-unknown-unknown"

; CHECK-LABEL: foo1
; CHECK: cinc w{{[0-9]+}}, w{{[0-9]+}}, ne
define i32 @foo1(i32 %b, i32 %c) nounwind readnone ssp {
entry:
  %not.tobool = icmp ne i32 %c, 0
  %add = zext i1 %not.tobool to i32
  %b.add = add i32 %c, %b
  %add1 = add i32 %b.add, %add
  ret i32 %add1
}

; CHECK-LABEL: foo2
; CHECK: cneg w{{[0-9]+}}, w{{[0-9]+}}, ne
define i32 @foo2(i32 %b, i32 %c) nounwind readnone ssp {
entry:
  %mul = sub i32 0, %b
  %tobool = icmp eq i32 %c, 0
  %b.mul = select i1 %tobool, i32 %b, i32 %mul
  %add = add nsw i32 %b.mul, %c
  ret i32 %add
}

; CHECK-LABEL: foo3
; CHECK: cinv w{{[0-9]+}}, w{{[0-9]+}}, ne
define i32 @foo3(i32 %b, i32 %c) nounwind readnone ssp {
entry:
  %not.tobool = icmp ne i32 %c, 0
  %xor = sext i1 %not.tobool to i32
  %b.xor = xor i32 %xor, %b
  %add = add nsw i32 %b.xor, %c
  ret i32 %add
}

; rdar://11632325
define i32@foo4(i32 %a) nounwind ssp {
; CHECK-LABEL: foo4
; CHECK: cneg
; CHECK-NEXT: ret
  %cmp = icmp sgt i32 %a, -1
  %neg = sub nsw i32 0, %a
  %cond = select i1 %cmp, i32 %a, i32 %neg
  ret i32 %cond
}

define i32@foo5(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK-LABEL: foo5
; CHECK: subs
; CHECK-NEXT: cneg
; CHECK-NEXT: ret
  %sub = sub nsw i32 %a, %b
  %cmp = icmp sgt i32 %sub, -1
  %sub3 = sub nsw i32 0, %sub
  %cond = select i1 %cmp, i32 %sub, i32 %sub3
  ret i32 %cond
}

; make sure we can handle branch instruction in optimizeCompare.
define i32@foo6(i32 %a, i32 %b) nounwind ssp {
; CHECK-LABEL: foo6
; CHECK: b
  %sub = sub nsw i32 %a, %b
  %cmp = icmp sgt i32 %sub, 0
  br i1 %cmp, label %l.if, label %l.else

l.if:
  ret i32 1

l.else:
  ret i32 %sub
}

; If CPSR is used multiple times and V flag is used, we don't remove cmp.
define i32 @foo7(i32 %a, i32 %b) nounwind {
entry:
; CHECK-LABEL: foo7:
; CHECK: sub
; FIXME: Misspelled CHECK-NEXT
; CHECK-next: adds
; CHECK-next: csneg
; CHECK-next: b
  %sub = sub nsw i32 %a, %b
  %cmp = icmp sgt i32 %sub, -1
  %sub3 = sub nsw i32 0, %sub
  %cond = select i1 %cmp, i32 %sub, i32 %sub3
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %cmp2 = icmp slt i32 %sub, -1
  %sel = select i1 %cmp2, i32 %cond, i32 %a
  ret i32 %sel

if.else:
  ret i32 %cond
}

define i32 @foo8(i32 %v, i32 %a, i32 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: foo8:
; CHECK: cmp w0, #0
; CHECK: csinv w0, w1, w2, ne
  %tobool = icmp eq i32 %v, 0
  %neg = xor i32 -1, %b
  %cond = select i1 %tobool, i32 %neg, i32 %a
  ret i32 %cond
}

define i32 @foo9(i32 %v) nounwind readnone optsize ssp {
entry:
; CHECK-LABEL: foo9:
; CHECK: cmp w0, #0
; CHECK: orr w[[REG:[0-9]+]], wzr, #0x4
; CHECK: cinv w0, w[[REG]], eq
  %tobool = icmp ne i32 %v, 0
  %cond = select i1 %tobool, i32 4, i32 -5
  ret i32 %cond
}

define i64 @foo10(i64 %v) nounwind readnone optsize ssp {
entry:
; CHECK-LABEL: foo10:
; CHECK: cmp x0, #0
; CHECK: orr w[[REG:[0-9]+]], wzr, #0x4
; CHECK: cinv x0, x[[REG]], eq
  %tobool = icmp ne i64 %v, 0
  %cond = select i1 %tobool, i64 4, i64 -5
  ret i64 %cond
}

define i32 @foo11(i32 %v) nounwind readnone optsize ssp {
entry:
; CHECK-LABEL: foo11:
; CHECK: cmp w0, #0
; CHECK: orr w[[REG:[0-9]+]], wzr, #0x4
; CHECK: cneg w0, w[[REG]], eq
  %tobool = icmp ne i32 %v, 0
  %cond = select i1 %tobool, i32 4, i32 -4
  ret i32 %cond
}

define i64 @foo12(i64 %v) nounwind readnone optsize ssp {
entry:
; CHECK-LABEL: foo12:
; CHECK: cmp x0, #0
; CHECK: orr w[[REG:[0-9]+]], wzr, #0x4
; CHECK: cneg x0, x[[REG]], eq
  %tobool = icmp ne i64 %v, 0
  %cond = select i1 %tobool, i64 4, i64 -4
  ret i64 %cond
}

define i32 @foo13(i32 %v, i32 %a, i32 %b) nounwind readnone optsize ssp {
entry:
; CHECK-LABEL: foo13:
; CHECK: cmp w0, #0
; CHECK: csneg w0, w1, w2, ne
  %tobool = icmp eq i32 %v, 0
  %sub = sub i32 0, %b
  %cond = select i1 %tobool, i32 %sub, i32 %a
  ret i32 %cond
}

define i64 @foo14(i64 %v, i64 %a, i64 %b) nounwind readnone optsize ssp {
entry:
; CHECK-LABEL: foo14:
; CHECK: cmp x0, #0
; CHECK: csneg x0, x1, x2, ne
  %tobool = icmp eq i64 %v, 0
  %sub = sub i64 0, %b
  %cond = select i1 %tobool, i64 %sub, i64 %a
  ret i64 %cond
}

define i32 @foo15(i32 %a, i32 %b) nounwind readnone optsize ssp {
entry:
; CHECK-LABEL: foo15:
; CHECK: cmp w0, w1
; CHECK: orr w[[REG:[0-9]+]], wzr, #0x1
; CHECK: cinc w0, w[[REG]], gt
  %cmp = icmp sgt i32 %a, %b
  %. = select i1 %cmp, i32 2, i32 1
  ret i32 %.
}

define i32 @foo16(i32 %a, i32 %b) nounwind readnone optsize ssp {
entry:
; CHECK-LABEL: foo16:
; CHECK: cmp w0, w1
; CHECK: orr w[[REG:[0-9]+]], wzr, #0x1
; CHECK: cinc w0, w[[REG]], le
  %cmp = icmp sgt i32 %a, %b
  %. = select i1 %cmp, i32 1, i32 2
  ret i32 %.
}

define i64 @foo17(i64 %a, i64 %b) nounwind readnone optsize ssp {
entry:
; CHECK-LABEL: foo17:
; CHECK: cmp x0, x1
; CHECK: orr w[[REG:[0-9]+]], wzr, #0x1
; CHECK: cinc x0, x[[REG]], gt
  %cmp = icmp sgt i64 %a, %b
  %. = select i1 %cmp, i64 2, i64 1
  ret i64 %.
}

define i64 @foo18(i64 %a, i64 %b) nounwind readnone optsize ssp {
entry:
; CHECK-LABEL: foo18:
; CHECK: cmp x0, x1
; CHECK: orr w[[REG:[0-9]+]], wzr, #0x1
; CHECK: cinc x0, x[[REG]], le
  %cmp = icmp sgt i64 %a, %b
  %. = select i1 %cmp, i64 1, i64 2
  ret i64 %.
}

define i64 @foo19(i64 %a, i64 %b, i64 %c) {
entry:
; CHECK-LABEL: foo19:
; CHECK: cinc x0, x2
; CHECK-NOT: add
  %cmp = icmp ult i64 %a, %b
  %inc = zext i1 %cmp to i64
  %inc.c = add i64 %inc, %c
  ret i64 %inc.c
}

define i32 @foo20(i32 %x) {
; CHECK-LABEL: foo20:
; CHECK: cmp w0, #5
; CHECK: orr w[[REG:[0-9]+]], wzr, #0x6
; CHECK: csinc w0, w[[REG]], wzr, eq
  %cmp = icmp eq i32 %x, 5
  %res = select i1 %cmp, i32 6, i32 1
  ret i32 %res
}

define i64 @foo21(i64 %x) {
; CHECK-LABEL: foo21:
; CHECK: cmp x0, #5
; CHECK: orr w[[REG:[0-9]+]], wzr, #0x6
; CHECK: csinc x0, x[[REG]], xzr, eq
  %cmp = icmp eq i64 %x, 5
  %res = select i1 %cmp, i64 6, i64 1
  ret i64 %res
}

define i32 @foo22(i32 %x) {
; CHECK-LABEL: foo22:
; CHECK: cmp w0, #5
; CHECK: orr w[[REG:[0-9]+]], wzr, #0x6
; CHECK: csinc w0, w[[REG]], wzr, ne
  %cmp = icmp eq i32 %x, 5
  %res = select i1 %cmp, i32 1, i32 6
  ret i32 %res
}

define i64 @foo23(i64 %x) {
; CHECK-LABEL: foo23:
; CHECK: cmp x0, #5
; CHECK: orr w[[REG:[0-9]+]], wzr, #0x6
; CHECK: csinc x0, x[[REG]], xzr, ne
  %cmp = icmp eq i64 %x, 5
  %res = select i1 %cmp, i64 1, i64 6
  ret i64 %res
}
