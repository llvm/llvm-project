; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a9 | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=swift     | FileCheck %s
; Avoid some 's' 16-bit instruction which partially update CPSR (and add false
; dependency) when it isn't dependent on last CPSR defining instruction.
; rdar://8928208

define i32 @t1(i32 %a, i32 %b, i32 %c, i32 %d) nounwind readnone {
 entry:
; CHECK: t1:
; CHECK: muls [[REG:(r[0-9]+)]], r3, r2
; CHECK-NEXT: mul  [[REG2:(r[0-9]+)]], r1, r0
; CHECK-NEXT: muls r0, [[REG]], [[REG2]]
  %0 = mul nsw i32 %a, %b
  %1 = mul nsw i32 %c, %d
  %2 = mul nsw i32 %0, %1
  ret i32 %2
}

; Avoid partial CPSR dependency via loop backedge.
; rdar://10357570
define void @t2(i32* nocapture %ptr1, i32* %ptr2, i32 %c) nounwind {
entry:
; CHECK: t2:
  %tobool7 = icmp eq i32* %ptr2, null
  br i1 %tobool7, label %while.end, label %while.body

while.body:
; CHECK: while.body
; CHECK: mul r{{[0-9]+}}
; CHECK-NOT: muls
  %ptr1.addr.09 = phi i32* [ %add.ptr, %while.body ], [ %ptr1, %entry ]
  %ptr2.addr.08 = phi i32* [ %incdec.ptr, %while.body ], [ %ptr2, %entry ]
  %0 = load i32* %ptr1.addr.09, align 4
  %arrayidx1 = getelementptr inbounds i32* %ptr1.addr.09, i32 1
  %1 = load i32* %arrayidx1, align 4
  %arrayidx3 = getelementptr inbounds i32* %ptr1.addr.09, i32 2
  %2 = load i32* %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds i32* %ptr1.addr.09, i32 3
  %3 = load i32* %arrayidx4, align 4
  %add.ptr = getelementptr inbounds i32* %ptr1.addr.09, i32 4
  %mul = mul i32 %1, %0
  %mul5 = mul i32 %mul, %2
  %mul6 = mul i32 %mul5, %3
  store i32 %mul6, i32* %ptr2.addr.08, align 4
  %incdec.ptr = getelementptr inbounds i32* %ptr2.addr.08, i32 -1
  %tobool = icmp eq i32* %incdec.ptr, null
  br i1 %tobool, label %while.end, label %while.body

while.end:
  ret void
}
