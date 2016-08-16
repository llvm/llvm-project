; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a9 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-CORTEX
; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=swift     | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-SWIFT
; Avoid some 's' 16-bit instruction which partially update CPSR (and add false
; dependency) when it isn't dependent on last CPSR defining instruction.
; rdar://8928208

define i32 @t1(i32 %a, i32 %b, i32 %c, i32 %d) nounwind readnone {
 entry:
; CHECK-LABEL: t1:
; CHECK-CORTEX: muls [[REG:(r[0-9]+)]], r3, r2
; CHECK-CORTEX-NEXT: mul  [[REG2:(r[0-9]+)]], r1, r0
; CHECK-SWIFT: muls  [[REG2:(r[0-9]+)]], r1, r0
; CHECK-SWIFT-NEXT: mul [[REG:(r[0-9]+)]], r2, r3
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
; CHECK-LABEL: t2:
  br label %while.body

while.body:
; CHECK: while.body
; CHECK: mul r{{[0-9]+}}
; CHECK-NOT: muls
  %ptr1.addr.09 = phi i32* [ %add.ptr, %while.body ], [ %ptr1, %entry ]
  %ptr2.addr.08 = phi i32* [ %incdec.ptr, %while.body ], [ %ptr2, %entry ]
  %0 = load i32, i32* %ptr1.addr.09, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %ptr1.addr.09, i32 1
  %1 = load i32, i32* %arrayidx1, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %ptr1.addr.09, i32 2
  %2 = load i32, i32* %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds i32, i32* %ptr1.addr.09, i32 3
  %3 = load i32, i32* %arrayidx4, align 4
  %add.ptr = getelementptr inbounds i32, i32* %ptr1.addr.09, i32 4
  %mul = mul i32 %1, %0
  %mul5 = mul i32 %mul, %2
  %mul6 = mul i32 %mul5, %3
  store i32 %mul6, i32* %ptr2.addr.08, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %ptr2.addr.08, i32 -1
  %tobool = icmp eq i32* %incdec.ptr, null
  br i1 %tobool, label %while.end, label %while.body

while.end:
  ret void
}

; Allow partial CPSR dependency when code size is the priority.
; rdar://12878928
define void @t3(i32* nocapture %ptr1, i32* %ptr2, i32 %c) nounwind minsize {
entry:
; CHECK-LABEL: t3:
  br label %while.body

while.body:
; CHECK: while.body
; CHECK: mul r{{[0-9]+}}
; CHECK: muls
  %ptr1.addr.09 = phi i32* [ %add.ptr, %while.body ], [ %ptr1, %entry ]
  %ptr2.addr.08 = phi i32* [ %incdec.ptr, %while.body ], [ %ptr2, %entry ]
  %0 = load i32, i32* %ptr1.addr.09, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %ptr1.addr.09, i32 1
  %1 = load i32, i32* %arrayidx1, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %ptr1.addr.09, i32 2
  %2 = load i32, i32* %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds i32, i32* %ptr1.addr.09, i32 3
  %3 = load i32, i32* %arrayidx4, align 4
  %add.ptr = getelementptr inbounds i32, i32* %ptr1.addr.09, i32 4
  %mul = mul i32 %1, %0
  %mul5 = mul i32 %mul, %2
  %mul6 = mul i32 %mul5, %3
  store i32 %mul6, i32* %ptr2.addr.08, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %ptr2.addr.08, i32 -1
  %tobool = icmp eq i32* %incdec.ptr, null
  br i1 %tobool, label %while.end, label %while.body

while.end:
  ret void
}

; Avoid producing tMOVi8 after a high-latency flag-setting operation.
; <rdar://problem/13468102>
define void @t4(i32* nocapture %p, double* nocapture %q) {
entry:
; CHECK: t4
; CHECK: vmrs APSR_nzcv, fpscr
; CHECK: if.then
; CHECK-NOT: movs
  %0 = load double, double* %q, align 4
  %cmp = fcmp olt double %0, 1.000000e+01
  %incdec.ptr1 = getelementptr inbounds i32, i32* %p, i32 1
  br i1 %cmp, label %if.then, label %if.else

if.then:
  store i32 7, i32* %p, align 4
  %incdec.ptr2 = getelementptr inbounds i32, i32* %p, i32 2
  store i32 8, i32* %incdec.ptr1, align 4
  store i32 9, i32* %incdec.ptr2, align 4
  br label %if.end

if.else:
  store i32 3, i32* %p, align 4
  %incdec.ptr5 = getelementptr inbounds i32, i32* %p, i32 2
  store i32 5, i32* %incdec.ptr1, align 4
  store i32 6, i32* %incdec.ptr5, align 4
  br label %if.end

if.end:
  ret void
}
