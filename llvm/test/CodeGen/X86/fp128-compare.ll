; RUN: llc < %s -O2 -mtriple=x86_64-linux-android -mattr=+mmx \
; RUN:     -enable-legalize-types-checking | FileCheck %s
; RUN: llc < %s -O2 -mtriple=x86_64-linux-gnu -mattr=+mmx \
; RUN:     -enable-legalize-types-checking | FileCheck %s

define i32 @TestComp128GT(fp128 %d1, fp128 %d2) {
entry:
  %cmp = fcmp ogt fp128 %d1, %d2
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: TestComp128GT:
; CHECK:       callq __gttf2
; CHECK:       xorl  %ecx, %ecx
; CHECK:       setg  %cl
; CHECK:       movl  %ecx, %eax
; CHECK:       retq
}

define i32 @TestComp128GE(fp128 %d1, fp128 %d2) {
entry:
  %cmp = fcmp oge fp128 %d1, %d2
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: TestComp128GE:
; CHECK:       callq __getf2
; CHECK:       xorl  %ecx, %ecx
; CHECK:       testl %eax, %eax
; CHECK:       setns %cl
; CHECK:       movl  %ecx, %eax
; CHECK:       retq
}

define i32 @TestComp128LT(fp128 %d1, fp128 %d2) {
entry:
  %cmp = fcmp olt fp128 %d1, %d2
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: TestComp128LT:
; CHECK:       callq __lttf2
; CHECK-NEXT:  shrl $31, %eax
; CHECK:       retq
;
; The 'shrl' is a special optimization in llvm to combine
; the effect of 'fcmp olt' and 'zext'. The main purpose is
; to test soften call to __lttf2.
}

define i32 @TestComp128LE(fp128 %d1, fp128 %d2) {
entry:
  %cmp = fcmp ole fp128 %d1, %d2
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: TestComp128LE:
; CHECK:       callq __letf2
; CHECK:       xorl  %ecx, %ecx
; CHECK:       testl %eax, %eax
; CHECK:       setle %cl
; CHECK:       movl  %ecx, %eax
; CHECK:       retq
}

define i32 @TestComp128EQ(fp128 %d1, fp128 %d2) {
entry:
  %cmp = fcmp oeq fp128 %d1, %d2
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: TestComp128EQ:
; CHECK:       callq __eqtf2
; CHECK:       xorl  %ecx, %ecx
; CHECK:       testl %eax, %eax
; CHECK:       sete  %cl
; CHECK:       movl  %ecx, %eax
; CHECK:       retq
}

define i32 @TestComp128NE(fp128 %d1, fp128 %d2) {
entry:
  %cmp = fcmp une fp128 %d1, %d2
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: TestComp128NE:
; CHECK:       callq __netf2
; CHECK:       xorl  %ecx, %ecx
; CHECK:       testl %eax, %eax
; CHECK:       setne %cl
; CHECK:       movl  %ecx, %eax
; CHECK:       retq
}

define fp128 @TestMax(fp128 %x, fp128 %y) {
entry:
  %cmp = fcmp ogt fp128 %x, %y
  %cond = select i1 %cmp, fp128 %x, fp128 %y
  ret fp128 %cond
; CHECK-LABEL: TestMax:
; CHECK: movaps %xmm0
; CHECK: movaps %xmm1
; CHECK: callq __gttf2
; CHECK: movaps {{.*}}, %xmm0
; CHECK: testl %eax, %eax
; CHECK: movaps {{.*}}, %xmm0
; CHECK: retq
}
