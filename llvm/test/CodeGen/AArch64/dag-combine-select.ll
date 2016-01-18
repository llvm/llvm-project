; RUN: llc -o - %s | FileCheck %s
target triple = "arm64--"

@out = internal global i32 0, align 4

; Ensure that we transform select(C0, x, select(C1, x, y)) towards
; select(C0 | C1, x, y) so we can use CMP;CCMP for the implementation.
; CHECK-LABEL: test0:
; CHECK: cmp w0, #7
; CHECK: ccmp w1, #0, #0, ne
; CHECK: csel w0, w1, w2, gt
; CHECK: ret
define i32 @test0(i32 %v0, i32 %v1, i32 %v2) {
  %cmp1 = icmp eq i32 %v0, 7
  %cmp2 = icmp sgt i32 %v1, 0
  %sel0 = select i1 %cmp1, i32 %v1, i32 %v2
  %sel1 = select i1 %cmp2, i32 %v1, i32 %sel0
  ret i32 %sel1
}

; Usually we keep select(C0 | C1, x, y) as is on aarch64 to create CMP;CCMP
; sequences. This case should be transformed to select(C0, select(C1, x, y), y)
; anyway to get CSE effects.
; CHECK-LABEL: test1:
; CHECK-NOT: ccmp
; CHECK: cmp w0, #7
; CHECK: adrp x[[OUTNUM:[0-9]+]], out
; CHECK: csel w[[SEL0NUM:[0-9]+]], w1, w2, eq
; CHECK: cmp w[[SEL0NUM]], #13
; CHECK: csel w[[SEL1NUM:[0-9]+]], w1, w2, lo
; CHECK: cmp w0, #42
; CHECK: csel w[[SEL2NUM:[0-9]+]], w1, w[[SEL1NUM]], eq
; CHECK: str w[[SEL1NUM]], [x[[OUTNUM]], :lo12:out]
; CHECK: str w[[SEL2NUM]], [x[[OUTNUM]], :lo12:out]
; CHECK: ret
define void @test1(i32 %bitset, i32 %val0, i32 %val1) {
  %cmp1 = icmp eq i32 %bitset, 7
  %cond = select i1 %cmp1, i32 %val0, i32 %val1
  %cmp5 = icmp ult i32 %cond, 13
  %cond11 = select i1 %cmp5, i32 %val0, i32 %val1
  %cmp3 = icmp eq i32 %bitset, 42
  %or.cond = or i1 %cmp3, %cmp5
  %cond17 = select i1 %or.cond, i32 %val0, i32 %val1
  store volatile i32 %cond11, i32* @out, align 4
  store volatile i32 %cond17, i32* @out, align 4
  ret void
}
