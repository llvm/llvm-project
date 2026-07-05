; Check if extract_subvectors is handled properly in Hexagon backend when the
; the source vector is a vector-pair and result vector is not hvx vector size.
; https://github.com/llvm/llvm-project/issues/128775
;
; Example of such a case:
;    ...
;    t2: v64i32,ch = CopyFromReg t0, Register:v64i32 %0
;    t17: v2i32 = extract_subvector t2, Constant:i32<4>
;    ...

; RUN: llc -mtriple=hexagon -mattr="hvx-length128b" < %s | FileCheck %s

; CHECK-LABEL: extract_subvec:
; CHECK: r29 = and(r29,#-128)
; CHECK: [[R1:r([0-9]+)]] = add(r29,#0)
; CHECK: vmem([[R1]]+#0) = v0
; CHECK-DAG: r[[R4:[0-9]+]] = memw([[R1]]+#0)
; CHECK-DAG: r[[R5:[0-9]+]] = memw([[R1]]+#4)
; CHECK-DAG: r[[R6:[0-9]+]] = memw([[R1]]+#8)
; CHECK-DAG: r[[R7:[0-9]+]] = memw([[R1]]+#12)
; CHECK-DAG: r[[R8:[0-9]+]] = memw([[R1]]+#16)
; CHECK-DAG: r[[R9:[0-9]+]] = memw([[R1]]+#20)
; CHECK-DAG: r[[R2:[0-9]+]] = memw([[R1]]+#24)
; CHECK-DAG: r[[R3:[0-9]+]] = memw([[R1]]+#28)
; CHECK-DAG: memd(r0+#0) = r[[R5]]:[[R4]]
; CHECK-DAG: memd(r0+#8) = r[[R7]]:[[R6]]
; CHECK-DAG: memd(r0+#16) = r[[R9]]:[[R8]]
; CHECK-DAG: memw(r0+#24) = r[[R2]]
define void @extract_subvec(<56 x i32> %val, ptr %buf) {
entry:
  %split = shufflevector <56 x i32> %val, <56 x i32> zeroinitializer, <7 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6>
  store <7 x i32> %split, ptr %buf, align 32
  ret void
}
