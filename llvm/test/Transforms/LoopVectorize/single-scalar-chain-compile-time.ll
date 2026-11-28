; RUN: opt -passes=loop-vectorize -force-vector-width=4 -S %s | FileCheck %s

; Each %x.N below is used three times by the next step, so a naive recursive walk
; over operands visits the start of the chain an exponential number of times.
; Check that vputils::isSingleScalar handles this in reasonable time.

define void @uniform_chain(ptr %p, i32 %a, i64 %n) {
; CHECK-LABEL: define void @uniform_chain(
; CHECK:       vector.body:
; CHECK:         store <4 x i32>
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %x.0 = add i32 %a, 1
  %s.0 = lshr i32 %x.0, 3
  %t.0 = xor i32 %s.0, %x.0
  %x.1 = sub i32 %x.0, %t.0
  %s.1 = lshr i32 %x.1, 3
  %t.1 = xor i32 %s.1, %x.1
  %x.2 = sub i32 %x.1, %t.1
  %s.2 = lshr i32 %x.2, 3
  %t.2 = xor i32 %s.2, %x.2
  %x.3 = sub i32 %x.2, %t.2
  %s.3 = lshr i32 %x.3, 3
  %t.3 = xor i32 %s.3, %x.3
  %x.4 = sub i32 %x.3, %t.3
  %s.4 = lshr i32 %x.4, 3
  %t.4 = xor i32 %s.4, %x.4
  %x.5 = sub i32 %x.4, %t.4
  %s.5 = lshr i32 %x.5, 3
  %t.5 = xor i32 %s.5, %x.5
  %x.6 = sub i32 %x.5, %t.5
  %s.6 = lshr i32 %x.6, 3
  %t.6 = xor i32 %s.6, %x.6
  %x.7 = sub i32 %x.6, %t.6
  %s.7 = lshr i32 %x.7, 3
  %t.7 = xor i32 %s.7, %x.7
  %x.8 = sub i32 %x.7, %t.7
  %s.8 = lshr i32 %x.8, 3
  %t.8 = xor i32 %s.8, %x.8
  %x.9 = sub i32 %x.8, %t.8
  %s.9 = lshr i32 %x.9, 3
  %t.9 = xor i32 %s.9, %x.9
  %x.10 = sub i32 %x.9, %t.9
  %s.10 = lshr i32 %x.10, 3
  %t.10 = xor i32 %s.10, %x.10
  %x.11 = sub i32 %x.10, %t.10
  %s.11 = lshr i32 %x.11, 3
  %t.11 = xor i32 %s.11, %x.11
  %x.12 = sub i32 %x.11, %t.11
  %s.12 = lshr i32 %x.12, 3
  %t.12 = xor i32 %s.12, %x.12
  %x.13 = sub i32 %x.12, %t.12
  %s.13 = lshr i32 %x.13, 3
  %t.13 = xor i32 %s.13, %x.13
  %x.14 = sub i32 %x.13, %t.13
  %s.14 = lshr i32 %x.14, 3
  %t.14 = xor i32 %s.14, %x.14
  %x.15 = sub i32 %x.14, %t.14
  %s.15 = lshr i32 %x.15, 3
  %t.15 = xor i32 %s.15, %x.15
  %x.16 = sub i32 %x.15, %t.15
  %s.16 = lshr i32 %x.16, 3
  %t.16 = xor i32 %s.16, %x.16
  %x.17 = sub i32 %x.16, %t.16
  %s.17 = lshr i32 %x.17, 3
  %t.17 = xor i32 %s.17, %x.17
  %x.18 = sub i32 %x.17, %t.17
  %s.18 = lshr i32 %x.18, 3
  %t.18 = xor i32 %s.18, %x.18
  %x.19 = sub i32 %x.18, %t.18
  %s.19 = lshr i32 %x.19, 3
  %t.19 = xor i32 %s.19, %x.19
  %x.20 = sub i32 %x.19, %t.19
  %s.20 = lshr i32 %x.20, 3
  %t.20 = xor i32 %s.20, %x.20
  %x.21 = sub i32 %x.20, %t.20
  %s.21 = lshr i32 %x.21, 3
  %t.21 = xor i32 %s.21, %x.21
  %x.22 = sub i32 %x.21, %t.21
  %s.22 = lshr i32 %x.22, 3
  %t.22 = xor i32 %s.22, %x.22
  %x.23 = sub i32 %x.22, %t.22
  %s.23 = lshr i32 %x.23, 3
  %t.23 = xor i32 %s.23, %x.23
  %x.24 = sub i32 %x.23, %t.23
  %gep = getelementptr inbounds i32, ptr %p, i64 %iv
  store i32 %x.24, ptr %gep, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
