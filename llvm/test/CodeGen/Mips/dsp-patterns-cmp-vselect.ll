; RUN: llc -march=mips -mattr=dsp < %s | FileCheck %s

; CHECK-LABEL: select_v2q15_eq_:
; CHECK: cmp.eq.ph ${{[0-9]+}}, ${{[0-9]+}}
; CHECK: pick.ph ${{[0-9]+}}, $6, $7

define { i32 } @select_v2q15_eq_(i32 %a0.coerce, i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %1 = bitcast i32 %a1.coerce to <2 x i16>
  %2 = bitcast i32 %a2.coerce to <2 x i16>
  %3 = bitcast i32 %a3.coerce to <2 x i16>
  %cmp = icmp eq <2 x i16> %0, %1
  %or = select <2 x i1> %cmp, <2 x i16> %2, <2 x i16> %3
  %4 = bitcast <2 x i16> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: select_v2q15_lt_:
; CHECK: cmp.lt.ph $4, $5
; CHECK: pick.ph ${{[0-9]+}}, $6, $7

define { i32 } @select_v2q15_lt_(i32 %a0.coerce, i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %1 = bitcast i32 %a1.coerce to <2 x i16>
  %2 = bitcast i32 %a2.coerce to <2 x i16>
  %3 = bitcast i32 %a3.coerce to <2 x i16>
  %cmp = icmp slt <2 x i16> %0, %1
  %or = select <2 x i1> %cmp, <2 x i16> %2, <2 x i16> %3
  %4 = bitcast <2 x i16> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: select_v2q15_le_:
; CHECK: cmp.le.ph $4, $5
; CHECK: pick.ph ${{[0-9]+}}, $6, $7

define { i32 } @select_v2q15_le_(i32 %a0.coerce, i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %1 = bitcast i32 %a1.coerce to <2 x i16>
  %2 = bitcast i32 %a2.coerce to <2 x i16>
  %3 = bitcast i32 %a3.coerce to <2 x i16>
  %cmp = icmp sle <2 x i16> %0, %1
  %or = select <2 x i1> %cmp, <2 x i16> %2, <2 x i16> %3
  %4 = bitcast <2 x i16> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: select_v2q15_ne_:
; CHECK: cmp.eq.ph ${{[0-9]+}}, ${{[0-9]+}}
; CHECK: pick.ph ${{[0-9]+}}, $7, $6

define { i32 } @select_v2q15_ne_(i32 %a0.coerce, i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %1 = bitcast i32 %a1.coerce to <2 x i16>
  %2 = bitcast i32 %a2.coerce to <2 x i16>
  %3 = bitcast i32 %a3.coerce to <2 x i16>
  %cmp = icmp ne <2 x i16> %0, %1
  %or = select <2 x i1> %cmp, <2 x i16> %2, <2 x i16> %3
  %4 = bitcast <2 x i16> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: select_v2q15_gt_:
; CHECK: cmp.le.ph $4, $5
; CHECK: pick.ph ${{[0-9]+}}, $7, $6

define { i32 } @select_v2q15_gt_(i32 %a0.coerce, i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %1 = bitcast i32 %a1.coerce to <2 x i16>
  %2 = bitcast i32 %a2.coerce to <2 x i16>
  %3 = bitcast i32 %a3.coerce to <2 x i16>
  %cmp = icmp sgt <2 x i16> %0, %1
  %or = select <2 x i1> %cmp, <2 x i16> %2, <2 x i16> %3
  %4 = bitcast <2 x i16> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: select_v2q15_ge_:
; CHECK: cmp.lt.ph $4, $5
; CHECK: pick.ph ${{[0-9]+}}, $7, $6

define { i32 } @select_v2q15_ge_(i32 %a0.coerce, i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %1 = bitcast i32 %a1.coerce to <2 x i16>
  %2 = bitcast i32 %a2.coerce to <2 x i16>
  %3 = bitcast i32 %a3.coerce to <2 x i16>
  %cmp = icmp sge <2 x i16> %0, %1
  %or = select <2 x i1> %cmp, <2 x i16> %2, <2 x i16> %3
  %4 = bitcast <2 x i16> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: select_v4ui8_eq_:
; CHECK: cmpu.eq.qb ${{[0-9]+}}, ${{[0-9]+}}
; CHECK: pick.qb ${{[0-9]+}}, $6, $7

define { i32 } @select_v4ui8_eq_(i32 %a0.coerce, i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %1 = bitcast i32 %a1.coerce to <4 x i8>
  %2 = bitcast i32 %a2.coerce to <4 x i8>
  %3 = bitcast i32 %a3.coerce to <4 x i8>
  %cmp = icmp eq <4 x i8> %0, %1
  %or = select <4 x i1> %cmp, <4 x i8> %2, <4 x i8> %3
  %4 = bitcast <4 x i8> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: select_v4ui8_lt_:
; CHECK: cmpu.lt.qb $4, $5
; CHECK: pick.qb ${{[0-9]+}}, $6, $7

define { i32 } @select_v4ui8_lt_(i32 %a0.coerce, i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %1 = bitcast i32 %a1.coerce to <4 x i8>
  %2 = bitcast i32 %a2.coerce to <4 x i8>
  %3 = bitcast i32 %a3.coerce to <4 x i8>
  %cmp = icmp ult <4 x i8> %0, %1
  %or = select <4 x i1> %cmp, <4 x i8> %2, <4 x i8> %3
  %4 = bitcast <4 x i8> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: select_v4ui8_le_:
; CHECK: cmpu.le.qb $4, $5
; CHECK: pick.qb ${{[0-9]+}}, $6, $7

define { i32 } @select_v4ui8_le_(i32 %a0.coerce, i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %1 = bitcast i32 %a1.coerce to <4 x i8>
  %2 = bitcast i32 %a2.coerce to <4 x i8>
  %3 = bitcast i32 %a3.coerce to <4 x i8>
  %cmp = icmp ule <4 x i8> %0, %1
  %or = select <4 x i1> %cmp, <4 x i8> %2, <4 x i8> %3
  %4 = bitcast <4 x i8> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: select_v4ui8_ne_:
; CHECK: cmpu.eq.qb ${{[0-9]+}}, ${{[0-9]+}}
; CHECK: pick.qb ${{[0-9]+}}, $7, $6

define { i32 } @select_v4ui8_ne_(i32 %a0.coerce, i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %1 = bitcast i32 %a1.coerce to <4 x i8>
  %2 = bitcast i32 %a2.coerce to <4 x i8>
  %3 = bitcast i32 %a3.coerce to <4 x i8>
  %cmp = icmp ne <4 x i8> %0, %1
  %or = select <4 x i1> %cmp, <4 x i8> %2, <4 x i8> %3
  %4 = bitcast <4 x i8> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: select_v4ui8_gt_:
; CHECK: cmpu.le.qb $4, $5
; CHECK: pick.qb ${{[0-9]+}}, $7, $6

define { i32 } @select_v4ui8_gt_(i32 %a0.coerce, i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %1 = bitcast i32 %a1.coerce to <4 x i8>
  %2 = bitcast i32 %a2.coerce to <4 x i8>
  %3 = bitcast i32 %a3.coerce to <4 x i8>
  %cmp = icmp ugt <4 x i8> %0, %1
  %or = select <4 x i1> %cmp, <4 x i8> %2, <4 x i8> %3
  %4 = bitcast <4 x i8> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: select_v4ui8_ge_:
; CHECK: cmpu.lt.qb $4, $5
; CHECK: pick.qb ${{[0-9]+}}, $7, $6

define { i32 } @select_v4ui8_ge_(i32 %a0.coerce, i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %1 = bitcast i32 %a1.coerce to <4 x i8>
  %2 = bitcast i32 %a2.coerce to <4 x i8>
  %3 = bitcast i32 %a3.coerce to <4 x i8>
  %cmp = icmp uge <4 x i8> %0, %1
  %or = select <4 x i1> %cmp, <4 x i8> %2, <4 x i8> %3
  %4 = bitcast <4 x i8> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: select_v2ui16_lt_:
; CHECK-NOT: cmp
; CHECK-NOT: pick

define { i32 } @select_v2ui16_lt_(i32 %a0.coerce, i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %1 = bitcast i32 %a1.coerce to <2 x i16>
  %2 = bitcast i32 %a2.coerce to <2 x i16>
  %3 = bitcast i32 %a3.coerce to <2 x i16>
  %cmp = icmp ult <2 x i16> %0, %1
  %or = select <2 x i1> %cmp, <2 x i16> %2, <2 x i16> %3
  %4 = bitcast <2 x i16> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: select_v2ui16_le_:
; CHECK-NOT: cmp
; CHECK-NOT: pick

define { i32 } @select_v2ui16_le_(i32 %a0.coerce, i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %1 = bitcast i32 %a1.coerce to <2 x i16>
  %2 = bitcast i32 %a2.coerce to <2 x i16>
  %3 = bitcast i32 %a3.coerce to <2 x i16>
  %cmp = icmp ule <2 x i16> %0, %1
  %or = select <2 x i1> %cmp, <2 x i16> %2, <2 x i16> %3
  %4 = bitcast <2 x i16> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: select_v2ui16_gt_:
; CHECK-NOT: cmp
; CHECK-NOT: pick

define { i32 } @select_v2ui16_gt_(i32 %a0.coerce, i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %1 = bitcast i32 %a1.coerce to <2 x i16>
  %2 = bitcast i32 %a2.coerce to <2 x i16>
  %3 = bitcast i32 %a3.coerce to <2 x i16>
  %cmp = icmp ugt <2 x i16> %0, %1
  %or = select <2 x i1> %cmp, <2 x i16> %2, <2 x i16> %3
  %4 = bitcast <2 x i16> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: select_v2ui16_ge_:
; CHECK-NOT: cmp
; CHECK-NOT: pick

define { i32 } @select_v2ui16_ge_(i32 %a0.coerce, i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %1 = bitcast i32 %a1.coerce to <2 x i16>
  %2 = bitcast i32 %a2.coerce to <2 x i16>
  %3 = bitcast i32 %a3.coerce to <2 x i16>
  %cmp = icmp uge <2 x i16> %0, %1
  %or = select <2 x i1> %cmp, <2 x i16> %2, <2 x i16> %3
  %4 = bitcast <2 x i16> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: select_v4i8_lt_:
; CHECK-NOT: cmp
; CHECK-NOT: pick

define { i32 } @select_v4i8_lt_(i32 %a0.coerce, i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %1 = bitcast i32 %a1.coerce to <4 x i8>
  %2 = bitcast i32 %a2.coerce to <4 x i8>
  %3 = bitcast i32 %a3.coerce to <4 x i8>
  %cmp = icmp slt <4 x i8> %0, %1
  %or = select <4 x i1> %cmp, <4 x i8> %2, <4 x i8> %3
  %4 = bitcast <4 x i8> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: select_v4i8_le_:
; CHECK-NOT: cmp
; CHECK-NOT: pick

define { i32 } @select_v4i8_le_(i32 %a0.coerce, i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %1 = bitcast i32 %a1.coerce to <4 x i8>
  %2 = bitcast i32 %a2.coerce to <4 x i8>
  %3 = bitcast i32 %a3.coerce to <4 x i8>
  %cmp = icmp sle <4 x i8> %0, %1
  %or = select <4 x i1> %cmp, <4 x i8> %2, <4 x i8> %3
  %4 = bitcast <4 x i8> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: select_v4i8_gt_:
; CHECK-NOT: cmp
; CHECK-NOT: pick

define { i32 } @select_v4i8_gt_(i32 %a0.coerce, i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %1 = bitcast i32 %a1.coerce to <4 x i8>
  %2 = bitcast i32 %a2.coerce to <4 x i8>
  %3 = bitcast i32 %a3.coerce to <4 x i8>
  %cmp = icmp sgt <4 x i8> %0, %1
  %or = select <4 x i1> %cmp, <4 x i8> %2, <4 x i8> %3
  %4 = bitcast <4 x i8> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: select_v4i8_ge_:
; CHECK-NOT: cmp
; CHECK-NOT: pick

define { i32 } @select_v4i8_ge_(i32 %a0.coerce, i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %1 = bitcast i32 %a1.coerce to <4 x i8>
  %2 = bitcast i32 %a2.coerce to <4 x i8>
  %3 = bitcast i32 %a3.coerce to <4 x i8>
  %cmp = icmp sge <4 x i8> %0, %1
  %or = select <4 x i1> %cmp, <4 x i8> %2, <4 x i8> %3
  %4 = bitcast <4 x i8> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: compare_v2q15_eq_:
; CHECK: cmp.eq.ph ${{[0-9]+}}, ${{[0-9]+}}
; CHECK: pick.ph ${{[0-9]+}}, ${{[a-z0-9]+}}, ${{[a-z0-9]+}}

define { i32 } @compare_v2q15_eq_(i32 %a0.coerce, i32 %a1.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %1 = bitcast i32 %a1.coerce to <2 x i16>
  %cmp = icmp eq <2 x i16> %0, %1
  %sext = sext <2 x i1> %cmp to <2 x i16>
  %2 = bitcast <2 x i16> %sext to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: compare_v2q15_lt_:
; CHECK: cmp.lt.ph $4, $5
; CHECK: pick.ph ${{[0-9]+}}, ${{[a-z0-9]+}}, ${{[a-z0-9]+}}

define { i32 } @compare_v2q15_lt_(i32 %a0.coerce, i32 %a1.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %1 = bitcast i32 %a1.coerce to <2 x i16>
  %cmp = icmp slt <2 x i16> %0, %1
  %sext = sext <2 x i1> %cmp to <2 x i16>
  %2 = bitcast <2 x i16> %sext to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: compare_v2q15_le_:
; CHECK: cmp.le.ph $4, $5
; CHECK: pick.ph ${{[0-9]+}}, ${{[a-z0-9]+}}, ${{[a-z0-9]+}}

define { i32 } @compare_v2q15_le_(i32 %a0.coerce, i32 %a1.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %1 = bitcast i32 %a1.coerce to <2 x i16>
  %cmp = icmp sle <2 x i16> %0, %1
  %sext = sext <2 x i1> %cmp to <2 x i16>
  %2 = bitcast <2 x i16> %sext to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: compare_v2q15_ne_:
; CHECK: cmp.eq.ph ${{[0-9]+}}, ${{[0-9]+}}
; CHECK: pick.ph ${{[0-9]+}}, ${{[a-z0-9]+}}, ${{[a-z0-9]+}}

define { i32 } @compare_v2q15_ne_(i32 %a0.coerce, i32 %a1.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %1 = bitcast i32 %a1.coerce to <2 x i16>
  %cmp = icmp ne <2 x i16> %0, %1
  %sext = sext <2 x i1> %cmp to <2 x i16>
  %2 = bitcast <2 x i16> %sext to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: compare_v2q15_gt_:
; CHECK: cmp.le.ph $4, $5
; CHECK: pick.ph ${{[0-9]+}}, ${{[a-z0-9]+}}, ${{[a-z0-9]+}}

define { i32 } @compare_v2q15_gt_(i32 %a0.coerce, i32 %a1.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %1 = bitcast i32 %a1.coerce to <2 x i16>
  %cmp = icmp sgt <2 x i16> %0, %1
  %sext = sext <2 x i1> %cmp to <2 x i16>
  %2 = bitcast <2 x i16> %sext to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: compare_v2q15_ge_:
; CHECK: cmp.lt.ph $4, $5
; CHECK: pick.ph ${{[0-9]+}}, ${{[a-z0-9]+}}, ${{[a-z0-9]+}}

define { i32 } @compare_v2q15_ge_(i32 %a0.coerce, i32 %a1.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %1 = bitcast i32 %a1.coerce to <2 x i16>
  %cmp = icmp sge <2 x i16> %0, %1
  %sext = sext <2 x i1> %cmp to <2 x i16>
  %2 = bitcast <2 x i16> %sext to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: compare_v4ui8_eq_:
; CHECK: cmpu.eq.qb ${{[0-9]+}}, ${{[0-9]+}}
; CHECK: pick.qb ${{[0-9]+}}, ${{[a-z0-9]+}}, ${{[a-z0-9]+}}

define { i32 } @compare_v4ui8_eq_(i32 %a0.coerce, i32 %a1.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %1 = bitcast i32 %a1.coerce to <4 x i8>
  %cmp = icmp eq <4 x i8> %0, %1
  %sext = sext <4 x i1> %cmp to <4 x i8>
  %2 = bitcast <4 x i8> %sext to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: compare_v4ui8_lt_:
; CHECK: cmpu.lt.qb $4, $5
; CHECK: pick.qb ${{[0-9]+}}, ${{[a-z0-9]+}}, ${{[a-z0-9]+}}

define { i32 } @compare_v4ui8_lt_(i32 %a0.coerce, i32 %a1.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %1 = bitcast i32 %a1.coerce to <4 x i8>
  %cmp = icmp ult <4 x i8> %0, %1
  %sext = sext <4 x i1> %cmp to <4 x i8>
  %2 = bitcast <4 x i8> %sext to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: compare_v4ui8_le_:
; CHECK: cmpu.le.qb $4, $5
; CHECK: pick.qb ${{[0-9]+}}, ${{[a-z0-9]+}}, ${{[a-z0-9]+}}

define { i32 } @compare_v4ui8_le_(i32 %a0.coerce, i32 %a1.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %1 = bitcast i32 %a1.coerce to <4 x i8>
  %cmp = icmp ule <4 x i8> %0, %1
  %sext = sext <4 x i1> %cmp to <4 x i8>
  %2 = bitcast <4 x i8> %sext to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: compare_v4ui8_ne_:
; CHECK: cmpu.eq.qb ${{[0-9]+}}, ${{[0-9]+}}
; CHECK: pick.qb ${{[0-9]+}}, ${{[a-z0-9]+}}, ${{[a-z0-9]+}}

define { i32 } @compare_v4ui8_ne_(i32 %a0.coerce, i32 %a1.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %1 = bitcast i32 %a1.coerce to <4 x i8>
  %cmp = icmp ne <4 x i8> %0, %1
  %sext = sext <4 x i1> %cmp to <4 x i8>
  %2 = bitcast <4 x i8> %sext to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: compare_v4ui8_gt_:
; CHECK: cmpu.le.qb $4, $5
; CHECK: pick.qb ${{[0-9]+}}, ${{[a-z0-9]+}}, ${{[a-z0-9]+}}

define { i32 } @compare_v4ui8_gt_(i32 %a0.coerce, i32 %a1.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %1 = bitcast i32 %a1.coerce to <4 x i8>
  %cmp = icmp ugt <4 x i8> %0, %1
  %sext = sext <4 x i1> %cmp to <4 x i8>
  %2 = bitcast <4 x i8> %sext to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: compare_v4ui8_ge_:
; CHECK: cmpu.lt.qb $4, $5
; CHECK: pick.qb ${{[0-9]+}}, ${{[a-z0-9]+}}, ${{[a-z0-9]+}}

define { i32 } @compare_v4ui8_ge_(i32 %a0.coerce, i32 %a1.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %1 = bitcast i32 %a1.coerce to <4 x i8>
  %cmp = icmp uge <4 x i8> %0, %1
  %sext = sext <4 x i1> %cmp to <4 x i8>
  %2 = bitcast <4 x i8> %sext to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: compare_v2ui16_lt_:
; CHECK-NOT: cmp
; CHECK-NOT: pick

define { i32 } @compare_v2ui16_lt_(i32 %a0.coerce, i32 %a1.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %1 = bitcast i32 %a1.coerce to <2 x i16>
  %cmp = icmp ult <2 x i16> %0, %1
  %sext = sext <2 x i1> %cmp to <2 x i16>
  %2 = bitcast <2 x i16> %sext to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: compare_v2ui16_le_:
; CHECK-NOT: cmp
; CHECK-NOT: pick

define { i32 } @compare_v2ui16_le_(i32 %a0.coerce, i32 %a1.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %1 = bitcast i32 %a1.coerce to <2 x i16>
  %cmp = icmp ule <2 x i16> %0, %1
  %sext = sext <2 x i1> %cmp to <2 x i16>
  %2 = bitcast <2 x i16> %sext to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: compare_v2ui16_gt_:
; CHECK-NOT: cmp
; CHECK-NOT: pick

define { i32 } @compare_v2ui16_gt_(i32 %a0.coerce, i32 %a1.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %1 = bitcast i32 %a1.coerce to <2 x i16>
  %cmp = icmp ugt <2 x i16> %0, %1
  %sext = sext <2 x i1> %cmp to <2 x i16>
  %2 = bitcast <2 x i16> %sext to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: compare_v2ui16_ge_:
; CHECK-NOT: cmp
; CHECK-NOT: pick

define { i32 } @compare_v2ui16_ge_(i32 %a0.coerce, i32 %a1.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <2 x i16>
  %1 = bitcast i32 %a1.coerce to <2 x i16>
  %cmp = icmp uge <2 x i16> %0, %1
  %sext = sext <2 x i1> %cmp to <2 x i16>
  %2 = bitcast <2 x i16> %sext to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: compare_v4i8_lt_:
; CHECK-NOT: cmp
; CHECK-NOT: pick

define { i32 } @compare_v4i8_lt_(i32 %a0.coerce, i32 %a1.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %1 = bitcast i32 %a1.coerce to <4 x i8>
  %cmp = icmp slt <4 x i8> %0, %1
  %sext = sext <4 x i1> %cmp to <4 x i8>
  %2 = bitcast <4 x i8> %sext to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: compare_v4i8_le_:
; CHECK-NOT: cmp
; CHECK-NOT: pick

define { i32 } @compare_v4i8_le_(i32 %a0.coerce, i32 %a1.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %1 = bitcast i32 %a1.coerce to <4 x i8>
  %cmp = icmp sle <4 x i8> %0, %1
  %sext = sext <4 x i1> %cmp to <4 x i8>
  %2 = bitcast <4 x i8> %sext to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: compare_v4i8_gt_:
; CHECK-NOT: cmp
; CHECK-NOT: pick

define { i32 } @compare_v4i8_gt_(i32 %a0.coerce, i32 %a1.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %1 = bitcast i32 %a1.coerce to <4 x i8>
  %cmp = icmp sgt <4 x i8> %0, %1
  %sext = sext <4 x i1> %cmp to <4 x i8>
  %2 = bitcast <4 x i8> %sext to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}

; CHECK-LABEL: compare_v4i8_ge_:
; CHECK-NOT: cmp
; CHECK-NOT: pick

define { i32 } @compare_v4i8_ge_(i32 %a0.coerce, i32 %a1.coerce) {
entry:
  %0 = bitcast i32 %a0.coerce to <4 x i8>
  %1 = bitcast i32 %a1.coerce to <4 x i8>
  %cmp = icmp sge <4 x i8> %0, %1
  %sext = sext <4 x i1> %cmp to <4 x i8>
  %2 = bitcast <4 x i8> %sext to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %2, 0
  ret { i32 } %.fca.0.insert
}
