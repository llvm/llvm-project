; Test 32-bit additions of constants to memory.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i32 @foo()

; Check addition of 1.
define zeroext i1 @f1(ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: asi 0(%r2), 1
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: afi [[REG]], 1342177280
; CHECK: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %a = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 1)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %ptr
  ret i1 %obit
}

; Check the high end of the constant range.
define zeroext i1 @f2(ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: asi 0(%r2), 127
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: afi [[REG]], 1342177280
; CHECK: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %a = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 127)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %ptr
  ret i1 %obit
}

; Check the next constant up, which must use an addition and a store.
define zeroext i1 @f3(i32 %dummy, ptr %ptr) {
; CHECK-LABEL: f3:
; CHECK: l [[VAL:%r[0-5]]], 0(%r3)
; CHECK: ahi [[VAL]], 128
; CHECK-DAG: st [[VAL]], 0(%r3)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %a = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 128)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %ptr
  ret i1 %obit
}

; Check the low end of the constant range.
define zeroext i1 @f4(ptr %ptr) {
; CHECK-LABEL: f4:
; CHECK: asi 0(%r2), -128
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: afi [[REG]], 1342177280
; CHECK: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %a = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 -128)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %ptr
  ret i1 %obit
}

; Check the next value down, with the same comment as f3.
define zeroext i1 @f5(i32 %dummy, ptr %ptr) {
; CHECK-LABEL: f5:
; CHECK: l [[VAL:%r[0-5]]], 0(%r3)
; CHECK: ahi [[VAL]], -129
; CHECK-DAG: st [[VAL]], 0(%r3)
; CHECK-DAG: ipm [[REG:%r[0-5]]]
; CHECK-DAG: afi [[REG]], 1342177280
; CHECK-DAG: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %a = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 -129)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %ptr
  ret i1 %obit
}

; Check the high end of the aligned ASI range.
define zeroext i1 @f6(ptr %base) {
; CHECK-LABEL: f6:
; CHECK: asi 524284(%r2), 1
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: afi [[REG]], 1342177280
; CHECK: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 131071
  %a = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 1)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %ptr
  ret i1 %obit
}

; Check the next word up, which must use separate address logic.
; Other sequences besides this one would be OK.
define zeroext i1 @f7(ptr %base) {
; CHECK-LABEL: f7:
; CHECK: agfi %r2, 524288
; CHECK: asi 0(%r2), 1
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: afi [[REG]], 1342177280
; CHECK: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 131072
  %a = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 1)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %ptr
  ret i1 %obit
}

; Check the low end of the ASI range.
define zeroext i1 @f8(ptr %base) {
; CHECK-LABEL: f8:
; CHECK: asi -524288(%r2), 1
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: afi [[REG]], 1342177280
; CHECK: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 -131072
  %a = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 1)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %ptr
  ret i1 %obit
}

; Check the next word down, which must use separate address logic.
; Other sequences besides this one would be OK.
define zeroext i1 @f9(ptr %base) {
; CHECK-LABEL: f9:
; CHECK: agfi %r2, -524292
; CHECK: asi 0(%r2), 1
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: afi [[REG]], 1342177280
; CHECK: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %base, i64 -131073
  %a = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 1)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %ptr
  ret i1 %obit
}

; Check that ASI does not allow indices.
define zeroext i1 @f10(i64 %base, i64 %index) {
; CHECK-LABEL: f10:
; CHECK: agr %r2, %r3
; CHECK: asi 4(%r2), 1
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK: afi [[REG]], 1342177280
; CHECK: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 4
  %ptr = inttoptr i64 %add2 to ptr
  %a = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 1)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %ptr
  ret i1 %obit
}

; Check that adding 127 to a spilled value can use ASI.
define zeroext i1 @f11(ptr %ptr, i32 %sel) {
; CHECK-LABEL: f11:
; CHECK: asi {{[0-9]+}}(%r15), 127
; CHECK: br %r14
entry:
  %val0 = load volatile i32, ptr %ptr
  %val1 = load volatile i32, ptr %ptr
  %val2 = load volatile i32, ptr %ptr
  %val3 = load volatile i32, ptr %ptr
  %val4 = load volatile i32, ptr %ptr
  %val5 = load volatile i32, ptr %ptr
  %val6 = load volatile i32, ptr %ptr
  %val7 = load volatile i32, ptr %ptr
  %val8 = load volatile i32, ptr %ptr
  %val9 = load volatile i32, ptr %ptr
  %val10 = load volatile i32, ptr %ptr
  %val11 = load volatile i32, ptr %ptr
  %val12 = load volatile i32, ptr %ptr
  %val13 = load volatile i32, ptr %ptr
  %val14 = load volatile i32, ptr %ptr
  %val15 = load volatile i32, ptr %ptr

  %test = icmp ne i32 %sel, 0
  br i1 %test, label %add, label %store

add:
  %t0 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val0, i32 127)
  %add0 = extractvalue {i32, i1} %t0, 0
  %obit0 = extractvalue {i32, i1} %t0, 1
  %t1 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val1, i32 127)
  %add1 = extractvalue {i32, i1} %t1, 0
  %obit1 = extractvalue {i32, i1} %t1, 1
  %res1 = or i1 %obit0, %obit1
  %t2 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val2, i32 127)
  %add2 = extractvalue {i32, i1} %t2, 0
  %obit2 = extractvalue {i32, i1} %t2, 1
  %res2 = or i1 %res1, %obit2
  %t3 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val3, i32 127)
  %add3 = extractvalue {i32, i1} %t3, 0
  %obit3 = extractvalue {i32, i1} %t3, 1
  %res3 = or i1 %res2, %obit3
  %t4 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val4, i32 127)
  %add4 = extractvalue {i32, i1} %t4, 0
  %obit4 = extractvalue {i32, i1} %t4, 1
  %res4 = or i1 %res3, %obit4
  %t5 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val5, i32 127)
  %add5 = extractvalue {i32, i1} %t5, 0
  %obit5 = extractvalue {i32, i1} %t5, 1
  %res5 = or i1 %res4, %obit5
  %t6 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val6, i32 127)
  %add6 = extractvalue {i32, i1} %t6, 0
  %obit6 = extractvalue {i32, i1} %t6, 1
  %res6 = or i1 %res5, %obit6
  %t7 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val7, i32 127)
  %add7 = extractvalue {i32, i1} %t7, 0
  %obit7 = extractvalue {i32, i1} %t7, 1
  %res7 = or i1 %res6, %obit7
  %t8 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val8, i32 127)
  %add8 = extractvalue {i32, i1} %t8, 0
  %obit8 = extractvalue {i32, i1} %t8, 1
  %res8 = or i1 %res7, %obit8
  %t9 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val9, i32 127)
  %add9 = extractvalue {i32, i1} %t9, 0
  %obit9 = extractvalue {i32, i1} %t9, 1
  %res9 = or i1 %res8, %obit9
  %t10 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val10, i32 127)
  %add10 = extractvalue {i32, i1} %t10, 0
  %obit10 = extractvalue {i32, i1} %t10, 1
  %res10 = or i1 %res9, %obit10
  %t11 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val11, i32 127)
  %add11 = extractvalue {i32, i1} %t11, 0
  %obit11 = extractvalue {i32, i1} %t11, 1
  %res11 = or i1 %res10, %obit11
  %t12 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val12, i32 127)
  %add12 = extractvalue {i32, i1} %t12, 0
  %obit12 = extractvalue {i32, i1} %t12, 1
  %res12 = or i1 %res11, %obit12
  %t13 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val13, i32 127)
  %add13 = extractvalue {i32, i1} %t13, 0
  %obit13 = extractvalue {i32, i1} %t13, 1
  %res13 = or i1 %res12, %obit13
  %t14 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val14, i32 127)
  %add14 = extractvalue {i32, i1} %t14, 0
  %obit14 = extractvalue {i32, i1} %t14, 1
  %res14 = or i1 %res13, %obit14
  %t15 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val15, i32 127)
  %add15 = extractvalue {i32, i1} %t15, 0
  %obit15 = extractvalue {i32, i1} %t15, 1
  %res15 = or i1 %res14, %obit15

  br label %store

store:
  %new0 = phi i32 [ %val0, %entry ], [ %add0, %add ]
  %new1 = phi i32 [ %val1, %entry ], [ %add1, %add ]
  %new2 = phi i32 [ %val2, %entry ], [ %add2, %add ]
  %new3 = phi i32 [ %val3, %entry ], [ %add3, %add ]
  %new4 = phi i32 [ %val4, %entry ], [ %add4, %add ]
  %new5 = phi i32 [ %val5, %entry ], [ %add5, %add ]
  %new6 = phi i32 [ %val6, %entry ], [ %add6, %add ]
  %new7 = phi i32 [ %val7, %entry ], [ %add7, %add ]
  %new8 = phi i32 [ %val8, %entry ], [ %add8, %add ]
  %new9 = phi i32 [ %val9, %entry ], [ %add9, %add ]
  %new10 = phi i32 [ %val10, %entry ], [ %add10, %add ]
  %new11 = phi i32 [ %val11, %entry ], [ %add11, %add ]
  %new12 = phi i32 [ %val12, %entry ], [ %add12, %add ]
  %new13 = phi i32 [ %val13, %entry ], [ %add13, %add ]
  %new14 = phi i32 [ %val14, %entry ], [ %add14, %add ]
  %new15 = phi i32 [ %val15, %entry ], [ %add15, %add ]
  %res = phi i1 [ 0, %entry ], [ %res15, %add ]

  store volatile i32 %new0, ptr %ptr
  store volatile i32 %new1, ptr %ptr
  store volatile i32 %new2, ptr %ptr
  store volatile i32 %new3, ptr %ptr
  store volatile i32 %new4, ptr %ptr
  store volatile i32 %new5, ptr %ptr
  store volatile i32 %new6, ptr %ptr
  store volatile i32 %new7, ptr %ptr
  store volatile i32 %new8, ptr %ptr
  store volatile i32 %new9, ptr %ptr
  store volatile i32 %new10, ptr %ptr
  store volatile i32 %new11, ptr %ptr
  store volatile i32 %new12, ptr %ptr
  store volatile i32 %new13, ptr %ptr
  store volatile i32 %new14, ptr %ptr
  store volatile i32 %new15, ptr %ptr

  ret i1 %res
}

; Check that adding -128 to a spilled value can use ASI.
define zeroext i1 @f12(ptr %ptr, i32 %sel) {
; CHECK-LABEL: f12:
; CHECK: asi {{[0-9]+}}(%r15), -128
; CHECK: br %r14
entry:
  %val0 = load volatile i32, ptr %ptr
  %val1 = load volatile i32, ptr %ptr
  %val2 = load volatile i32, ptr %ptr
  %val3 = load volatile i32, ptr %ptr
  %val4 = load volatile i32, ptr %ptr
  %val5 = load volatile i32, ptr %ptr
  %val6 = load volatile i32, ptr %ptr
  %val7 = load volatile i32, ptr %ptr
  %val8 = load volatile i32, ptr %ptr
  %val9 = load volatile i32, ptr %ptr
  %val10 = load volatile i32, ptr %ptr
  %val11 = load volatile i32, ptr %ptr
  %val12 = load volatile i32, ptr %ptr
  %val13 = load volatile i32, ptr %ptr
  %val14 = load volatile i32, ptr %ptr
  %val15 = load volatile i32, ptr %ptr

  %test = icmp ne i32 %sel, 0
  br i1 %test, label %add, label %store

add:
  %t0 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val0, i32 -128)
  %add0 = extractvalue {i32, i1} %t0, 0
  %obit0 = extractvalue {i32, i1} %t0, 1
  %t1 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val1, i32 -128)
  %add1 = extractvalue {i32, i1} %t1, 0
  %obit1 = extractvalue {i32, i1} %t1, 1
  %res1 = or i1 %obit0, %obit1
  %t2 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val2, i32 -128)
  %add2 = extractvalue {i32, i1} %t2, 0
  %obit2 = extractvalue {i32, i1} %t2, 1
  %res2 = or i1 %res1, %obit2
  %t3 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val3, i32 -128)
  %add3 = extractvalue {i32, i1} %t3, 0
  %obit3 = extractvalue {i32, i1} %t3, 1
  %res3 = or i1 %res2, %obit3
  %t4 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val4, i32 -128)
  %add4 = extractvalue {i32, i1} %t4, 0
  %obit4 = extractvalue {i32, i1} %t4, 1
  %res4 = or i1 %res3, %obit4
  %t5 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val5, i32 -128)
  %add5 = extractvalue {i32, i1} %t5, 0
  %obit5 = extractvalue {i32, i1} %t5, 1
  %res5 = or i1 %res4, %obit5
  %t6 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val6, i32 -128)
  %add6 = extractvalue {i32, i1} %t6, 0
  %obit6 = extractvalue {i32, i1} %t6, 1
  %res6 = or i1 %res5, %obit6
  %t7 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val7, i32 -128)
  %add7 = extractvalue {i32, i1} %t7, 0
  %obit7 = extractvalue {i32, i1} %t7, 1
  %res7 = or i1 %res6, %obit7
  %t8 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val8, i32 -128)
  %add8 = extractvalue {i32, i1} %t8, 0
  %obit8 = extractvalue {i32, i1} %t8, 1
  %res8 = or i1 %res7, %obit8
  %t9 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val9, i32 -128)
  %add9 = extractvalue {i32, i1} %t9, 0
  %obit9 = extractvalue {i32, i1} %t9, 1
  %res9 = or i1 %res8, %obit9
  %t10 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val10, i32 -128)
  %add10 = extractvalue {i32, i1} %t10, 0
  %obit10 = extractvalue {i32, i1} %t10, 1
  %res10 = or i1 %res9, %obit10
  %t11 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val11, i32 -128)
  %add11 = extractvalue {i32, i1} %t11, 0
  %obit11 = extractvalue {i32, i1} %t11, 1
  %res11 = or i1 %res10, %obit11
  %t12 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val12, i32 -128)
  %add12 = extractvalue {i32, i1} %t12, 0
  %obit12 = extractvalue {i32, i1} %t12, 1
  %res12 = or i1 %res11, %obit12
  %t13 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val13, i32 -128)
  %add13 = extractvalue {i32, i1} %t13, 0
  %obit13 = extractvalue {i32, i1} %t13, 1
  %res13 = or i1 %res12, %obit13
  %t14 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val14, i32 -128)
  %add14 = extractvalue {i32, i1} %t14, 0
  %obit14 = extractvalue {i32, i1} %t14, 1
  %res14 = or i1 %res13, %obit14
  %t15 = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %val15, i32 -128)
  %add15 = extractvalue {i32, i1} %t15, 0
  %obit15 = extractvalue {i32, i1} %t15, 1
  %res15 = or i1 %res14, %obit15

  br label %store

store:
  %new0 = phi i32 [ %val0, %entry ], [ %add0, %add ]
  %new1 = phi i32 [ %val1, %entry ], [ %add1, %add ]
  %new2 = phi i32 [ %val2, %entry ], [ %add2, %add ]
  %new3 = phi i32 [ %val3, %entry ], [ %add3, %add ]
  %new4 = phi i32 [ %val4, %entry ], [ %add4, %add ]
  %new5 = phi i32 [ %val5, %entry ], [ %add5, %add ]
  %new6 = phi i32 [ %val6, %entry ], [ %add6, %add ]
  %new7 = phi i32 [ %val7, %entry ], [ %add7, %add ]
  %new8 = phi i32 [ %val8, %entry ], [ %add8, %add ]
  %new9 = phi i32 [ %val9, %entry ], [ %add9, %add ]
  %new10 = phi i32 [ %val10, %entry ], [ %add10, %add ]
  %new11 = phi i32 [ %val11, %entry ], [ %add11, %add ]
  %new12 = phi i32 [ %val12, %entry ], [ %add12, %add ]
  %new13 = phi i32 [ %val13, %entry ], [ %add13, %add ]
  %new14 = phi i32 [ %val14, %entry ], [ %add14, %add ]
  %new15 = phi i32 [ %val15, %entry ], [ %add15, %add ]
  %res = phi i1 [ 0, %entry ], [ %res15, %add ]

  store volatile i32 %new0, ptr %ptr
  store volatile i32 %new1, ptr %ptr
  store volatile i32 %new2, ptr %ptr
  store volatile i32 %new3, ptr %ptr
  store volatile i32 %new4, ptr %ptr
  store volatile i32 %new5, ptr %ptr
  store volatile i32 %new6, ptr %ptr
  store volatile i32 %new7, ptr %ptr
  store volatile i32 %new8, ptr %ptr
  store volatile i32 %new9, ptr %ptr
  store volatile i32 %new10, ptr %ptr
  store volatile i32 %new11, ptr %ptr
  store volatile i32 %new12, ptr %ptr
  store volatile i32 %new13, ptr %ptr
  store volatile i32 %new14, ptr %ptr
  store volatile i32 %new15, ptr %ptr

  ret i1 %res
}

; Check using the overflow result for a branch.
define void @f13(ptr %ptr) {
; CHECK-LABEL: f13:
; CHECK: asi 0(%r2), 1
; CHECK: jgo foo@PLT
; CHECK: br %r14
  %a = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 1)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %ptr
  br i1 %obit, label %call, label %exit

call:
  tail call i32 @foo()
  br label %exit

exit:
  ret void
}

; ... and the same with the inverted direction.
define void @f14(ptr %ptr) {
; CHECK-LABEL: f14:
; CHECK: asi 0(%r2), 1
; CHECK: jgno foo@PLT
; CHECK: br %r14
  %a = load i32, ptr %ptr
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %a, i32 1)
  %val = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  store i32 %val, ptr %ptr
  br i1 %obit, label %exit, label %call

call:
  tail call i32 @foo()
  br label %exit

exit:
  ret void
}

declare {i32, i1} @llvm.sadd.with.overflow.i32(i32, i32) nounwind readnone

