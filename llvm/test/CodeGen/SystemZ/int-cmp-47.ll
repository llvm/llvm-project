; Test the use of TEST UNDER MASK for 64-bit operations.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

@g = global i32 0

; Check the lowest useful TMLL value.
define void @f1(i64 %a) {
; CHECK-LABEL: f1:
; CHECK: tmll %r2, 1
; CHECK: ber %r14
; CHECK: br %r14
entry:
  %and = and i64 %a, 1
  %cmp = icmp eq i64 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check the high end of the TMLL range.
define void @f2(i64 %a) {
; CHECK-LABEL: f2:
; CHECK: tmll %r2, 65535
; CHECK: bner %r14
; CHECK: br %r14
entry:
  %and = and i64 %a, 65535
  %cmp = icmp ne i64 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check the lowest useful TMLH value, which is the next value up.
define void @f3(i64 %a) {
; CHECK-LABEL: f3:
; CHECK: tmlh %r2, 1
; CHECK: bner %r14
; CHECK: br %r14
entry:
  %and = and i64 %a, 65536
  %cmp = icmp ne i64 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check the next value up again, which cannot use TM.
define void @f4(i64 %a) {
; CHECK-LABEL: f4:
; CHECK-NOT: {{tm[lh].}}
; CHECK: br %r14
entry:
  %and = and i64 %a, 4294901759
  %cmp = icmp eq i64 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check the high end of the TMLH range.
define void @f5(i64 %a) {
; CHECK-LABEL: f5:
; CHECK: tmlh %r2, 65535
; CHECK: ber %r14
; CHECK: br %r14
entry:
  %and = and i64 %a, 4294901760
  %cmp = icmp eq i64 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check the lowest useful TMHL value.
define void @f6(i64 %a) {
; CHECK-LABEL: f6:
; CHECK: tmhl %r2, 1
; CHECK: ber %r14
; CHECK: br %r14
entry:
  %and = and i64 %a, 4294967296
  %cmp = icmp eq i64 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check the next value up again, which cannot use TM.
define void @f7(i64 %a) {
; CHECK-LABEL: f7:
; CHECK-NOT: {{tm[lh].}}
; CHECK: br %r14
entry:
  %and = and i64 %a, 4294967297
  %cmp = icmp ne i64 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check the high end of the TMHL range.
define void @f8(i64 %a) {
; CHECK-LABEL: f8:
; CHECK: tmhl %r2, 65535
; CHECK: bner %r14
; CHECK: br %r14
entry:
  %and = and i64 %a, 281470681743360
  %cmp = icmp ne i64 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check the lowest useful TMHH value.
define void @f9(i64 %a) {
; CHECK-LABEL: f9:
; CHECK: tmhh %r2, 1
; CHECK: bner %r14
; CHECK: br %r14
entry:
  %and = and i64 %a, 281474976710656
  %cmp = icmp ne i64 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check the high end of the TMHH range.
define void @f10(i64 %a) {
; CHECK-LABEL: f10:
; CHECK: tmhh %r2, 65535
; CHECK: ber %r14
; CHECK: br %r14
entry:
  %and = and i64 %a, 18446462598732840960
  %cmp = icmp eq i64 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check that we can fold an SHL into a TMxx mask.
define void @f11(i64 %a) {
; CHECK-LABEL: f11:
; CHECK: tmhl %r2, 32768
; CHECK: bner %r14
; CHECK: br %r14
entry:
  %shl = shl i64 %a, 1
  %and = and i64 %shl, 281474976710656
  %cmp = icmp ne i64 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check that we can fold an SHR into a TMxx mask.
define void @f12(i64 %a) {
; CHECK-LABEL: f12:
; CHECK: tmhh %r2, 256
; CHECK: bner %r14
; CHECK: br %r14
entry:
  %shr = lshr i64 %a, 56
  %and = and i64 %shr, 1
  %cmp = icmp ne i64 %and, 0
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check a case where TMHH can be used to implement a ult comparison.
define void @f13(i64 %a) {
; CHECK-LABEL: f13:
; CHECK: tmhh %r2, 49152
; CHECK: bnor %r14
; CHECK: br %r14
entry:
  %cmp = icmp ult i64 %a, 13835058055282163712
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; And again with ule.
define void @f14(i64 %a) {
; CHECK-LABEL: f14:
; CHECK: tmhh %r2, 49152
; CHECK: bnor %r14
; CHECK: br %r14
entry:
  %cmp = icmp ule i64 %a, 13835058055282163711
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; And again with ugt.
define void @f15(i64 %a) {
; CHECK-LABEL: f15:
; CHECK: tmhh %r2, 49152
; CHECK: bor %r14
; CHECK: br %r14
entry:
  %cmp = icmp ugt i64 %a, 13835058055282163711
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; And again with uge.
define void @f16(i64 %a) {
; CHECK-LABEL: f16:
; CHECK: tmhh %r2, 49152
; CHECK: bor %r14
; CHECK: br %r14
entry:
  %cmp = icmp uge i64 %a, 13835058055282163712
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Decrease the constant from f13 to make TMHH invalid.
define void @f17(i64 %a) {
; CHECK-LABEL: f17:
; CHECK-NOT: tmhh
; CHECK: srlg [[REG:%r[0-5]]], %r2, 48
; CHECK: cgfi [[REG]], 49151
; CHECK-NOT: tmhh
; CHECK: br %r14
entry:
  %cmp = icmp ult i64 %a, 13834776580305453056
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check that we don't use TMHH just to test the top bit.
define void @f18(i64 %a) {
; CHECK-LABEL: f18:
; CHECK-NOT: tmhh
; CHECK: cgibhe %r2, 0, 0(%r14)
; CHECK: br %r14
entry:
  %cmp = icmp ult i64 %a, 9223372036854775808
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

; Check that we don't fold a shift if the comparison value
; would need to be shifted out of range
define void @f19(i64 %a) {
; CHECK-LABEL: f19:
; CHECK-NOT: tmhh
; CHECK: srlg [[REG:%r[0-5]]], %r2, 63
; CHECK: cgibl [[REG]], 3, 0(%r14)
; CHECK: br %r14
entry:
  %shr = lshr i64 %a, 63
  %cmp = icmp ult i64 %shr, 3
  br i1 %cmp, label %exit, label %store

store:
  store i32 1, i32 *@g
  br label %exit

exit:
  ret void
}

