; RUN: llc -verify-machineinstrs -O0 -mcpu=ppc64 -mtriple=powerpc64-unknown-linux-gnu -fast-isel=false < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mcpu=g4 -mtriple=powerpc-apple-darwin8 < %s | FileCheck -check-prefix=DARWIN32 %s
; RUN: llc -verify-machineinstrs -O0 -mcpu=ppc970 -mtriple=powerpc64-apple-darwin8 < %s | FileCheck -check-prefix=DARWIN64 %s

; Test case for PR 14779: anonymous aggregates are not handled correctly.
; Darwin bug report PR 15821 is similar.
; The bug is triggered by passing a byval structure after an anonymous
; aggregate.

%tarray = type { i64, i8* }

define i8* @func1({ i64, i8* } %array, i8* %ptr) {
entry:
  %array_ptr = extractvalue {i64, i8* } %array, 1
  %cond = icmp eq i8* %array_ptr, %ptr
  br i1 %cond, label %equal, label %unequal
equal:
  ret i8* %array_ptr
unequal:
  ret i8* %ptr
}

; CHECK-LABEL: func1:
; CHECK: cmpld {{([0-9]+,)?}}4, 5
; CHECK-DAG: std 4, -[[OFFSET1:[0-9]+]]
; CHECK-DAG: std 5, -[[OFFSET2:[0-9]+]]
; CHECK: ld 3, -[[OFFSET1]](1)
; CHECK: ld 3, -[[OFFSET2]](1)

; DARWIN32: _func1:
; DARWIN32: mr
; DARWIN32: mr r[[REG1:[0-9]+]], r[[REGA:[0-9]+]]
; DARWIN32: mr r[[REG2:[0-9]+]], r[[REGB:[0-9]+]]
; DARWIN32: cmplw {{(cr[0-9]+,)?}}r[[REGA]], r[[REGB]]
; DARWIN32: stw r[[REG1]], -[[OFFSET1:[0-9]+]]
; DARWIN32: stw r[[REG2]], -[[OFFSET2:[0-9]+]]
; DARWIN32: lwz r3, -[[OFFSET1]]
; DARWIN32: lwz r3, -[[OFFSET2]]

; DARWIN64: _func1:
; DARWIN64: mr
; DARWIN64: mr r[[REG1:[0-9]+]], r[[REGA:[0-9]+]]
; DARWIN64: mr r[[REG2:[0-9]+]], r[[REGB:[0-9]+]]
; DARWIN64: cmpld {{(cr[0-9]+,)?}}r[[REGA]], r[[REGB]]
; DARWIN64: std r[[REG1]], -[[OFFSET1:[0-9]+]]
; DARWIN64: std r[[REG2]], -[[OFFSET2:[0-9]+]]
; DARWIN64: ld r3, -[[OFFSET1]]
; DARWIN64: ld r3, -[[OFFSET2]]


define i8* @func2({ i64, i8* } %array1, %tarray* byval %array2) {
entry:
  %array1_ptr = extractvalue {i64, i8* } %array1, 1
  %tmp = getelementptr inbounds %tarray, %tarray* %array2, i32 0, i32 1
  %array2_ptr = load i8*, i8** %tmp
  %cond = icmp eq i8* %array1_ptr, %array2_ptr
  br i1 %cond, label %equal, label %unequal
equal:
  ret i8* %array1_ptr
unequal:
  ret i8* %array2_ptr
}
; CHECK-LABEL: func2:
; CHECK: cmpld {{([0-9]+,)?}}4, 6
; CHECK-DAG: std 6, 72(1)
; CHECK-DAG: std 5, 64(1)
; CHECK-DAG: std 6, -[[OFFSET1:[0-9]+]]
; CHECK-DAG: std 4, -[[OFFSET2:[0-9]+]]
; CHECK: ld 3, -[[OFFSET2]](1)
; CHECK: ld 3, -[[OFFSET1]](1)

; DARWIN32-LABEL: _func2
; DARWIN32-DAG: addi r[[REG8:[0-9]+]], r[[REGSP:[0-9]+]], 36
; DARWIN32-DAG: lwz r[[REG2:[0-9]+]], 44(r[[REGSP]])
; DARWIN32: mr
; DARWIN32: mr r[[REG7:[0-9]+]], r5
; DARWIN32-DAG: cmplw {{(cr[0-9]+,)?}}r5, r[[REG2]]
; DARWIN32-DAG: stw r[[REG7]], -[[OFFSET1:[0-9]+]]
; DARWIN32-DAG: stw r[[REG2]], -[[OFFSET2:[0-9]+]]
; DARWIN32-DAG: lwz r3, -[[OFFSET1]]
; DARWIN32-DAG: lwz r3, -[[OFFSET2]]


; DARWIN64: _func2:
; DARWIN64: ld r[[REG2:[0-9]+]], 72(r1)
; DARWIN64: mr
; DARWIN64: mr r[[REG3:[0-9]+]], r[[REGA:[0-9]+]]
; DARWIN64: cmpld {{(cr[0-9]+,)?}}r[[REGA]], r[[REG2]]
; DARWIN64: std r[[REG2]], -[[OFFSET2:[0-9]+]]
; DARWIN64: std r[[REG3]], -[[OFFSET1:[0-9]+]]
; DARWIN64: ld r3, -[[OFFSET1]]
; DARWIN64: ld r3, -[[OFFSET2]]


define i8* @func3({ i64, i8* }* byval %array1, %tarray* byval %array2) {
entry:
  %tmp1 = getelementptr inbounds { i64, i8* }, { i64, i8* }* %array1, i32 0, i32 1
  %array1_ptr = load i8*, i8** %tmp1
  %tmp2 = getelementptr inbounds %tarray, %tarray* %array2, i32 0, i32 1
  %array2_ptr = load i8*, i8** %tmp2
  %cond = icmp eq i8* %array1_ptr, %array2_ptr
  br i1 %cond, label %equal, label %unequal
equal:
  ret i8* %array1_ptr
unequal:
  ret i8* %array2_ptr
}

; CHECK-LABEL: func3:
; CHECK: cmpld {{([0-9]+,)?}}4, 6
; CHECK-DAG: std 4, -[[OFFSET2:[0-9]+]](1)
; CHECK-DAG: std 6, -[[OFFSET1:[0-9]+]](1)
; CHECK: ld 3, -[[OFFSET2]](1)
; CHECK: ld 3, -[[OFFSET1]](1)

; DARWIN32-LABEL: _func3:
; DARWIN32-DAG: stw r[[REG8:[0-9]+]], 44(r[[REGSP:[0-9]+]])
; DARWIN32-DAG: stw r[[REG5:[0-9]+]], 32(r[[REGSP]])
; DARWIN32-DAG: addi r[[REG5a:[0-9]+]], r[[REGSP:[0-9]+]], 36
; DARWIN32-DAG: addi r[[REG8a:[0-9]+]], r[[REGSP]], 24
; DARWIN32-DAG: lwz r[[REG5a:[0-9]+]], 44(r[[REGSP]])
; DARWIN32-DAG: lwz r[[REG8a:[0-9]+]], 32(r[[REGSP]])
; DARWIN32-DAG: cmplw {{(cr[0-9]+,)?}}r[[REG8a]], r[[REG5a]]
; DARWIN32-DAG: stw r[[REG5a]], -[[OFFSET1:[0-9]+]]
; DARWIN32-DAG: stw r[[REG8a]], -[[OFFSET2:[0-9]+]]
; DARWIN32-DAG: lwz r3, -[[OFFSET1:[0-9]+]]
; DARWIN32-DAG: lwz r3, -[[OFFSET2:[0-9]+]]

; DARWIN64: _func3:
; DARWIN64: ld r[[REG3:[0-9]+]], 72(r1)
; DARWIN64: ld r[[REG4:[0-9]+]], 56(r1)
; DARWIN64: cmpld {{(cr[0-9]+,)?}}r[[REG4]], r[[REG3]]
; DARWIN64: std r[[REG3]], -[[OFFSET1:[0-9]+]]
; DARWIN64: std r[[REG4]], -[[OFFSET2:[0-9]+]]
; DARWIN64: ld r3, -[[OFFSET2]]
; DARWIN64: ld r3, -[[OFFSET1]]


define i8* @func4(i64 %p1, i64 %p2, i64 %p3, i64 %p4,
                  i64 %p5, i64 %p6, i64 %p7, i64 %p8,
                  { i64, i8* } %array1, %tarray* byval %array2) {
entry:
  %array1_ptr = extractvalue {i64, i8* } %array1, 1
  %tmp = getelementptr inbounds %tarray, %tarray* %array2, i32 0, i32 1
  %array2_ptr = load i8*, i8** %tmp
  %cond = icmp eq i8* %array1_ptr, %array2_ptr
  br i1 %cond, label %equal, label %unequal
equal:
  ret i8* %array1_ptr
unequal:
  ret i8* %array2_ptr
}

; CHECK-LABEL: func4:
; CHECK: ld [[REG3:[0-9]+]], 136(1)
; CHECK: ld [[REG2:[0-9]+]], 120(1)
; CHECK: cmpld {{([0-9]+,)?}}[[REG2]], [[REG3]]
; CHECK: std [[REG3]], -[[OFFSET2:[0-9]+]](1)
; CHECK: std [[REG2]], -[[OFFSET1:[0-9]+]](1)
; CHECK: ld 3, -[[OFFSET1]](1)
; CHECK: ld 3, -[[OFFSET2]](1)

; DARWIN32: _func4:
; DARWIN32: lwz r[[REG4:[0-9]+]], 96(r1)
; DARWIN32: addi r[[REG1:[0-9]+]], r1, 100
; DARWIN32: lwz r[[REG3:[0-9]+]], 108(r1)
; DARWIN32: mr r[[REG2:[0-9]+]], r[[REG4]]
; DARWIN32: cmplw {{(cr[0-9]+,)?}}r[[REG4]], r[[REG3]]
; DARWIN32: stw r[[REG2]], -[[OFFSET1:[0-9]+]]
; DARWIN32: stw r[[REG3]], -[[OFFSET2:[0-9]+]]
; DARWIN32: lwz r[[REG1]], -[[OFFSET1]]
; DARWIN32: lwz r[[REG1]], -[[OFFSET2]]

; DARWIN64: _func4:
; DARWIN64: ld r[[REG2:[0-9]+]], 120(r1)
; DARWIN64: ld r[[REG3:[0-9]+]], 136(r1)
; DARWIN64: mr r[[REG4:[0-9]+]], r[[REG2]]
; DARWIN64: cmpld {{(cr[0-9]+,)?}}r[[REG2]], r[[REG3]]
; DARWIN64: std r[[REG4]], -[[OFFSET1:[0-9]+]]
; DARWIN64: std r[[REG3]], -[[OFFSET2:[0-9]+]]
; DARWIN64: ld r3, -[[OFFSET1]]
; DARWIN64: ld r3, -[[OFFSET2]]
