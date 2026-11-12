; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-unknown \
; RUN:  -ppc-asm-full-reg-names -mcpu=pwr9 < %s  | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-unknown \
; RUN:  -ppc-asm-full-reg-names -mcpu=pwr9 < %s  | FileCheck %s

define i64 @addze1(i64 %X, i64 %Z) {
; CHECK-LABEL: addze1:
; CHECK: # %bb.0:
; CHECK-NEXT: addic [[REG1:r[0-9]+]], [[REG1]], -1
; CHECK-NEXT: addze [[REG2:r[0-9]+]], [[REG2]]
; CHECK-NEXT: blr
  %cmp = icmp ne i64 %Z, 0
  %conv1 = zext i1 %cmp to i64
  %add = add nsw i64 %conv1, %X
  ret i64 %add
}

define i64 @addze2(i64 %X, i64 %Z) {
; CHECK-LABEL: addze2:
; CHECK: # %bb.0:
; CHECK-NEXT: subfic [[REG1:r[0-9]+]], [[REG1]], 0
; CHECK-NEXT: addze  [[REG2:r[0-9]+]], [[REG2]]
; CHECK-NEXT: blr
  %cmp = icmp eq i64 %Z, 0
  %conv1 = zext i1 %cmp to i64
  %add = add nsw i64 %conv1, %X
  ret i64 %add
}

define i64 @addze3(i64 %X, i64 %Z) {
; CHECK-LABEL: addze3:
; CHECK: # %bb.0:
; CHECK-NEXT: addi  [[REG1:r[0-9]+]], [[REG1]], -32768
; CHECK-NEXT: addic [[REG1]], [[REG1]], -1
; CHECK-NEXT: addze [[REG2:r[0-9]+]], [[REG2]]
; CHECK-NEXT: blr
  %cmp = icmp ne i64 %Z, 32768
  %conv1 = zext i1 %cmp to i64
  %add = add nsw i64 %conv1, %X
  ret i64 %add
}

define i64 @addze4(i64 %X, i64 %Z) {
; CHECK-LABEL: addze4:
; CHECK: # %bb.0:
; CHECK-NEXT: addi   [[REG1:r[0-9]+]], [[REG1]], -32768
; CHECK-NEXT: subfic [[REG1]], [[REG1]], 0
; CHECK-NEXT: addze  [[REG2:r[0-9]+]], [[REG2]]
; CHECK-NEXT: blr
  %cmp = icmp eq i64 %Z, 32768
  %conv1 = zext i1 %cmp to i64
  %add = add nsw i64 %conv1, %X
  ret i64 %add
}

define i64 @addze5(i64 %X, i64 %Z) {
; CHECK-LABEL: addze5:
; CHECK: # %bb.0:
; CHECK-NEXT: addi  [[REG1:r[0-9]+]], [[REG1]], 32767
; CHECK-NEXT: addic [[REG1]], [[REG1]], -1
; CHECK-NEXT: addze [[REG2:r[0-9]+]], [[REG2]]
; CHECK-NEXT: blr
  %cmp = icmp ne i64 %Z, -32767
  %conv1 = zext i1 %cmp to i64
  %add = add nsw i64 %conv1, %X
  ret i64 %add
}

define i64 @addze6(i64 %X, i64 %Z) {
; CHECK-LABEL: addze6:
; CHECK: # %bb.0:
; CHECK-NEXT: addi   [[REG1:r[0-9]+]], [[REG1]], 32767
; CHECK-NEXT: subfic [[REG1]], [[REG1]], 0
; CHECK-NEXT: addze  [[REG2:r[0-9]+]], [[REG2]]
; CHECK-NEXT: blr
  %cmp = icmp eq i64 %Z, -32767
  %conv1 = zext i1 %cmp to i64
  %add = add nsw i64 %conv1, %X
  ret i64 %add
}

; element is out of range
define i64 @test1(i64 %X, i64 %Z) {
; CHECK-LABEL: test1:
; CHECK: # %bb.0:
; CHECK-NEXT: li    [[REG1:r[0-9]+]], -32768
; CHECK-NEXT: xor   [[REG2:r[0-9]+]], [[REG2]], [[REG1]]
; CHECK-NEXT: addic [[REG1]], [[REG2]], -1
; CHECK-NEXT: subfe [[REG2]], [[REG1]], [[REG2]]
; CHECK-NEXT: add   [[REG3:r[0-9]+]], [[REG2]], [[REG3]]
; CHECK-NEXT: blr
  %cmp = icmp ne i64 %Z, -32768
  %conv1 = zext i1 %cmp to i64
  %add = add nsw i64 %conv1, %X
  ret i64 %add
}

define i64 @test2(i64 %X, i64 %Z) {
; CHECK-LABEL: test2:
; CHECK: # %bb.0:
; CHECK-NEXT: li     [[REG1:r[0-9]+]], -32768
; CHECK-NEXT: xor    [[REG2:r[0-9]+]], [[REG2]], [[REG1]]
; CHECK-NEXT: cntlzd [[REG2]], [[REG2]]
; CHECK-NEXT: rldicl [[REG2]], [[REG2]], 58, 63
; CHECK-NEXT: add    [[REG3:r[0-9]+]], [[REG2]], [[REG3]]
; CHECK-NEXT: blr
  %cmp = icmp eq i64 %Z, -32768
  %conv1 = zext i1 %cmp to i64
  %add = add nsw i64 %conv1, %X
  ret i64 %add
}

define i64 @test3(i64 %X, i64 %Z) {
; CHECK-LABEL: test3:
; CHECK: # %bb.0:
; CHECK-NEXT: li    [[REG1:r[0-9]+]], 0
; CHECK-NEXT: ori   [[REG1]], [[REG1]], 32769
; CHECK-NEXT: xor   [[REG2:r[0-9]+]], [[REG2]], [[REG1]]
; CHECK-NEXT: addic [[REG1]], [[REG2]], -1
; CHECK-NEXT: subfe [[REG2]], [[REG1]], [[REG2]]
; CHECK-NEXT: add   [[REG3:r[0-9]+]], [[REG2]], [[REG3]]
; CHECK-NEXT: blr
  %cmp = icmp ne i64 %Z, 32769
  %conv1 = zext i1 %cmp to i64
  %add = add nsw i64 %conv1, %X
  ret i64 %add
}

define i64 @test4(i64 %X, i64 %Z) {
; CHECK-LABEL: test4:
; CHECK: # %bb.0:
; CHECK-NEXT: li     [[REG1:r[0-9]+]], 0
; CHECK-NEXT: ori    [[REG1]], [[REG1]], 32769
; CHECK-NEXT: xor    [[REG2:r[0-9]+]], [[REG2]], [[REG1]]
; CHECK-NEXT: cntlzd [[REG2]], [[REG2]]
; CHECK-NEXT: rldicl [[REG2]], [[REG2]], 58, 63
; CHECK-NEXT: add    [[REG3:r[0-9]+]], [[REG2]], [[REG3]]
; CHECK-NEXT: blr
  %cmp = icmp eq i64 %Z, 32769
  %conv1 = zext i1 %cmp to i64
  %add = add nsw i64 %conv1, %X
  ret i64 %add
}

; comparison of two registers
define i64 @test5(i64 %X, i64 %Y, i64 %Z) {
; CHECK-LABEL: test5:
; CHECK: # %bb.0:
; CHECK-NEXT: xor   [[REG2:r[0-9]+]], [[REG2]], [[REG1:r[0-9]+]]
; CHECK-NEXT: addic [[REG1]], [[REG2]], -1
; CHECK-NEXT: subfe [[REG2]], [[REG1]], [[REG2]]
; CHECK-NEXT: add   [[REG3:r[0-9]+]], [[REG2]], [[REG3]]
; CHECK-NEXT: blr
  %cmp = icmp ne i64 %Y, %Z
  %conv1 = zext i1 %cmp to i64
  %add = add nsw i64 %conv1, %X
  ret i64 %add
}

define i64 @test6(i64 %X, i64 %Y, i64 %Z) {
; CHECK-LABEL: test6:
; CHECK: # %bb.0:
; CHECK-NEXT: xor    [[REG2:r[0-9]+]], [[REG2]], [[REG1:r[0-9]+]]
; CHECK-NEXT: cntlzd [[REG2]], [[REG2]]
; CHECK-NEXT: rldicl [[REG2]], [[REG2]], 58, 63
; CHECK-NEXT: add    [[REG3:r[0-9]+]], [[REG2]], [[REG3]]
; CHECK-NEXT: blr
  %cmp = icmp eq i64 %Y, %Z
  %conv1 = zext i1 %cmp to i64
  %add = add nsw i64 %conv1, %X
  ret i64 %add
}
