; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu < %s \
; RUN:   -verify-machineinstrs | FileCheck %s

@b = common dso_local local_unnamed_addr global i64* null, align 8
@a = common dso_local local_unnamed_addr global i8 0, align 1

define void @testADDEPromoteResult() {
entry:
  %0 = load i64*, i64** @b, align 8
  %1 = load i64, i64* %0, align 8
  %cmp = icmp ne i64* %0, null
  %conv1 = zext i1 %cmp to i64
  %add = add nsw i64 %1, %conv1
  %2 = trunc i64 %add to i8
  %conv2 = and i8 %2, 5
  store i8 %conv2, i8* @a, align 1
  ret void

; CHECK-LABEL: @testADDEPromoteResult
; CHECK:      # %bb.0:
; CHECK-DAG:   addis [[REG1:[0-9]+]], [[REG2:[0-9]+]], [[VAR1:[a-z0-9A-Z_.]+]]@toc@ha
; CHECK-DAG:   ld [[REG3:[0-9]+]], [[VAR1]]@toc@l([[REG1]])
; CHECK-DAG:   lbz [[REG4:[0-9]+]], 0([[REG3]])
; CHECK-DAG:   addic [[REG5:[0-9]+]], [[REG3]], -1
; CHECK-DAG:   extsb [[REG6:[0-9]+]], [[REG4]]
; CHECK-DAG:   addze [[REG7:[0-9]+]], [[REG6]]
; CHECK-DAG:   addis [[REG8:[0-9]+]], [[REG2]], [[VAR2:[a-z0-9A-Z_.]+]]@toc@ha
; CHECK-DAG:   andi. [[REG9:[0-9]+]], [[REG7]], 5
; CHECK-DAG:   stb [[REG9]], [[VAR2]]@toc@l([[REG8]])
; CHECK:       blr
}
