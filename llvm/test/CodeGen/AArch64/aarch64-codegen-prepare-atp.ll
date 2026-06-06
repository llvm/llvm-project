; RUN: opt -passes='require<profile-summary>,function(codegenprepare)' < %s -S | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

%struct.match_state = type { i64, i64  }

; %add is also promoted by forking an extra sext.
define void @promoteTwoOne(i32 %i, i32 %j, ptr %P1, ptr %P2 ) {
; CHECK-LABEL: @promoteTwoOne
; CHECK-LABEL: entry:
; CHECK: %[[SEXT1:.*]] = sext i32 %i to i64
; CHECK: %[[SEXT2:.*]] = sext i32 %j to i64
; CHECK: %add = add nsw i64 %[[SEXT1]], %[[SEXT2]]
entry:
  %add = add nsw i32 %i, %j
  %s = sext i32 %add to i64
  %addr1 = getelementptr inbounds i64, ptr %P1, i64 %s
  store i64 %s, ptr %addr1
  %s2 = sext i32 %i to i64
  %addr2 = getelementptr inbounds i64, ptr %P2, i64 %s2
  store i64 %s2, ptr %addr2
  ret void
}

; Both %add1 and %add2 are promoted by forking extra sexts.
define void @promoteTwoTwo(i32 %i, i32 %j, i32 %k, ptr %P1, ptr %P2) {
; CHECK-LABEL: @promoteTwoTwo
; CHECK-LABEL:entry:
; CHECK: %[[SEXT1:.*]] = sext i32 %j to i64
; CHECK: %[[SEXT2:.*]]  = sext i32 %i to i64
; CHECK: %add1 = add nsw i64 %[[SEXT1]], %[[SEXT2]]
; CHECK: %[[SEXT3:.*]] = sext i32 %k to i64
; CHECK: %add2 = add nsw i64 %[[SEXT1]], %[[SEXT3]]
entry:
  %add1 = add nsw i32 %j, %i
  %s = sext i32 %add1 to i64
  %addr1 = getelementptr inbounds i64, ptr %P1, i64 %s
  store i64 %s, ptr %addr1
  %add2 = add nsw i32 %j, %k
  %s2 = sext i32 %add2 to i64
  %addr2 = getelementptr inbounds i64, ptr %P2, i64 %s2
  store i64 %s2, ptr %addr2
  ret void
}

define i64 @promoteGEPSunk(i1 %cond, ptr %base, i32 %i) {
; CHECK-LABEL: @promoteGEPSunk
; CHECK-LABEL: entry:
; CHECK:  %[[SEXT:.*]] = sext i32 %i to i64
; CHECK:  %add = add nsw i64 %[[SEXT]], 1
; CHECK:  %add2 = add nsw i64 %[[SEXT]], 2
entry:
  %add = add nsw i32 %i, 1
  %s = sext i32 %add to i64
  %addr = getelementptr inbounds i64, ptr %base, i64 %s
  %add2 = add nsw i32 %i,  2
  %s2 = sext i32 %add2 to i64
  %addr2 = getelementptr inbounds i64, ptr %base, i64 %s2
  br i1 %cond, label %if.then, label %if.then2
if.then:
  %v = load i64, ptr %addr
  %v2 = load i64, ptr %addr2
  %r = add i64 %v, %v2
  ret i64 %r
if.then2:
  ret i64 0;
}
