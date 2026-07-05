; NOTE: This test verifies that a redundant load immediate of zero is folded
; NOTE: from its use in an isel and deleted as it is no longer in use.
; RUN:  llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:      -ppc-asm-full-reg-names -verify-machineinstrs < %s | FileCheck %s
; RUN:  llc -mcpu=pwr9 -mtriple=powerpc64-unknown-linux-gnu \
; RUN:      -ppc-asm-full-reg-names -verify-machineinstrs < %s | FileCheck %s

%0 = type { i32, i16 }

@val = common dso_local local_unnamed_addr global ptr null, align 8

define dso_local signext i32 @redunLoadImm(ptr %arg) {
; CHECK-LABEL: redunLoadImm:
; verify that the load immediate has been folded into the isel and deleted
; CHECK-NOT:   li r[[REG1:[0-9]+]], 0
; CHECK:       iseleq r[[REG2:[0-9]+]], 0, r[[REG3:[0-9]+]]

bb:
  %tmp = icmp eq ptr %arg, null
  br i1 %tmp, label %bb9, label %bb1

bb1:                                              ; preds = %bb
  %tmp2 = getelementptr inbounds %0, ptr %arg, i64 0, i32 1
  br label %bb3

bb3:                                              ; preds = %bb3, %bb1
  %tmp4 = load i16, ptr %tmp2, align 4
  %tmp5 = sext i16 %tmp4 to i64
  %tmp6 = getelementptr inbounds %0, ptr %arg, i64 %tmp5
  %tmp7 = icmp eq i16 %tmp4, 0
  %tmp8 = select i1 %tmp7, ptr null, ptr %tmp6
  store ptr %tmp8, ptr @val, align 8
  br label %bb3

bb9:                                              ; preds = %bb
  %tmp10 = load ptr, ptr @val, align 8
  %tmp12 = load i32, ptr %tmp10, align 4
  ret i32 %tmp12
}
