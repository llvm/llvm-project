; RUN: opt -passes='default<O2>' -S < %s | FileCheck %s
; RUN: opt -passes='default<O3>' -S < %s | FileCheck %s

declare void @use(i64)

define i64 @ftp_state_list(i64 %0, i64 %1) {
  %3 = sub i64 %0, %1
  %4 = icmp eq i64 %0, %1
  br i1 %4, label %5, label %common.ret

5:
; CHECK: tail call void @use(i64 0)
  tail call void @use(i64 %3)
  br label %common.ret

common.ret:
  ret i64 %3
}
