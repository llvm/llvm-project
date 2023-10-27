; RUN: llc -verify-machineinstrs -aarch64-enable-atomic-cfg-tidy=0 < %s -mtriple=aarch64 -fast-isel=false | FileCheck %s


define i32 @ule_64_max(i64 %p) {
entry:
; CHECK-LABEL: ule_64_max:
; CHECK: cmn x0, #1
; CHECK: b.hi [[RET_ZERO:.LBB[0-9]+_[0-9]+]]
  %cmp = icmp ule i64 %p, 18446744073709551615 ; 0xffffffffffffffff
  br i1 %cmp, label %ret_one, label %ret_zero

ret_one:
  ret i32 1

ret_zero:
; CHECK: [[RET_ZERO]]:
; CHECK-NEXT: mov w0, wzr
  ret i32 0
}

define i32 @ugt_64_max(i64 %p) {
entry:
; CHECK-LABEL: ugt_64_max:
; CHECK: cmn x0, #1
; CHECK: b.ls [[RET_ZERO:.LBB[0-9]+_[0-9]+]]
  %cmp = icmp ugt i64 %p, 18446744073709551615 ; 0xffffffffffffffff
  br i1 %cmp, label %ret_one, label %ret_zero

ret_one:
  ret i32 1

ret_zero:
; CHECK: [[RET_ZERO]]:
; CHECK-NEXT: mov w0, wzr
  ret i32 0
}
