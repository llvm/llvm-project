; RUN: llc -fast-isel -pass-remarks-missed=isel < %s 2>&1 >/dev/null | FileCheck -check-prefix=STDERR -allow-empty %s
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-linux-gnu"

declare { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } @ret_s10i64()

define i64 @call_ret_s10i64() {
; STDERR: FastISel missed call:   %ret = call { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } @ret_s10i64() (in function: call_ret_s10i64)
  %ret = call { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } @ret_s10i64()
  %ext0 = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %ret, 0
  %ext1 = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %ret, 1
  %sum = add i64 %ext0, %ext1
  ret i64 %sum
}
