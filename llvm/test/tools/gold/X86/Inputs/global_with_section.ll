target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @deadfunc2_called_from_section() {
  ret void
}

define void @deadfunc2_called_from_nonC_section() {
  ret void
}
