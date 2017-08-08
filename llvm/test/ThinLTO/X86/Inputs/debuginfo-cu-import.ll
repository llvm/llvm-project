; ModuleID = 'debuginfo-cu-import2.c'
source_filename = "debuginfo-cu-import2.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @main() {
entry:
  call void (...) @foo()
  ret i32 0
}

declare void @foo(...) #1
