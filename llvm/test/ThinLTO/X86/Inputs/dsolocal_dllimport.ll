target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.0"
define dso_local dllexport i32 @foo() {
  ret i32 42
}
