target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

define void @main() {
entry:
  call i8* @lwt_fun()
  ret void
}

declare i8* @lwt_fun()
