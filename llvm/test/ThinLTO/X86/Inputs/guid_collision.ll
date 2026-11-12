target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; The source for the GUID for this symbol will be -:F
source_filename = "-"
define internal fastcc i64 @F() {
  ret i64 0
}

@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

define i64 @G() {
  ;%1 = load i32, ptr @dummy2, align 4
  ret i64 0
}
