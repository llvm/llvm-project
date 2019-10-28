target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; The source for the GUID for this symbol will be -:F
source_filename = "-"
define internal fastcc i64 @F() {
  ret i64 0
}

; Needed to give llvm-lto2 something to do
@dummy2 = global i32 0

