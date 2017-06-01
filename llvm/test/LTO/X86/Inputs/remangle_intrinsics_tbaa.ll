target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

%some_named_struct = type { i8, i8 }

define void @bar(%some_named_struct*) {
  ret void
}
