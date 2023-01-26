target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @g() {
entry:
  ret void
}

@analias = alias void (...), ptr @aliasee
define void @aliasee() {
entry:
  ret void
}
