target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"



@analias = alias void (...), bitcast (void ()* @aliasee to void (...)*)

; Function Attrs: nounwind uwtable
define void @aliasee() #0 {
entry:
    ret void
}

