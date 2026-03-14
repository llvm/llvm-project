; ModuleID = 'thinlto_weak_library1.c'
source_filename = "thinlto_weak_library1.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define weak i32 @f() local_unnamed_addr {
entry:
  ret i32 1
}

; Function Attrs: norecurse nounwind readnone uwtable
define i32 @test1() local_unnamed_addr {
entry:
  %call = tail call i32 @f()
  ret i32 %call
}
