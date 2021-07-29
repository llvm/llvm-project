target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() {
entry:
  call void @llvm.pseudoprobe(i64 6699318081062747564, i64 1, i32 0, i64 -1)
  ret void
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64) #0

attributes #0 = { inaccessiblememonly nounwind willreturn }

!llvm.pseudo_probe_desc = !{!0}

!0 = !{i64 6699318081062747564, i64 4294967295, !"foo", null}