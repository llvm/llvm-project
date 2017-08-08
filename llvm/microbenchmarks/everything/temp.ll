; ModuleID = '<stdin>'
source_filename = "everything.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

; Function Attrs: noinline norecurse nounwind readnone ssp uwtable
define i32 @SpawnUnswitch_SmallBlock_RedundantSpawn_foo() local_unnamed_addr #0 {
entry:
  ret i32 10
}

; Function Attrs: noinline norecurse nounwind readnone ssp uwtable
define i32 @SpawnUnswitch_SmallBlock_RedundantSpawn_main() local_unnamed_addr #0 {
entry:
  ret i32 9
}

attributes #0 = { noinline norecurse nounwind readnone ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 5.0.0 (git@github.com:wsmoses/Cilk-Clang 5942594810265567795884c83b5a37a8cbc98d3e) (git@github.com:wsmoses/Parallel-IR 8f57e0739bf9fc6736472c89f91a533630efd5c3)"}
