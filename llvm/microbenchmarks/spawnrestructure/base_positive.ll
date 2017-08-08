; ModuleID = 'base_positive.c'
source_filename = "base_positive.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

; Function Attrs: noinline nounwind ssp uwtable
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  detach label %det.achd, label %det.cont

det.achd:                                         ; preds = %entry
  %0 = bitcast i32 undef to i32
  %call = call i32 (...) @foo()
  %call1 = call i32 (...) @bar()
  store i32 2, i32* %c, align 4
  reattach label %det.cont

det.cont:                                         ; preds = %det.achd, %entry
  detach label %det.achd2, label %det.cont5

det.achd2:                                        ; preds = %det.cont
  %1 = bitcast i32 undef to i32
  %call3 = call i32 (...) @foo()
  %call4 = call i32 (...) @foo()
  reattach label %det.cont5

det.cont5:                                        ; preds = %det.achd2, %det.cont
  %call6 = call i32 (...) @bar()
  ret i32 %call6
}

declare i32 @foo(...) #1

declare i32 @bar(...) #1

attributes #0 = { noinline nounwind ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 4.0.0 (git@github.com:wsmoses/Cilk-Clang cc78c4b6082bb80687e64c8104bf9744e6fa8fdc) (git@github.com:wsmoses/Parallel-IR 52889bc31182f3faebcfce24918670967b5b96f6)"}
