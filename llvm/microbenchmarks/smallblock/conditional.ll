; ModuleID = 'conditional.c'
source_filename = "conditional.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

; Function Attrs: nounwind ssp uwtable
define i32 @main() local_unnamed_addr #0 {
entry:
  %call = tail call i32 (...) @foo() #2
  %mul = shl nsw i32 %call, 1
  %cmp = icmp sgt i32 %mul, 1
  br i1 %cmp, label %if.then, label %if.else5

if.then:                                          ; preds = %entry
  detach label %det.achd, label %if.end17

det.achd:                                         ; preds = %if.then
  %cmp1 = icmp sgt i32 %call, 1
  br i1 %cmp1, label %if.then2, label %if.else

if.then2:                                         ; preds = %det.achd
  %call3 = tail call i32 (...) @bar() #2
  br label %if.end

if.else:                                          ; preds = %det.achd
  %call4 = tail call i32 (...) @foo() #2
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then2
  reattach label %if.end17

if.else5:                                         ; preds = %entry
  %cmp7 = icmp slt i32 %call, 1
  br i1 %cmp7, label %if.then8, label %if.else12

if.then8:                                         ; preds = %if.else5
  detach label %det.achd9, label %if.end17

det.achd9:                                        ; preds = %if.then8
  %call10 = tail call i32 (...) @bar() #2
  reattach label %if.end17

if.else12:                                        ; preds = %if.else5
  detach label %det.achd13, label %if.end17

det.achd13:                                       ; preds = %if.else12
  %call14 = tail call i32 (...) @foo() #2
  reattach label %if.end17

if.end17:                                         ; preds = %det.achd9, %if.then8, %det.achd13, %if.else12, %if.then, %if.end
  ret i32 %call
}

declare i32 @foo(...) local_unnamed_addr #1

declare i32 @bar(...) local_unnamed_addr #1

attributes #0 = { nounwind ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 4.0.0 (git@github.com:wsmoses/Cilk-Clang cc78c4b6082bb80687e64c8104bf9744e6fa8fdc) (git@github.com:wsmoses/Parallel-IR 52889bc31182f3faebcfce24918670967b5b96f6)"}
