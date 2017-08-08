; ModuleID = '<stdin>'
source_filename = "conditional.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

; Function Attrs: noinline nounwind ssp uwtable
define i32 @SmallBlock_main() #0 {
entry:
  %retval = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %call = call i32 (...) @foo()
  store i32 %call, i32* %c, align 4
  %0 = load i32, i32* %c, align 4
  %mul = mul nsw i32 %0, 2
  %cmp = icmp sgt i32 %mul, 1
  br i1 %cmp, label %if.then, label %if.else5

if.then:                                          ; preds = %entry
  br label %det.achd

det.achd:                                         ; preds = %if.then
  %1 = bitcast i32 undef to i32
  %2 = load i32, i32* %c, align 4
  %cmp1 = icmp sgt i32 %2, 1
  br i1 %cmp1, label %if.then2, label %if.else

if.then2:                                         ; preds = %det.achd
  %call3 = call i32 (...) @bar()
  br label %if.end

if.else:                                          ; preds = %det.achd
  %call4 = call i32 (...) @foo()
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then2
  br label %det.cont

det.cont:                                         ; preds = %if.end
  br label %if.end17

if.else5:                                         ; preds = %entry
  %3 = load i32, i32* %c, align 4
  %mul6 = mul nsw i32 %3, 3
  %cmp7 = icmp slt i32 %mul6, 1
  br i1 %cmp7, label %if.then8, label %if.else12

if.then8:                                         ; preds = %if.else5
  br label %det.achd9

det.achd9:                                        ; preds = %if.then8
  %4 = bitcast i32 undef to i32
  %call10 = call i32 (...) @bar()
  br label %det.cont11

det.cont11:                                       ; preds = %det.achd9
  br label %if.end16

if.else12:                                        ; preds = %if.else5
  br label %det.achd13

det.achd13:                                       ; preds = %if.else12
  %5 = bitcast i32 undef to i32
  %call14 = call i32 (...) @foo()
  br label %det.cont15

det.cont15:                                       ; preds = %det.achd13
  br label %if.end16

if.end16:                                         ; preds = %det.cont15, %det.cont11
  br label %if.end17

if.end17:                                         ; preds = %if.end16, %det.cont
  %6 = load i32, i32* %c, align 4
  ret i32 %6
}

declare i32 @foo(...) #1

declare i32 @bar(...) #1

attributes #0 = { noinline nounwind ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 4.0.0 (git@github.com:wsmoses/Cilk-Clang cc78c4b6082bb80687e64c8104bf9744e6fa8fdc) (git@github.com:wsmoses/Parallel-IR 52889bc31182f3faebcfce24918670967b5b96f6)"}
