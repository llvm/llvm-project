; RUN: opt < %s -inline -pass-remarks=inline -pass-remarks-missed=inline -pass-remarks-analysis=inline -S 2>&1 | FileCheck %s

; CHECK: foo should always be inlined (cost=always)
; CHECK: foo inlined into bar
; CHECK: foz should never be inlined (cost=never)
; CHECK: foz will not be inlined into bar

; Function Attrs: alwaysinline nounwind uwtable
define i32 @foo(i32 %x, i32 %y) #0 {
entry:
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  store i32 %y, i32* %y.addr, align 4
  %0 = load i32, i32* %x.addr, align 4
  %1 = load i32, i32* %y.addr, align 4
  %add = add nsw i32 %0, %1
  ret i32 %add
}

; Function Attrs: noinline nounwind uwtable
define float @foz(i32 %x, i32 %y) #1 {
entry:
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  store i32 %y, i32* %y.addr, align 4
  %0 = load i32, i32* %x.addr, align 4
  %1 = load i32, i32* %y.addr, align 4
  %mul = mul nsw i32 %0, %1
  %conv = sitofp i32 %mul to float
  ret float %conv
}

; Function Attrs: nounwind uwtable
define i32 @bar(i32 %j) #2 {
entry:
  %j.addr = alloca i32, align 4
  store i32 %j, i32* %j.addr, align 4
  %0 = load i32, i32* %j.addr, align 4
  %1 = load i32, i32* %j.addr, align 4
  %sub = sub nsw i32 %1, 2
  %call = call i32 @foo(i32 %0, i32 %sub)
  %conv = sitofp i32 %call to float
  %2 = load i32, i32* %j.addr, align 4
  %sub1 = sub nsw i32 %2, 2
  %3 = load i32, i32* %j.addr, align 4
  %call2 = call float @foz(i32 %sub1, i32 %3)
  %mul = fmul float %conv, %call2
  %conv3 = fptosi float %mul to i32
  ret i32 %conv3
}

attributes #0 = { alwaysinline nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.5.0 "}
