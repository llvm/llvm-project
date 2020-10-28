; RUN: opt < %s -inline -pass-remarks=inline -pass-remarks-missed=inline \
; RUN:       -pass-remarks-analysis=inline -S 2>&1 | \
; RUN:       FileCheck -check-prefixes=CHECK,NO_HOTNESS,ALWAYS %s
; RUN: opt < %s -inline -pass-remarks=inline -pass-remarks-missed=inline \
; RUN:       -pass-remarks-analysis=inline -pass-remarks-with-hotness -S 2>&1 | \
; RUN:       FileCheck -check-prefixes=CHECK,HOTNESS,ALWAYS %s

; RUN: opt < %s -passes=always-inline -pass-remarks=inline -pass-remarks-missed=inline \
; RUN:       -pass-remarks-analysis=inline -S 2>&1 | \
; RUN:       FileCheck -check-prefixes=ALWAYS %s
; RUN: opt < %s -passes=always-inline -pass-remarks=inline -pass-remarks-missed=inline \
; RUN:       -pass-remarks-analysis=inline -pass-remarks-with-hotness -S 2>&1 | \
; RUN:       FileCheck -check-prefixes=ALWAYS %s

; RUN: opt < %s -passes=inline -pass-remarks=inline -pass-remarks-missed=inline \
; RUN:       -pass-remarks-analysis=inline -S 2>&1 | \
; RUN:       FileCheck -check-prefixes=CHECK,NO_HOTNESS,ALWAYS %s
; RUN: opt < %s -passes=inline -pass-remarks=inline -pass-remarks-missed=inline \
; RUN:       -pass-remarks-analysis=inline -pass-remarks-with-hotness -S 2>&1 | \
; RUN:       FileCheck -check-prefixes=CHECK,HOTNESS,ALWAYS %s

; RUN: opt < %s -passes=inliner-wrapper -pass-remarks=inline -pass-remarks-missed=inline \
; RUN:       -pass-remarks-analysis=inline -S 2>&1 | \
; RUN:       FileCheck -check-prefixes=CHECK,NO_HOTNESS,ALWAYS %s
; RUN: opt < %s -passes=inliner-wrapper -pass-remarks=inline -pass-remarks-missed=inline \
; RUN:       -pass-remarks-analysis=inline -pass-remarks-with-hotness -S 2>&1 | \
; RUN:       FileCheck -check-prefixes=CHECK,HOTNESS,ALWAYS %s

; HOTNESS-DAG: fox will not be inlined into bar because its definition is unavailable
; NO_HOTNESS-NOT: fox will not be inlined into bar because its definition is unavailable
; ALWAYS-DAG: foo inlined into bar with (cost=always): always inline attribute
; CHECK-DAG: foz not inlined into bar because it should never be inlined (cost=never): noinline function attribute

; Function Attrs: alwaysinline nounwind uwtable
define i32 @foo(i32 %x, i32 %y) #0 !prof !1 {
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
define float @foz(i32 %x, i32 %y) #1 !prof !1 {
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

declare i32 @fox()

; Function Attrs: nounwind uwtable
define i32 @bar(i32 %j) #2 !prof !1 {
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
  %call3 = call i32 @fox()
  %add = add i32 %conv3, %call 
  ret i32 %add
}

attributes #0 = { alwaysinline nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.5.0 "}
!1 = !{!"function_entry_count", i64 10}
