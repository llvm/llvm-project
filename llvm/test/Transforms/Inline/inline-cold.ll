; RUN: opt < %s -passes='require<profile-summary>,cgscc(inline)' -S -inlinecold-threshold=25 | FileCheck %s
; Test that functions with attribute Cold are not inlined while the 
; same function without attribute Cold will be inlined.

; RUN: opt < %s -passes='require<profile-summary>,cgscc(inline)' -S -inline-threshold=600 -enable-new-pm=0 | FileCheck %s -check-prefix=OVERRIDE
; The command line argument for inline-threshold should override
; the default cold threshold, so a cold function with size bigger
; than the default cold threshold (225) will be inlined.

; RUN: opt < %s -passes='require<profile-summary>,cgscc(inline)' -S | FileCheck %s -check-prefix=DEFAULT
; The same cold function will not be inlined with the default behavior.

@a = global i32 4

; This function should be larger than the cold threshold (75), but smaller
; than the regular threshold.
; Function Attrs: nounwind readnone uwtable
define i32 @simpleFunction(i32 %a) #0 "function-inline-cost"="80" {
entry:
  ret i32 %a
}

; Function Attrs: nounwind cold readnone uwtable
define i32 @ColdFunction(i32 %a) #1 "function-inline-cost"="30" {
; CHECK-LABEL: @ColdFunction
; CHECK: ret
; OVERRIDE-LABEL: @ColdFunction
; OVERRIDE: ret
; DEFAULT-LABEL: @ColdFunction
; DEFAULT: ret
entry:
  ret i32 %a
}

; This function should be larger than the default cold threshold (225).
define i32 @ColdFunction2(i32 %a) #1 "function-inline-cost"="250" {
; CHECK-LABEL: @ColdFunction2
; CHECK: ret
; OVERRIDE-LABEL: @ColdFunction2
; OVERRIDE: ret
; DEFAULT-LABEL: @ColdFunction2
; DEFAULT: ret
entry:
  ret i32 %a
}

; Function Attrs: nounwind readnone uwtable
define i32 @bar(i32 %a) #0 {
; CHECK-LABEL: @bar
; CHECK: call i32 @ColdFunction(i32 5)
; CHECK-NOT: call i32 @simpleFunction(i32 6)
; CHECK: call i32 @ColdFunction2(i32 5)
; CHECK: ret
; OVERRIDE-LABEL: @bar
; OVERRIDE-NOT: call i32 @ColdFunction(i32 5)
; OVERRIDE-NOT: call i32 @simpleFunction(i32 6)
; OVERRIDE-NOT: call i32 @ColdFunction2(i32 5)
; OVERRIDE: ret
; DEFAULT-LABEL: @bar
; DEFAULT-NOT: call i32 @ColdFunction(i32 5)
; DEFAULT-NOT: call i32 @simpleFunction(i32 6)
; DEFAULT: call i32 @ColdFunction2(i32 5)
; DEFAULT: ret
entry:
  %0 = tail call i32 @ColdFunction(i32 5)
  %1 = tail call i32 @simpleFunction(i32 6)
  %2 = tail call i32 @ColdFunction2(i32 5)
  %3 = add i32 %0, %1
  %add = add i32 %2, %3
  ret i32 %add
}

declare void @extern()
attributes #0 = { nounwind readnone uwtable }
attributes #1 = { nounwind cold readnone uwtable }
