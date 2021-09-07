; Test that -inline-threshold overrides thresholds derived from opt levels.
; RUN: opt < %s -O2 -inline-threshold=500 -S  | FileCheck %s
; RUN: opt < %s -O3 -inline-threshold=500 -S  | FileCheck %s
; RUN: opt < %s -Os -inline-threshold=500 -S  | FileCheck %s
; RUN: opt < %s -Oz -inline-threshold=500 -S  | FileCheck %s

@a = global i32 4

define i32 @simpleFunction(i32 %a) #0 {
entry:
  %a1 = load volatile i32, i32* @a
  %x1 = add i32 %a1,  %a
  ret i32 %x1
}

; Function Attrs: nounwind readnone uwtable
define i32 @bar(i32 %a) #0 {
; CHECK-LABEL: @bar
; CHECK: load volatile
; CHECK-NEXT: add i32
; CHECK-NEXT: call i32 @simpleFunction
; CHECK: ret
entry:
  %i = tail call i32 @simpleFunction(i32 6) "function-inline-cost"="749"
  %j = tail call i32 @simpleFunction(i32 %i) "function-inline-cost"="750"
  ret i32 %j
}

attributes #0 = { nounwind readnone uwtable }
