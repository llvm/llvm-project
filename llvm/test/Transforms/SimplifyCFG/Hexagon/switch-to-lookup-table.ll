; RUN: opt -S -O2 < %s | FileCheck %s -check-prefix=ENABLE
; RUN: opt -S -hexagon-emit-lookup-tables=true -O2 < %s | FileCheck %s -check-prefix=ENABLE
; RUN: opt -S -hexagon-emit-lookup-tables=false -O2 < %s | FileCheck %s -check-prefix=DISABLE


; ENABLE: @{{.*}} = private unnamed_addr constant [6 x i32] [i32 9, i32 20, i32 14, i32 22, i32 12, i32 5]
; DISABLE-NOT: @{{.*}} = private unnamed_addr constant [6 x i32] [i32 9, i32 20, i32 14, i32 22, i32 12, i32 5]
; DISABLE : = phi i32 [ 19, %{{.*}} ], [ 5, %{{.*}} ], [ 12, %{{.*}} ], [ 22, %{{.*}} ], [ 14, %{{.*}} ], [ 20, %{{.*}} ], [ 9, %{{.*}} ]

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon-unknown--elf"

; Function Attrs: noinline nounwind
define i32 @foo(i32 %x) #0 section ".tcm_text" {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  %0 = load i32, i32* %x.addr, align 4
  switch i32 %0, label %sw.default [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
    i32 3, label %sw.bb3
    i32 4, label %sw.bb4
    i32 5, label %sw.bb5
  ]

sw.bb:                                            ; preds = %entry
  store i32 9, i32* %retval, align 4
  br label %return

sw.bb1:                                           ; preds = %entry
  store i32 20, i32* %retval, align 4
  br label %return

sw.bb2:                                           ; preds = %entry
  store i32 14, i32* %retval, align 4
  br label %return

sw.bb3:                                           ; preds = %entry
  store i32 22, i32* %retval, align 4
  br label %return

sw.bb4:                                           ; preds = %entry
  store i32 12, i32* %retval, align 4
  br label %return

sw.bb5:                                           ; preds = %entry
  store i32 5, i32* %retval, align 4
  br label %return

sw.default:                                       ; preds = %entry
  store i32 19, i32* %retval, align 4
  br label %return

return:                                           ; preds = %sw.default, %sw.bb5, %sw.bb4, %sw.bb3, %sw.bb2, %sw.bb1, %sw.bb
  %1 = load i32, i32* %retval, align 4
  ret i32 %1
}

attributes #0 = { noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv60" "target-features"="-hvx,-long-calls" "unsafe-fp-math"="false" "use-soft-float"="false" }
