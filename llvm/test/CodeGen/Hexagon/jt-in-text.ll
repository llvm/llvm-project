; RUN: llc -hexagon-emit-jt-text=true < %s | FileCheck %s
target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon-unknown--elf"

; CHECK: .text
; CHECK-NOT: .rodata
; CHECK: .word

@lane0_pwr_st = global i32 0, align 4
@lane1_pwr_st = global i32 0, align 4
@lane2_pwr_st = global i32 0, align 4
@lane3_pwr_st = global i32 0, align 4

; Function Attrs: noinline nounwind
define void @test2(i32 %lane_id, i32 %rx_pwr_st) #0 {
entry:
  %lane_id.addr = alloca i32, align 4
  %rx_pwr_st.addr = alloca i32, align 4
  store i32 %lane_id, i32* %lane_id.addr, align 4
  store i32 %rx_pwr_st, i32* %rx_pwr_st.addr, align 4
  %0 = load i32, i32* %lane_id.addr, align 4
  switch i32 %0, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
    i32 3, label %sw.bb3
    i32 15, label %sw.bb4
  ]

sw.bb:                                            ; preds = %entry
  store i32 1, i32* @lane0_pwr_st, align 4
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  store i32 1, i32* @lane1_pwr_st, align 4
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  store i32 1, i32* @lane2_pwr_st, align 4
  br label %sw.epilog

sw.bb3:                                           ; preds = %entry
  store i32 1, i32* @lane3_pwr_st, align 4
  br label %sw.epilog

sw.bb4:                                           ; preds = %entry
  store i32 1, i32* @lane0_pwr_st, align 4
  store i32 1, i32* @lane1_pwr_st, align 4
  store i32 1, i32* @lane2_pwr_st, align 4
  store i32 1, i32* @lane3_pwr_st, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb4, %sw.bb3, %sw.bb2, %sw.bb1, %sw.bb
  ret void
}

attributes #0 = { noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv60" "target-features"="-hvx-double,-long-calls" "unsafe-fp-math"="false" "use-soft-float"="false" }
