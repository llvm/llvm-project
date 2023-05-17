;RUN: llc -O2 -hexagon-emit-lut-text=true < %s | FileCheck --check-prefix=TEXT %s
;This test checks the placement of lookup table in text section.
target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon-unknown--elf"

;TEXT: .text
;TEXT-NOT: .section{{.*}}.rodata
;TEXT: .Lswitch.table:
;TEXT-NEXT: .word
@switch.table = private unnamed_addr constant [9 x i32] [i32 9, i32 20, i32 14, i32 22, i32 12, i32 5, i32 98, i32 8, i32 11]

; Function Attrs: norecurse nounwind readnone
define i32 @foo(i32 %x) local_unnamed_addr #0 {
entry:
  %0 = icmp ult i32 %x, 9
  br i1 %0, label %switch.lookup, label %return

switch.lookup:                                    ; preds = %entry
  %switch.gep = getelementptr inbounds [9 x i32], ptr @switch.table, i32 0, i32 %x
  %switch.load = load i32, ptr %switch.gep, align 4
  ret i32 %switch.load

return:                                           ; preds = %entry
  ret i32 19
}

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv60" "target-features"="-hvx,-long-calls" "unsafe-fp-math"="false" "use-soft-float"="false" }
