; RUN: llc %s -aarch64-min-jump-table-entries=4 -o - | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; Function Attrs: nounwind
define dso_local void @f(i64 %v) local_unnamed_addr #0 {
entry:
  %call = tail call i32 @test() #0
  %and = and i32 %call, 2
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %if.then, label %if.else
; CHECK: tbz
; CHECK-NEXT: b
if.then:                                          ; preds = %entry
  switch i64 %v, label %sw.epilog [
    i64 0, label %sw.bb
    i64 1, label %sw.bb1
    i64 2, label %sw.bb2
    i64 3, label %sw.bb3
  ]

sw.bb:                                            ; preds = %if.then
  tail call void @g0() #0
  br label %sw.bb1

sw.bb1:                                           ; preds = %if.then, %sw.bb
  tail call void @g1() #0
  br label %sw.bb2

sw.bb2:                                           ; preds = %if.then, %sw.bb1
  tail call void @g2() #0
  br label %sw.bb3

sw.bb3:                                           ; preds = %if.then, %sw.bb2
  tail call void @g3() #0
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.bb3, %if.then
  %dummy = tail call i64 @llvm.aarch64.space(i32 32700, i64 %v)
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void @e() #0
  br label %if.end

if.end:                                           ; preds = %if.else, %sw.epilog
  ret void
}

declare dso_local i32 @test(...) local_unnamed_addr #0

declare dso_local void @g0(...) local_unnamed_addr #0

declare dso_local void @g1(...) local_unnamed_addr #0

declare dso_local void @g2(...) local_unnamed_addr #0

declare dso_local void @g3(...) local_unnamed_addr #0

declare dso_local void @e(...) local_unnamed_addr #0

declare dso_local i64 @llvm.aarch64.space(i32, i64) local_unnamed_addr #0

attributes #0 = { nounwind "branch-target-enforcement"="true" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon,+v8.5a" "unsafe-fp-math"="false" "use-soft-float"="false" }
