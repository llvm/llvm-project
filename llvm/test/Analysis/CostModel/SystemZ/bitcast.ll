; RUN: opt < %s -mtriple=systemz-unknown -mcpu=z15 -passes="print<cost-model>" \
; RUN:   -disable-output 2>&1 | FileCheck %s

; Check bitcast from scalar to vector.

@Glob = dso_local local_unnamed_addr global i32 0, align 4

define dso_local void @fun() {
entry:
  %d.sroa.0 = alloca i64, align 8
  store i64 0, ptr %d.sroa.0, align 8
  store i32 2, ptr @Glob, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.cond1, %entry
  %L = load i64, ptr %d.sroa.0, align 8
  %A0 = and i64 %L, 4294967295
  store i64 %A0, ptr %d.sroa.0, align 8
  %BC = bitcast i64 %A0 to <2 x i32>
  %0 = and <2 x i32> %BC, splat (i32 10)
  store <2 x i32> %0, ptr %d.sroa.0, align 8
  br label %for.cond1

; CHECK:      Printing analysis 'Cost Model Analysis' for function 'fun':
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   %d.sroa.0 = alloca i64, align 8
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   store i64 0, ptr %d.sroa.0, align 8
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   store i32 2, ptr @Glob, align 4
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   br label %for.cond1
; CHECK-NEXT: Cost Model: Found an estimated cost of 0 for instruction:   %L = load i64, ptr %d.sroa.0, align 8
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %A0 = and i64 %L, 4294967295
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   store i64 %A0, ptr %d.sroa.0, align 8
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %BC = bitcast i64 %A0 to <2 x i32>
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %0 = and <2 x i32> %BC, splat (i32 10)
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   store <2 x i32> %0, ptr %d.sroa.0, align 8
; CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   br label %for.cond1
}
