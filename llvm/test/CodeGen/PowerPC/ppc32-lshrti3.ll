; RUN: llc -verify-machineinstrs -O=2 < %s -mtriple=powerpc-netbsd | FileCheck %s

; CHECK-NOT: bl __lshrti3

; ModuleID = 'lshrti3-ppc32.c'
target datalayout = "E-m:e-p:32:32-i64:64-n32"
target triple = "powerpc--netbsd"

; Function Attrs: nounwind uwtable
define i32 @fn1() #0 {
entry:
  %.promoted = load i72, ptr inttoptr (i32 1 to ptr), align 4
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %bf.set3 = phi i72 [ %bf.set, %while.cond ], [ %.promoted, %entry ]
  %bf.lshr = lshr i72 %bf.set3, 40
  %bf.lshr.tr = trunc i72 %bf.lshr to i32
  %bf.cast = and i32 %bf.lshr.tr, 65535
  %dec = add nsw i32 %bf.lshr.tr, 65535
  %0 = zext i32 %dec to i72
  %bf.value = shl nuw i72 %0, 40
  %bf.shl = and i72 %bf.value, 72056494526300160
  %bf.clear2 = and i72 %bf.set3, -72056494526300161
  %bf.set = or i72 %bf.shl, %bf.clear2
  %tobool = icmp eq i32 %bf.cast, 0
  br i1 %tobool, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  %bf.set.lcssa = phi i72 [ %bf.set, %while.cond ]
  store i72 %bf.set.lcssa, ptr inttoptr (i32 1 to ptr), align 4
  ret i32 undef
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.5.0 (213754)"}
