; RUN: opt -passes=loop-vectorize < %s -S -o - | FileCheck %s --check-prefix=CHECK
; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s --check-prefix=CHECK-COST
; REQUIRES: asserts

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-none-none-eabi"

; CHECK-LABEL: test
; CHECK-COST: LV: Found an estimated cost of 0 for VF 1 For instruction:   %and515 = shl i32 %l41, 3
; CHECK-COST: LV: Found an estimated cost of 1 for VF 1 For instruction:   %l45 = and i32 %and515, 131072
; CHECK-COST: Cost of 2 for VF 4: WIDEN ir<%and515> = shl ir<%l41>, ir<3>
; CHECK-COST: Cost of 2 for VF 4: WIDEN ir<%l45> = and ir<%and515>, ir<131072>
; CHECK-NOT: vector.body

define void @test(ptr %src, i32 %N) #0 {
entry:
  br label %for.body386

for.body386:                                      ; preds = %entry, %l77
  %add387 = phi i32 [ %inc532, %l77 ], [ 0, %entry ]
  %arrayidx388 = getelementptr inbounds [101 x i32], ptr %src, i32 0, i32 %add387
  %l41 = load i32, ptr %arrayidx388, align 4
  %l42 = and i32 %l41, 65535
  %l43 = icmp eq i32 %l42, 0
  br i1 %l43, label %l77, label %l44

l44:                                               ; preds = %for.body386
  %and515 = shl i32 %l41, 3
  %l45 = and i32 %and515, 131072
  %and506 = shl i32 %l41, 5
  %l46 = and i32 %and506, 262144
  %and497 = shl i32 %l41, 7
  %l47 = and i32 %and497, 524288
  %and488 = shl i32 %l41, 9
  %l48 = and i32 %and488, 1048576
  %and479 = shl i32 %l41, 11
  %l49 = and i32 %and479, 2097152
  %and470 = shl i32 %l41, 13
  %l50 = and i32 %and470, 4194304
  %and461 = shl i32 %l41, 15
  %l51 = and i32 %and461, 8388608
  %and452 = shl i32 %l41, 17
  %l52 = and i32 %and452, 16777216
  %and443 = shl i32 %l41, 19
  %l53 = and i32 %and443, 33554432
  %and434 = shl i32 %l41, 21
  %l54 = and i32 %and434, 67108864
  %and425 = shl i32 %l41, 23
  %l55 = and i32 %and425, 134217728
  %and416 = shl i32 %l41, 25
  %l56 = and i32 %and416, 268435456
  %and407 = shl i32 %l41, 27
  %l57 = and i32 %and407, 536870912
  %and398 = shl i32 %l41, 29
  %l58 = and i32 %and398, 1073741824
  %l59 = shl i32 %l41, 31
  %l60 = or i32 %l59, %l41
  %l61 = or i32 %l58, %l60
  %l62 = or i32 %l57, %l61
  %l63 = or i32 %l56, %l62
  %l64 = or i32 %l55, %l63
  %l65 = or i32 %l54, %l64
  %l66 = or i32 %l53, %l65
  %l67 = or i32 %l52, %l66
  %l68 = or i32 %l51, %l67
  %l69 = or i32 %l50, %l68
  %l70 = or i32 %l49, %l69
  %l71 = or i32 %l48, %l70
  %l72 = or i32 %l47, %l71
  %l73 = or i32 %l46, %l72
  %l74 = or i32 %l45, %l73
  %and524 = shl i32 %l41, 1
  %l75 = and i32 %and524, 65536
  %l76 = or i32 %l75, %l74
  store i32 %l76, ptr %arrayidx388, align 4
  br label %l77

l77:                                               ; preds = %for.body386, %l44
  %inc532 = add nuw nsw i32 %add387, 1
  %exitcond649 = icmp eq i32 %inc532, %N
  br i1 %exitcond649, label %exit, label %for.body386

exit:
  ret void
}

attributes #0 = { nounwind "min-legal-vector-width"="0" "target-cpu"="generic" "target-features"="+armv8.1-m.main,+fp-armv8d16sp,+fp16,+fullfp16,+hwdiv,+lob,+mve.fp,+ras,+strict-align,+thumb-mode,+vfp2sp,+vfp3d16sp,+vfp4d16sp" "use-soft-float"="false" }
