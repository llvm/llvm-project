target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-linux-gnu"

define void @foo(i64 %var_4, i1 %tobool.not.us) #0 {
entry:
  br label %for.body.us

for.body.us:                                      ; preds = %for.body.us, %entry
  %0 = insertelement <16 x i64> poison, i64 %var_4, i64 0
  %1 = icmp slt <16 x i64> %0, zeroinitializer
  %2 = bitcast <16 x i1> %1 to i16
  %3 = icmp ne i16 %2, 0
  %4 = insertelement <16 x i64> %0, i64 0, i64 0
  %5 = insertelement <16 x i64> %4, i64 1, i64 2
  %6 = icmp slt <16 x i64> %4, zeroinitializer
  %7 = bitcast <16 x i1> %6 to i16
  %8 = icmp ne i16 %7, 0
  %9 = insertelement <16 x i64> %5, i64 0, i64 3
  %10 = icmp slt <16 x i64> %9, zeroinitializer
  %11 = bitcast <16 x i1> %10 to i16
  %12 = icmp ne i16 %11, 0
  br i1 %12, label %for.body.us, label %for.cond13.preheader.us

for.cond13.preheader.us:                          ; preds = %for.body.us
  br i1 %tobool.not.us, label %for.body23.us.us484.us.us.us.preheader, label %for.body23.us.us499.us.us.us.preheader

for.body23.us.us499.us.us.us.preheader:           ; preds = %for.cond13.preheader.us
  br i1 %8, label %for.body30.us363.us449.us.us.us.us, label %vector.ph4767

for.body23.us.us484.us.us.us.preheader:           ; preds = %for.cond13.preheader.us
  %op.rdx6735 = or i1 false, %3
  br i1 %12, label %for.body30.us351.us.us.us611.us.us, label %vector.ph3672

vector.ph3672:                                    ; preds = %for.body23.us.us484.us.us.us.preheader
  ret void

for.body30.us351.us.us.us611.us.us:               ; preds = %for.body23.us.us484.us.us.us.preheader
  ret void

vector.ph4767:                                    ; preds = %for.body23.us.us499.us.us.us.preheader
  ret void

for.body30.us363.us449.us.us.us.us:               ; preds = %for.body23.us.us499.us.us.us.preheader
  ret void
}

attributes #0 = { "target-features"="+v" }