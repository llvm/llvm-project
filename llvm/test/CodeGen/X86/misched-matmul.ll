; REQUIRES: asserts
; RUN: llc < %s -mtriple=x86_64-- -mcpu=core2 -pre-RA-sched=source -enable-misched -stats 2>&1 | FileCheck %s
;
; Verify that register pressure heuristics are working in MachineScheduler.
;
; We can further reduce spills in this case with a global register
; pressure heuristic, like sethi-ullman numbers or biasing toward
; scheduled subtrees. However, these heuristics are marginally
; beneficial on x86_64 and exacerbate register pressure in other
; more complex cases.
;
; CHECK: @wrap_mul4
; CHECK: 23 regalloc - Number of spills inserted

define void @wrap_mul4(double* nocapture %Out, [4 x double]* nocapture %A, [4 x double]* nocapture %B) #0 {
entry:
  %arrayidx1.i = getelementptr inbounds [4 x double], [4 x double]* %A, i64 0, i64 0
  %0 = load double, double* %arrayidx1.i, align 8
  %arrayidx3.i = getelementptr inbounds [4 x double], [4 x double]* %B, i64 0, i64 0
  %1 = load double, double* %arrayidx3.i, align 8
  %mul.i = fmul double %0, %1
  %arrayidx5.i = getelementptr inbounds [4 x double], [4 x double]* %A, i64 0, i64 1
  %2 = load double, double* %arrayidx5.i, align 8
  %arrayidx7.i = getelementptr inbounds [4 x double], [4 x double]* %B, i64 1, i64 0
  %3 = load double, double* %arrayidx7.i, align 8
  %mul8.i = fmul double %2, %3
  %add.i = fadd double %mul.i, %mul8.i
  %arrayidx10.i = getelementptr inbounds [4 x double], [4 x double]* %A, i64 0, i64 2
  %4 = load double, double* %arrayidx10.i, align 8
  %arrayidx12.i = getelementptr inbounds [4 x double], [4 x double]* %B, i64 2, i64 0
  %5 = load double, double* %arrayidx12.i, align 8
  %mul13.i = fmul double %4, %5
  %add14.i = fadd double %add.i, %mul13.i
  %arrayidx16.i = getelementptr inbounds [4 x double], [4 x double]* %A, i64 0, i64 3
  %6 = load double, double* %arrayidx16.i, align 8
  %arrayidx18.i = getelementptr inbounds [4 x double], [4 x double]* %B, i64 3, i64 0
  %7 = load double, double* %arrayidx18.i, align 8
  %mul19.i = fmul double %6, %7
  %add20.i = fadd double %add14.i, %mul19.i
  %arrayidx25.i = getelementptr inbounds [4 x double], [4 x double]* %B, i64 0, i64 1
  %8 = load double, double* %arrayidx25.i, align 8
  %mul26.i = fmul double %0, %8
  %arrayidx30.i = getelementptr inbounds [4 x double], [4 x double]* %B, i64 1, i64 1
  %9 = load double, double* %arrayidx30.i, align 8
  %mul31.i = fmul double %2, %9
  %add32.i = fadd double %mul26.i, %mul31.i
  %arrayidx36.i = getelementptr inbounds [4 x double], [4 x double]* %B, i64 2, i64 1
  %10 = load double, double* %arrayidx36.i, align 8
  %mul37.i = fmul double %4, %10
  %add38.i = fadd double %add32.i, %mul37.i
  %arrayidx42.i = getelementptr inbounds [4 x double], [4 x double]* %B, i64 3, i64 1
  %11 = load double, double* %arrayidx42.i, align 8
  %mul43.i = fmul double %6, %11
  %add44.i = fadd double %add38.i, %mul43.i
  %arrayidx49.i = getelementptr inbounds [4 x double], [4 x double]* %B, i64 0, i64 2
  %12 = load double, double* %arrayidx49.i, align 8
  %mul50.i = fmul double %0, %12
  %arrayidx54.i = getelementptr inbounds [4 x double], [4 x double]* %B, i64 1, i64 2
  %13 = load double, double* %arrayidx54.i, align 8
  %mul55.i = fmul double %2, %13
  %add56.i = fadd double %mul50.i, %mul55.i
  %arrayidx60.i = getelementptr inbounds [4 x double], [4 x double]* %B, i64 2, i64 2
  %14 = load double, double* %arrayidx60.i, align 8
  %mul61.i = fmul double %4, %14
  %add62.i = fadd double %add56.i, %mul61.i
  %arrayidx66.i = getelementptr inbounds [4 x double], [4 x double]* %B, i64 3, i64 2
  %15 = load double, double* %arrayidx66.i, align 8
  %mul67.i = fmul double %6, %15
  %add68.i = fadd double %add62.i, %mul67.i
  %arrayidx73.i = getelementptr inbounds [4 x double], [4 x double]* %B, i64 0, i64 3
  %16 = load double, double* %arrayidx73.i, align 8
  %mul74.i = fmul double %0, %16
  %arrayidx78.i = getelementptr inbounds [4 x double], [4 x double]* %B, i64 1, i64 3
  %17 = load double, double* %arrayidx78.i, align 8
  %mul79.i = fmul double %2, %17
  %add80.i = fadd double %mul74.i, %mul79.i
  %arrayidx84.i = getelementptr inbounds [4 x double], [4 x double]* %B, i64 2, i64 3
  %18 = load double, double* %arrayidx84.i, align 8
  %mul85.i = fmul double %4, %18
  %add86.i = fadd double %add80.i, %mul85.i
  %arrayidx90.i = getelementptr inbounds [4 x double], [4 x double]* %B, i64 3, i64 3
  %19 = load double, double* %arrayidx90.i, align 8
  %mul91.i = fmul double %6, %19
  %add92.i = fadd double %add86.i, %mul91.i
  %arrayidx95.i = getelementptr inbounds [4 x double], [4 x double]* %A, i64 1, i64 0
  %20 = load double, double* %arrayidx95.i, align 8
  %mul98.i = fmul double %1, %20
  %arrayidx100.i = getelementptr inbounds [4 x double], [4 x double]* %A, i64 1, i64 1
  %21 = load double, double* %arrayidx100.i, align 8
  %mul103.i = fmul double %3, %21
  %add104.i = fadd double %mul98.i, %mul103.i
  %arrayidx106.i = getelementptr inbounds [4 x double], [4 x double]* %A, i64 1, i64 2
  %22 = load double, double* %arrayidx106.i, align 8
  %mul109.i = fmul double %5, %22
  %add110.i = fadd double %add104.i, %mul109.i
  %arrayidx112.i = getelementptr inbounds [4 x double], [4 x double]* %A, i64 1, i64 3
  %23 = load double, double* %arrayidx112.i, align 8
  %mul115.i = fmul double %7, %23
  %add116.i = fadd double %add110.i, %mul115.i
  %mul122.i = fmul double %8, %20
  %mul127.i = fmul double %9, %21
  %add128.i = fadd double %mul122.i, %mul127.i
  %mul133.i = fmul double %10, %22
  %add134.i = fadd double %add128.i, %mul133.i
  %mul139.i = fmul double %11, %23
  %add140.i = fadd double %add134.i, %mul139.i
  %mul146.i = fmul double %12, %20
  %mul151.i = fmul double %13, %21
  %add152.i = fadd double %mul146.i, %mul151.i
  %mul157.i = fmul double %14, %22
  %add158.i = fadd double %add152.i, %mul157.i
  %mul163.i = fmul double %15, %23
  %add164.i = fadd double %add158.i, %mul163.i
  %mul170.i = fmul double %16, %20
  %mul175.i = fmul double %17, %21
  %add176.i = fadd double %mul170.i, %mul175.i
  %mul181.i = fmul double %18, %22
  %add182.i = fadd double %add176.i, %mul181.i
  %mul187.i = fmul double %19, %23
  %add188.i = fadd double %add182.i, %mul187.i
  %arrayidx191.i = getelementptr inbounds [4 x double], [4 x double]* %A, i64 2, i64 0
  %24 = load double, double* %arrayidx191.i, align 8
  %mul194.i = fmul double %1, %24
  %arrayidx196.i = getelementptr inbounds [4 x double], [4 x double]* %A, i64 2, i64 1
  %25 = load double, double* %arrayidx196.i, align 8
  %mul199.i = fmul double %3, %25
  %add200.i = fadd double %mul194.i, %mul199.i
  %arrayidx202.i = getelementptr inbounds [4 x double], [4 x double]* %A, i64 2, i64 2
  %26 = load double, double* %arrayidx202.i, align 8
  %mul205.i = fmul double %5, %26
  %add206.i = fadd double %add200.i, %mul205.i
  %arrayidx208.i = getelementptr inbounds [4 x double], [4 x double]* %A, i64 2, i64 3
  %27 = load double, double* %arrayidx208.i, align 8
  %mul211.i = fmul double %7, %27
  %add212.i = fadd double %add206.i, %mul211.i
  %mul218.i = fmul double %8, %24
  %mul223.i = fmul double %9, %25
  %add224.i = fadd double %mul218.i, %mul223.i
  %mul229.i = fmul double %10, %26
  %add230.i = fadd double %add224.i, %mul229.i
  %mul235.i = fmul double %11, %27
  %add236.i = fadd double %add230.i, %mul235.i
  %mul242.i = fmul double %12, %24
  %mul247.i = fmul double %13, %25
  %add248.i = fadd double %mul242.i, %mul247.i
  %mul253.i = fmul double %14, %26
  %add254.i = fadd double %add248.i, %mul253.i
  %mul259.i = fmul double %15, %27
  %add260.i = fadd double %add254.i, %mul259.i
  %mul266.i = fmul double %16, %24
  %mul271.i = fmul double %17, %25
  %add272.i = fadd double %mul266.i, %mul271.i
  %mul277.i = fmul double %18, %26
  %add278.i = fadd double %add272.i, %mul277.i
  %mul283.i = fmul double %19, %27
  %add284.i = fadd double %add278.i, %mul283.i
  %arrayidx287.i = getelementptr inbounds [4 x double], [4 x double]* %A, i64 3, i64 0
  %28 = load double, double* %arrayidx287.i, align 8
  %mul290.i = fmul double %1, %28
  %arrayidx292.i = getelementptr inbounds [4 x double], [4 x double]* %A, i64 3, i64 1
  %29 = load double, double* %arrayidx292.i, align 8
  %mul295.i = fmul double %3, %29
  %add296.i = fadd double %mul290.i, %mul295.i
  %arrayidx298.i = getelementptr inbounds [4 x double], [4 x double]* %A, i64 3, i64 2
  %30 = load double, double* %arrayidx298.i, align 8
  %mul301.i = fmul double %5, %30
  %add302.i = fadd double %add296.i, %mul301.i
  %arrayidx304.i = getelementptr inbounds [4 x double], [4 x double]* %A, i64 3, i64 3
  %31 = load double, double* %arrayidx304.i, align 8
  %mul307.i = fmul double %7, %31
  %add308.i = fadd double %add302.i, %mul307.i
  %mul314.i = fmul double %8, %28
  %mul319.i = fmul double %9, %29
  %add320.i = fadd double %mul314.i, %mul319.i
  %mul325.i = fmul double %10, %30
  %add326.i = fadd double %add320.i, %mul325.i
  %mul331.i = fmul double %11, %31
  %add332.i = fadd double %add326.i, %mul331.i
  %mul338.i = fmul double %12, %28
  %mul343.i = fmul double %13, %29
  %add344.i = fadd double %mul338.i, %mul343.i
  %mul349.i = fmul double %14, %30
  %add350.i = fadd double %add344.i, %mul349.i
  %mul355.i = fmul double %15, %31
  %add356.i = fadd double %add350.i, %mul355.i
  %mul362.i = fmul double %16, %28
  %mul367.i = fmul double %17, %29
  %add368.i = fadd double %mul362.i, %mul367.i
  %mul373.i = fmul double %18, %30
  %add374.i = fadd double %add368.i, %mul373.i
  %mul379.i = fmul double %19, %31
  %add380.i = fadd double %add374.i, %mul379.i
  store double %add20.i, double* %Out, align 8
  %Res.i.sroa.1.8.idx2 = getelementptr inbounds double, double* %Out, i64 1
  store double %add44.i, double* %Res.i.sroa.1.8.idx2, align 8
  %Res.i.sroa.2.16.idx4 = getelementptr inbounds double, double* %Out, i64 2
  store double %add68.i, double* %Res.i.sroa.2.16.idx4, align 8
  %Res.i.sroa.3.24.idx6 = getelementptr inbounds double, double* %Out, i64 3
  store double %add92.i, double* %Res.i.sroa.3.24.idx6, align 8
  %Res.i.sroa.4.32.idx8 = getelementptr inbounds double, double* %Out, i64 4
  store double %add116.i, double* %Res.i.sroa.4.32.idx8, align 8
  %Res.i.sroa.5.40.idx10 = getelementptr inbounds double, double* %Out, i64 5
  store double %add140.i, double* %Res.i.sroa.5.40.idx10, align 8
  %Res.i.sroa.6.48.idx12 = getelementptr inbounds double, double* %Out, i64 6
  store double %add164.i, double* %Res.i.sroa.6.48.idx12, align 8
  %Res.i.sroa.7.56.idx14 = getelementptr inbounds double, double* %Out, i64 7
  store double %add188.i, double* %Res.i.sroa.7.56.idx14, align 8
  %Res.i.sroa.8.64.idx16 = getelementptr inbounds double, double* %Out, i64 8
  store double %add212.i, double* %Res.i.sroa.8.64.idx16, align 8
  %Res.i.sroa.9.72.idx18 = getelementptr inbounds double, double* %Out, i64 9
  store double %add236.i, double* %Res.i.sroa.9.72.idx18, align 8
  %Res.i.sroa.10.80.idx20 = getelementptr inbounds double, double* %Out, i64 10
  store double %add260.i, double* %Res.i.sroa.10.80.idx20, align 8
  %Res.i.sroa.11.88.idx22 = getelementptr inbounds double, double* %Out, i64 11
  store double %add284.i, double* %Res.i.sroa.11.88.idx22, align 8
  %Res.i.sroa.12.96.idx24 = getelementptr inbounds double, double* %Out, i64 12
  store double %add308.i, double* %Res.i.sroa.12.96.idx24, align 8
  %Res.i.sroa.13.104.idx26 = getelementptr inbounds double, double* %Out, i64 13
  store double %add332.i, double* %Res.i.sroa.13.104.idx26, align 8
  %Res.i.sroa.14.112.idx28 = getelementptr inbounds double, double* %Out, i64 14
  store double %add356.i, double* %Res.i.sroa.14.112.idx28, align 8
  %Res.i.sroa.15.120.idx30 = getelementptr inbounds double, double* %Out, i64 15
  store double %add380.i, double* %Res.i.sroa.15.120.idx30, align 8
  ret void
}

attributes #0 = { noinline nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
