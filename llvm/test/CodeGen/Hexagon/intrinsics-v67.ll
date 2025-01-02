; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: @t1
; CHECK: r{{[0-9]+}}:{{[0-9]+}} += dfmpylh(r{{[0-9]+}}:{{[0-9]+}},r{{[0-9]+}}:{{[0-9]+}})
define dso_local double @t1(double %a, double %b, double %c) local_unnamed_addr #0 {
entry:
  %0 = tail call double @llvm.hexagon.F2.dfmpylh(double %a, double %b, double %c) #2
  ret double %0
}

declare double @llvm.hexagon.F2.dfmpylh(double, double, double) #1

; CHECK-LABEL: @t2
; CHECK: r{{[0-9]+}}:{{[0-9]+}} += dfmpyhh(r{{[0-9]+}}:{{[0-9]+}},r{{[0-9]+}}:{{[0-9]+}})
define dso_local double @t2(double %a, double %b, double %c) local_unnamed_addr #0 {
entry:
  %0 = tail call double @llvm.hexagon.F2.dfmpyhh(double %a, double %b, double %c) #2
  ret double %0
}

declare double @llvm.hexagon.F2.dfmpyhh(double, double, double) #1

; CHECK-LABEL: @t3
; CHECK: r{{[0-9]+}}:{{[0-9]+}} = dfmax(r{{[0-9]+}}:{{[0-9]+}},r{{[0-9]+}}:{{[0-9]+}})
define dso_local double @t3(double %a, double %b) local_unnamed_addr #0 {
entry:
  %0 = tail call double @llvm.hexagon.F2.dfmax(double %a, double %b) #2
  ret double %0
}

declare double @llvm.hexagon.F2.dfmax(double, double) #1

; CHECK-LABEL: @t4
; CHECK: r{{[0-9]+}}:{{[0-9]+}} = dfmin(r{{[0-9]+}}:{{[0-9]+}},r{{[0-9]+}}:{{[0-9]+}})
define dso_local double @t4(double %a, double %b) local_unnamed_addr #0 {
entry:
  %0 = tail call double @llvm.hexagon.F2.dfmin(double %a, double %b) #2
  ret double %0
}

declare double @llvm.hexagon.F2.dfmin(double, double) #1

; CHECK-LABEL: @t5
; CHECK: r{{[0-9]+}}:{{[0-9]+}} = dfmpyfix(r{{[0-9]+}}:{{[0-9]+}},r{{[0-9]+}}:{{[0-9]+}})
define dso_local double @t5(double %a, double %b) local_unnamed_addr #0 {
entry:
  %0 = tail call double @llvm.hexagon.F2.dfmpyfix(double %a, double %b) #2
  ret double %0
}

declare double @llvm.hexagon.F2.dfmpyfix(double, double) #1

; CHECK-LABEL: @t6
; CHECK: r{{[0-9]+}}:{{[0-9]+}} = dfmpyll(r{{[0-9]+}}:{{[0-9]+}},r{{[0-9]+}}:{{[0-9]+}})
define dso_local double @t6(double %a, double %b) local_unnamed_addr #0 {
entry:
  %0 = tail call double @llvm.hexagon.F2.dfmpyll(double %a, double %b) #2
  ret double %0
}

declare double @llvm.hexagon.F2.dfmpyll(double, double) #1

; CHECK-LABEL: @t7
; CHECK:r{{[0-9]+}}:{{[0-9]+}} = cmpyrw(r{{[0-9]+}}:{{[0-9]+}},r{{[0-9]+}}:{{[0-9]+}}*)
define dso_local i64 @t7(i64 %a, i64 %b) local_unnamed_addr #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M7.vdmpy(i64 %a, i64 %b)
  ret i64 %0
}

declare i64 @llvm.hexagon.M7.vdmpy(i64, i64) #1

; CHECK-LABEL: @t8
; CHECK:r{{[0-9]+}}:{{[0-9]+}} += cmpyrw(r{{[0-9]+}}:{{[0-9]+}},r{{[0-9]+}}:{{[0-9]+}}*)
define dso_local i64 @t8(i64 %a, i64 %b, i64 %c) local_unnamed_addr #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M7.vdmpy.acc(i64 %a, i64 %b, i64 %c)
  ret i64 %0
}

declare i64 @llvm.hexagon.M7.vdmpy.acc(i64, i64, i64) #1

; CHECK-LABEL: @t9
; CHECK: r1:0 = cmpyrw(r1:0,r3:2)
define i64 @t9(i64 %rss, i64 %rtt) local_unnamed_addr #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M7.dcmpyrw(i64 %rss, i64 %rtt)
  ret i64 %0
}

declare i64 @llvm.hexagon.M7.dcmpyrw(i64, i64) #1

; CHECK-LABEL: @t10
; CHECK: r1:0 += cmpyrw(r3:2,r5:4)
define i64 @t10(i64 %rxx, i64 %rss, i64 %rtt) local_unnamed_addr #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M7.dcmpyrw.acc(i64 %rxx, i64 %rss, i64 %rtt)
  ret i64 %0
}

declare i64 @llvm.hexagon.M7.dcmpyrw.acc(i64, i64, i64) #1

; CHECK-LABEL: @t11
; CHECK: r1:0 = cmpyrw(r1:0,r3:2*)
define i64 @t11(i64 %rss, i64 %rtt) local_unnamed_addr #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M7.dcmpyrwc(i64 %rss, i64 %rtt)
  ret i64 %0
}

declare i64 @llvm.hexagon.M7.dcmpyrwc(i64, i64) #1

; CHECK-LABEL: @t12
; CHECK: r1:0 += cmpyrw(r3:2,r5:4*)
define i64 @t12(i64 %rxx, i64 %rss, i64 %rtt) local_unnamed_addr #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M7.dcmpyrwc.acc(i64 %rxx, i64 %rss, i64 %rtt)
  ret i64 %0
}

declare i64 @llvm.hexagon.M7.dcmpyrwc.acc(i64, i64, i64) #1

; CHECK-LABEL: @t13
; CHECK: r1:0 = cmpyiw(r1:0,r3:2)
define i64 @t13(i64 %rss, i64 %rtt) local_unnamed_addr #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M7.dcmpyiw(i64 %rss, i64 %rtt)
  ret i64 %0
}

declare i64 @llvm.hexagon.M7.dcmpyiw(i64, i64) #1

; CHECK-LABEL: @t14
; CHECK: r1:0 += cmpyiw(r3:2,r5:4)
define i64 @t14(i64 %rxx, i64 %rss, i64 %rtt) local_unnamed_addr #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M7.dcmpyiw.acc(i64 %rxx, i64 %rss, i64 %rtt)
  ret i64 %0
}

declare i64 @llvm.hexagon.M7.dcmpyiw.acc(i64, i64, i64) #1

; CHECK-LABEL: @t15
; CHECK: r1:0 = cmpyiw(r1:0,r3:2*)
define i64 @t15(i64 %rss, i64 %rtt) local_unnamed_addr #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M7.dcmpyiwc(i64 %rss, i64 %rtt)
  ret i64 %0
}

declare i64 @llvm.hexagon.M7.dcmpyiwc(i64, i64) #1

; CHECK-LABEL: @t16
; CHECK: r1:0 += cmpyiw(r3:2,r5:4*)
define i64 @t16(i64 %rxx, i64 %rss, i64 %rtt) local_unnamed_addr #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.M7.dcmpyiwc.acc(i64 %rxx, i64 %rss, i64 %rtt)
  ret i64 %0
}

declare i64 @llvm.hexagon.M7.dcmpyiwc.acc(i64, i64, i64) #1

; CHECK-LABEL: @t17
; CHECK: r0 = cmpyrw(r1:0,r3:2):<<1:sat
define i32 @t17(i64 %rss, i64 %rtt) local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M7.wcmpyrw(i64 %rss, i64 %rtt)
  ret i32 %0
}

declare i32 @llvm.hexagon.M7.wcmpyrw(i64, i64) #1

; CHECK-LABEL: @t18
; CHECK: r0 = cmpyrw(r1:0,r3:2*):<<1:sat
define i32 @t18(i64 %rss, i64 %rtt) local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M7.wcmpyrwc(i64 %rss, i64 %rtt)
  ret i32 %0
}

declare i32 @llvm.hexagon.M7.wcmpyrwc(i64, i64) #1

; CHECK-LABEL: @t19
; CHECK: r0 = cmpyiw(r1:0,r3:2):<<1:sat
define i32 @t19(i64 %rss, i64 %rtt) local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M7.wcmpyiw(i64 %rss, i64 %rtt)
  ret i32 %0
}

declare i32 @llvm.hexagon.M7.wcmpyiw(i64, i64) #1

; CHECK-LABEL: @t20
; CHECK: r0 = cmpyiw(r1:0,r3:2*):<<1:sat
define i32 @t20(i64 %rss, i64 %rtt) local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M7.wcmpyiwc(i64 %rss, i64 %rtt)
  ret i32 %0
}

declare i32 @llvm.hexagon.M7.wcmpyiwc(i64, i64) #1

; CHECK-LABEL: @t21
; CHECK: r0 = cmpyrw(r1:0,r3:2):<<1:rnd:sat
define i32 @t21(i64 %rss, i64 %rtt) local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M7.wcmpyrw.rnd(i64 %rss, i64 %rtt)
  ret i32 %0
}

declare i32 @llvm.hexagon.M7.wcmpyrw.rnd(i64, i64) #1

; CHECK-LABEL: @t22
; CHECK: r0 = cmpyrw(r1:0,r3:2*):<<1:rnd:sat
define i32 @t22(i64 %rss, i64 %rtt) local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M7.wcmpyrwc.rnd(i64 %rss, i64 %rtt)
  ret i32 %0
}

declare i32 @llvm.hexagon.M7.wcmpyrwc.rnd(i64, i64) #1

; CHECK-LABEL: @t23
; CHECK: r0 = cmpyiw(r1:0,r3:2):<<1:rnd:sat
define i32 @t23(i64 %rss, i64 %rtt) local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M7.wcmpyiw.rnd(i64 %rss, i64 %rtt)
  ret i32 %0
}

declare i32 @llvm.hexagon.M7.wcmpyiw.rnd(i64, i64) #1

; CHECK-LABEL: @t24
; CHECK: r0 = cmpyiw(r1:0,r3:2*):<<1:rnd:sat
define i32 @t24(i64 %rss, i64 %rtt) local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.M7.wcmpyiwc.rnd(i64 %rss, i64 %rtt)
  ret i32 %0
}

declare i32 @llvm.hexagon.M7.wcmpyiwc.rnd(i64, i64) #1

; CHECK-LABEL: @t25
; CHECK: r1:0 = cround(r1:0,#0)
define i64 @t25(i64 %rss) local_unnamed_addr #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.A7.croundd.ri(i64 %rss, i32 0)
  ret i64 %0
}

declare i64 @llvm.hexagon.A7.croundd.ri(i64, i32) #1

; CHECK-LABEL: @t26
; CHECK: r1:0 = cround(r1:0,r2)
define i64 @t26(i64 %rss, i32 %rt) local_unnamed_addr #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.A7.croundd.rr(i64 %rss, i32 %rt)
  ret i64 %0
}

declare i64 @llvm.hexagon.A7.croundd.rr(i64, i32) #1

; CHECK-LABEL: @t27
; CHECK: r0 = clip(r0,#0)
define i32 @t27(i32 %rs) local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.A7.clip(i32 %rs, i32 0)
  ret i32 %0
}

declare i32 @llvm.hexagon.A7.clip(i32, i32) #1

; CHECK-LABEL: @t28
; CHECK: r1:0 = vclip(r1:0,#0)
define i64 @t28(i64 %rs) local_unnamed_addr #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.A7.vclip(i64 %rs, i32 0)
  ret i64 %0
}

declare i64 @llvm.hexagon.A7.vclip(i64, i32) #1

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv67" "target-features"="-long-calls" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { readnone }
attributes #2 = { nounwind }
