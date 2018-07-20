; RUN: llc -mtriple=arm64-apple-ios   -mcpu=cyclone   < %s | FileCheck %s -check-prefixes=ALL,CYCLONE
; RUN: llc -mtriple=arm64-apple-ios   -mcpu=cyclone -mattr=+fullfp16 < %s | FileCheck %s -check-prefixes=CYCLONE-FULLFP16
; RUN: llc -mtriple=aarch64-gnu-linux -mcpu=exynos-m1 < %s | FileCheck %s -check-prefixes=ALL,OTHERS
; RUN: llc -mtriple=aarch64-gnu-linux -mcpu=exynos-m3 < %s | FileCheck %s -check-prefixes=ALL,OTHERS
; RUN: llc -mtriple=aarch64-gnu-linux -mcpu=kryo      < %s | FileCheck %s -check-prefixes=ALL,OTHERS
; RUN: llc -mtriple=aarch64-gnu-linux -mcpu=falkor    < %s | FileCheck %s -check-prefixes=ALL,OTHERS

declare void @bar(half, float, double, <2 x double>)
declare void @bari(i32, i32)
declare void @barl(i64, i64)
declare void @barf(float, float)

define void @t1() nounwind ssp {
entry:
; ALL-LABEL: t1:
; ALL-NOT: fmov
; ALL:     ldr h0,{{.*}}
; CYCLONE: fmov s1, wzr
; CYCLONE: fmov d2, xzr
; CYCLONE: movi.16b v3, #0
; CYCLONE-FULLFP16: fmov h0, wzr
; CYCLONE-FULLFP16: fmov s1, wzr
; CYCLONE-FULLFP16: fmov d2, xzr
; CYCLONE-FULLFP16: movi.16b v3, #0
; OTHERS: movi v{{[0-3]+}}.2d, #0000000000000000
; OTHERS: movi v{{[0-3]+}}.2d, #0000000000000000
; OTHERS: movi v{{[0-3]+}}.2d, #0000000000000000
  tail call void @bar(half 0.000000e+00, float 0.000000e+00, double 0.000000e+00, <2 x double> <double 0.000000e+00, double 0.000000e+00>) nounwind
  ret void
}

define void @t2() nounwind ssp {
entry:
; ALL-LABEL: t2:
; ALL-NOT: mov w0, wzr
; ALL: mov w{{[0-3]+}}, #0
; ALL: mov w{{[0-3]+}}, #0
  tail call void @bari(i32 0, i32 0) nounwind
  ret void
}

define void @t3() nounwind ssp {
entry:
; ALL-LABEL: t3:
; ALL-NOT: mov x0, xzr
; ALL: mov x{{[0-3]+}}, #0
; ALL: mov x{{[0-3]+}}, #0
  tail call void @barl(i64 0, i64 0) nounwind
  ret void
}

define void @t4() nounwind ssp {
; ALL-LABEL: t4:
; ALL-NOT: fmov
; CYCLONE: fmov s{{[0-3]+}}, wzr
; CYCLONE: fmov s{{[0-3]+}}, wzr
; CYCLONE-FULLFP16: fmov s{{[0-3]+}}, wzr
; CYCLONE-FULLFP16: fmov s{{[0-3]+}}, wzr
; OTHERS: movi v{{[0-3]+}}.2d, #0000000000000000
; OTHERS: movi v{{[0-3]+}}.2d, #0000000000000000
  tail call void @barf(float 0.000000e+00, float 0.000000e+00) nounwind
  ret void
}

; We used to produce spills+reloads for a Q register with zero cycle zeroing
; enabled.
; ALL-LABEL: foo:
; ALL-NOT: str q{{[0-9]+}}
; ALL-NOT: ldr q{{[0-9]+}}
define double @foo(i32 %n) {
entry:
  br label %for.body

for.body:
  %phi0 = phi double [ 1.0, %entry ], [ %v0, %for.body ]
  %i.076 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %conv21 = sitofp i32 %i.076 to double
  %call = tail call fast double @sin(double %conv21)
  %cmp.i = fcmp fast olt double %phi0, %call
  %v0 = select i1 %cmp.i, double %call, double %phi0
  %inc = add nuw nsw i32 %i.076, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret double %v0
}

define <2 x i64> @t6() {
; ALL-LABEL: t6:
; CYCLONE: movi.16b v0, #0
; OTHERS: movi v0.2d, #0000000000000000
 ret <2 x i64> zeroinitializer
}


declare double @sin(double)
