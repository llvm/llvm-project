; Test the Test Data Class instruction, as used by fpclassify.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
;

declare float @llvm.fabs.f32(float)
declare double @llvm.fabs.f64(double)
declare fp128 @llvm.fabs.f128(fp128)

define i32 @fpc(double %x) {
entry:
; CHECK-LABEL: fpc
; CHECK: lhi %r2, 5
; CHECK: ltdbr %f0, %f0
; CHECK: je [[RET:.L.*]]
  %testeq = fcmp oeq double %x, 0.000000e+00
  br i1 %testeq, label %ret, label %nonzero, !prof !1

nonzero:
; CHECK: lhi %r2, 1
; CHECK: cdbr %f0, %f0
; CHECK: jo [[RET]]
  %testnan = fcmp uno double %x, 0.000000e+00
  br i1 %testnan, label %ret, label %nonzeroord, !prof !1

nonzeroord:
; CHECK: lhi %r2, 2
; CHECK: tcdb %f0, 48
; CHECK: jl [[RET]]
  %abs = tail call double @llvm.fabs.f64(double %x)
  %testinf = fcmp oeq double %abs, 0x7FF0000000000000
  br i1 %testinf, label %ret, label %finite, !prof !1

finite:
; CHECK: lhi %r2, 3
; CHECK: tcdb %f0, 831
; CHECK: blr %r14
; CHECK: lhi %r2, 4
  %testnormal = fcmp uge double %abs, 0x10000000000000
  %finres = select i1 %testnormal, i32 3, i32 4
  br label %ret

ret:
; CHECK: [[RET]]:
; CHECK: br %r14
  %res = phi i32 [ 5, %entry ], [ 1, %nonzero ], [ 2, %nonzeroord ], [ %finres, %finite ]
  ret i32 %res
}

!1 = !{!"branch_weights", i32 1, i32 1}
