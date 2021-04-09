; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-- \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-- \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names < %s | FileCheck %s

; CHECK-NOT: Impossible reg-to-reg copy
; CHECK-LABEL: test_xxsplti32dx
; CHECK:         xxsplti32dx

;; Test reduced from larger application where bug was initially detected.
;; This test checks that the correct register class is used for xxsplti32dx.

declare dso_local void @callee() local_unnamed_addr

define dso_local void @test_xxsplti32dx() local_unnamed_addr {
entry:
  %i1 = load double, double* undef, align 8
  br label %for.body124

for.body124:
  %E0 = phi double [ 0.000000e+00, %entry ], [ %E1, %for.end1072 ]
  br i1 undef, label %for.body919.preheader, label %for.end1072

for.body919.preheader:
  %i4 = load double, double* null, align 8
  %i5 = load double, double* null, align 8
  %i15 = insertelement <2 x double> poison, double %i5, i32 0
  %i23 = insertelement <2 x double> undef, double %i4, i32 1
  %i24 = insertelement <2 x double> %i15, double 0x3FC5555555555555, i32 1
  %i25 = fmul fast <2 x double> %i23, %i24
  %mul986 = extractelement <2 x double> %i25, i32 1
  %sub994 = fsub fast double %E0, %mul986
  br label %for.end1072

for.end1072:
  %E1 = phi double [ %E0, %for.body124 ], [ %sub994, %for.body919.preheader ]
  %i28 = phi <2 x double> [ zeroinitializer, %for.body124 ], [ %i15, %for.body919.preheader ]
  tail call void @callee()
  store <2 x double> %i28, <2 x double>* undef, align 8
  br label %for.body124
}
