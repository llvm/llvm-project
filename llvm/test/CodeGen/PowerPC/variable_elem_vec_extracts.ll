; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64le-unknown-unknown -ppc-late-peephole=true < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64-unknown-unknown < %s | FileCheck %s \
; RUN:  --check-prefix=CHECK-BE
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple=powerpc64-unknown-unknown < %s | FileCheck %s \
; RUN:  --check-prefix=CHECK-P7

; Function Attrs: norecurse nounwind readnone
define signext i32 @geti(<4 x i32> %a, i32 signext %b) {
entry:
  %vecext = extractelement <4 x i32> %a, i32 %b
  ret i32 %vecext
; CHECK-LABEL: @geti
; CHECK-P7-LABEL: @geti
; CHECK-BE-LABEL: @geti
; CHECK-DAG: li [[TRUNCREG:[0-9]+]], 2
; CHECK-DAG: andc [[MASKREG:[0-9]+]], [[TRUNCREG]], 5
; CHECK-DAG: sldi [[SHIFTREG:[0-9]+]], [[MASKREG]], 2
; CHECK-DAG: lvsl [[SHMSKREG:[0-9]+]], 0, [[SHIFTREG]]
; CHECK-DAG: vperm [[PERMVEC:[0-9]+]], 2, 2, [[SHMSKREG]]
; CHECK-DAG: li [[ONEREG:[0-9]+]], 1
; CHECK-DAG: and [[ELEMSREG:[0-9]+]], [[ONEREG]], 5
; CHECK-DAG: sldi [[SHAMREG:[0-9]+]], [[ELEMSREG]], 5
; CHECK: mfvsrd [[TOGPR:[0-9]+]],
; CHECK: srd [[RSHREG:[0-9]+]], [[TOGPR]], [[SHAMREG]]
; CHECK: extsw 3, [[RSHREG]]
; CHECK-P7-DAG: rlwinm [[ELEMOFFREG:[0-9]+]], 5, 2, 28, 29
; CHECK-P7-DAG: stxvw4x 34,
; CHECK-P7: lwax 3, 3, [[ELEMOFFREG]]
; CHECK-BE-DAG: andi. [[ANDREG:[0-9]+]], 5, 2
; CHECK-BE-DAG: sldi [[SLREG:[0-9]+]], [[ANDREG]], 2
; CHECK-BE-DAG: lvsl [[SHMSKREG:[0-9]+]], 0, [[SLREG]]
; CHECK-BE-DAG: vperm {{[0-9]+}}, 2, 2, [[SHMSKREG]]
; CHECK-BE-DAG: li [[IMMREG:[0-9]+]], 1
; CHECK-BE-DAG: andc [[ANDCREG:[0-9]+]], [[IMMREG]], 5
; CHECK-BE-DAG: sldi [[SHAMREG:[0-9]+]], [[ANDCREG]], 5
; CHECK-BE: mfvsrd [[TOGPR:[0-9]+]],
; CHECK-BE: srd [[RSHREG:[0-9]+]], [[TOGPR]], [[SHAMREG]]
; CHECK-BE: extsw 3, [[RSHREG]]
}

; Function Attrs: norecurse nounwind readnone
define i64 @getl(<2 x i64> %a, i32 signext %b) {
entry:
  %vecext = extractelement <2 x i64> %a, i32 %b
  ret i64 %vecext
; CHECK-LABEL: @getl
; CHECK-P7-LABEL: @getl
; CHECK-BE-LABEL: @getl
; CHECK-DAG: li [[TRUNCREG:[0-9]+]], 1
; CHECK-DAG: andc [[MASKREG:[0-9]+]], [[TRUNCREG]], 5
; CHECK-DAG: sldi [[SHIFTREG:[0-9]+]], [[MASKREG]], 3
; CHECK-DAG: lvsl [[SHMSKREG:[0-9]+]], 0, [[SHIFTREG]]
; CHECK-DAG: vperm [[PERMVEC:[0-9]+]], 2, 2, [[SHMSKREG]]
; CHECK: mfvsrd 3,
; CHECK-P7-DAG: rlwinm [[ELEMOFFREG:[0-9]+]], 5, 3, 28, 28
; CHECK-P7-DAG: stxvd2x 34,
; CHECK-P7: ldx 3, 3, [[ELEMOFFREG]]
; CHECK-BE-DAG: andi. [[ANDREG:[0-9]+]], 5, 1
; CHECK-BE-DAG: sldi [[SLREG:[0-9]+]], [[ANDREG]], 3
; CHECK-BE-DAG: lvsl [[SHMSKREG:[0-9]+]], 0, [[SLREG]]
; CHECK-BE-DAG: vperm {{[0-9]+}}, 2, 2, [[SHMSKREG]]
; CHECK-BE: mfvsrd 3,
}

; Function Attrs: norecurse nounwind readnone
define float @getf(<4 x float> %a, i32 signext %b) {
entry:
  %vecext = extractelement <4 x float> %a, i32 %b
  ret float %vecext
; CHECK-LABEL: @getf
; CHECK-P7-LABEL: @getf
; CHECK-BE-LABEL: @getf
; CHECK: xori [[TRUNCREG:[0-9]+]], 5, 3
; CHECK: sldi [[SHIFTREG:[0-9]+]], [[TRUNCREG]], 2
; CHECK: lvsl [[SHMSKREG:[0-9]+]], 0, [[SHIFTREG]]
; CHECK: vperm {{[0-9]+}}, 2, 2, [[SHMSKREG]]
; CHECK: xscvspdpn 1,
; CHECK-P7-DAG: rlwinm [[ELEMOFFREG:[0-9]+]], 5, 2, 28, 29
; CHECK-P7-DAG: stxvw4x 34,
; CHECK-P7: lfsx 1, 3, [[ELEMOFFREG]]
; CHECK-BE: sldi [[ELNOREG:[0-9]+]], 5, 2
; CHECK-BE: lvsl [[SHMSKREG:[0-9]+]], 0, [[ELNOREG]]
; CHECK-BE: vperm {{[0-9]+}}, 2, 2, [[SHMSKREG]]
; CHECK-BE: xscvspdpn 1,
}

; Function Attrs: norecurse nounwind readnone
define double @getd(<2 x double> %a, i32 signext %b) {
entry:
  %vecext = extractelement <2 x double> %a, i32 %b
  ret double %vecext
; CHECK-LABEL: @getd
; CHECK-P7-LABEL: @getd
; CHECK-BE-LABEL: @getd
; CHECK: li [[TRUNCREG:[0-9]+]], 1
; CHECK: andc [[MASKREG:[0-9]+]], [[TRUNCREG]], 5
; CHECK: sldi [[SHIFTREG:[0-9]+]], [[MASKREG]], 3
; CHECK: lvsl [[SHMSKREG:[0-9]+]], 0, [[SHIFTREG]]
; CHECK: vperm {{[0-9]+}}, 2, 2, [[SHMSKREG]]
; FIXME: the instruction below is a redundant regclass copy, to be removed
; CHECK: xxlor 1,
; CHECK-P7-DAG: andi. [[ANDREG:[0-9]+]], 5, 1
; CHECK-P7-DAG: sldi [[SLREG:[0-9]+]], [[ANDREG]], 3
; CHECK-P7-DAG: lvsl [[SHMSKREG:[0-9]+]], 0, [[SLREG]]
; CHECK-P7-DAG: vperm {{[0-9]+}}, 2, 2, [[SHMSKREG]]
; FIXME: the instruction below is a redundant regclass copy, to be removed
; CHECK-P7: xxlor 1,
; CHECK-BE-DAG: andi. [[ANDREG:[0-9]+]], 5, 1
; CHECK-BE-DAG: sldi [[SLREG:[0-9]+]], [[ANDREG]], 3
; CHECK-BE-DAG: lvsl [[SHMSKREG:[0-9]+]], 0, [[SLREG]]
; CHECK-BE-DAG: vperm {{[0-9]+}}, 2, 2, [[SHMSKREG]]
; FIXME: the instruction below is a redundant regclass copy, to be removed
; CHECK-BE: xxlor 1,
}
