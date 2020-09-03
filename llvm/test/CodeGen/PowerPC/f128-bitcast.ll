; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown -verify-machineinstrs \
; RUN:   -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | FileCheck %s
; RUN: llc -mcpu=pwr9 -mtriple=powerpc64-unknown-unknown -verify-machineinstrs \
; RUN:   -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | \
; RUN:   FileCheck %s --check-prefix=CHECK-BE

; Function Attrs: norecurse nounwind readnone
define i64 @getPart1(fp128 %in) local_unnamed_addr {
entry:
  %0 = bitcast fp128 %in to i128
  %a.sroa.0.0.extract.trunc = trunc i128 %0 to i64
  ret i64 %a.sroa.0.0.extract.trunc
; CHECK-LABEL: getPart1
; CHECK:       mfvsrld r3, v2
; CHECK-NEXT:  blr
; CHECK-BE-LABEL: getPart1
; CHECK-BE:       mfvsrld r3, v2
; CHECK-BE-NEXT:  blr
}

; Function Attrs: norecurse nounwind readnone
define i64 @getPart2(fp128 %in) local_unnamed_addr {
entry:
  %0 = bitcast fp128 %in to i128
  %a.sroa.0.8.extract.shift = lshr i128 %0, 64
  %a.sroa.0.8.extract.trunc = trunc i128 %a.sroa.0.8.extract.shift to i64
  ret i64 %a.sroa.0.8.extract.trunc
; CHECK-LABEL: getPart2
; CHECK:       mfvsrd r3, v2
; CHECK-NEXT:  blr
; CHECK-BE-LABEL: getPart2
; CHECK-BE:       mfvsrd r3, v2
; CHECK-BE-NEXT:  blr
}

; Function Attrs: norecurse nounwind readnone
define i64 @checkBitcast(fp128 %in, <2 x i64> %in2, <2 x i64> *%out) local_unnamed_addr {
entry:
  %0 = bitcast fp128 %in to <2 x i64>
  %1 = extractelement <2 x i64> %0, i64 0
  %2 = add <2 x i64> %0, %in2
  store <2 x i64> %2, <2 x i64> *%out, align 16
  ret i64 %1
; CHECK-LABEL: checkBitcast
; CHECK:       mfvsrld r3, v2
; CHECK:       blr
; CHECK-BE-LABEL: checkBitcast
; CHECK-BE:       mfvsrd r3, v2
; CHECK-BE:       blr
}

