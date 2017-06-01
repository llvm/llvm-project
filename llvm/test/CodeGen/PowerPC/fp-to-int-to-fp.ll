; RUN: llc -verify-machineinstrs -mcpu=a2 < %s | FileCheck %s -check-prefix=FPCVT
; RUN: llc -verify-machineinstrs -mcpu=ppc64 < %s | FileCheck %s -check-prefix=PPC64
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind readnone
define float @fool(float %X) #0 {
entry:
  %conv = fptosi float %X to i64
  %conv1 = sitofp i64 %conv to float
  ret float %conv1

; FPCVT-LABEL: @fool
; FPCVT: fctidz [[REG1:[0-9]+]], 1
; FPCVT: fcfids 1, [[REG1]]
; FPCVT: blr

; PPC64-LABEL: @fool
; PPC64: fctidz [[REG1:[0-9]+]], 1
; PPC64: fcfid [[REG2:[0-9]+]], [[REG1]]
; PPC64: frsp 1, [[REG2]]
; PPC64: blr
}

; Function Attrs: nounwind readnone
define double @foodl(double %X) #0 {
entry:
  %conv = fptosi double %X to i64
  %conv1 = sitofp i64 %conv to double
  ret double %conv1

; FPCVT-LABEL: @foodl
; FPCVT: fctidz [[REG1:[0-9]+]], 1
; FPCVT: fcfid 1, [[REG1]]
; FPCVT: blr

; PPC64-LABEL: @foodl
; PPC64: fctidz [[REG1:[0-9]+]], 1
; PPC64: fcfid 1, [[REG1]]
; PPC64: blr
}

; Function Attrs: nounwind readnone
define float @fooul(float %X) #0 {
entry:
  %conv = fptoui float %X to i64
  %conv1 = uitofp i64 %conv to float
  ret float %conv1

; FPCVT-LABEL: @fooul
; FPCVT: fctiduz [[REG1:[0-9]+]], 1
; FPCVT: fcfidus 1, [[REG1]]
; FPCVT: blr
}

; Function Attrs: nounwind readnone
define double @fooudl(double %X) #0 {
entry:
  %conv = fptoui double %X to i64
  %conv1 = uitofp i64 %conv to double
  ret double %conv1

; FPCVT-LABEL: @fooudl
; FPCVT: fctiduz [[REG1:[0-9]+]], 1
; FPCVT: fcfidu 1, [[REG1]]
; FPCVT: blr
}

attributes #0 = { nounwind readnone }

