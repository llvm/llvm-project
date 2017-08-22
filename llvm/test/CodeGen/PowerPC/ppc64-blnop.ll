; RUN: llc < %s -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s
; RUN: llc < %s -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 | FileCheck %s
; RUN: llc < %s -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 | FileCheck %s
; RUN: llc < %s -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s
; RUN: llc < %s -function-sections -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s -check-prefix=CHECK-FS
; RUN: llc < %s -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu | FileCheck %s
; RUN: llc < %s -function-sections -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu | FileCheck %s -check-prefix=CHECK-FS
; RUN: llc < %s -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN: -code-model=small -mcpu=pwr8 | FileCheck %s -check-prefix=SCM

%class.T = type { [2 x i8] }

define void @e_callee(%class.T* %this, i8* %c) { ret void }
define void @e_caller(%class.T* %this, i8* %c) {
  call void @e_callee(%class.T* %this, i8* %c)
  ret void

; CHECK-LABEL: e_caller:
; CHECK: bl e_callee
; CHECK-NEXT: nop

; CHECK-FS-LABEL: e_caller:
; CHECK-FS: bl e_callee
; CHECK-FS-NEXT: nop
}

define void @e_scallee(%class.T* %this, i8* %c) section "different" { ret void }
define void @e_scaller(%class.T* %this, i8* %c) {
  call void @e_scallee(%class.T* %this, i8* %c)
  ret void

; CHECK-LABEL: e_scaller:
; CHECK: bl e_scallee
; CHECK-NEXT: nop
}

define void @e_s2callee(%class.T* %this, i8* %c) { ret void }
define void @e_s2caller(%class.T* %this, i8* %c) section "different" {
  call void @e_s2callee(%class.T* %this, i8* %c)
  ret void

; CHECK-LABEL: e_s2caller:
; CHECK: bl e_s2callee
; CHECK-NEXT: nop
}

$cd1 = comdat any
$cd2 = comdat any

define void @e_ccallee(%class.T* %this, i8* %c) comdat($cd1) { ret void }
define void @e_ccaller(%class.T* %this, i8* %c) comdat($cd2) {
  call void @e_ccallee(%class.T* %this, i8* %c)
  ret void

; CHECK-LABEL: e_ccaller:
; CHECK: bl e_ccallee
; CHECK-NEXT: nop
}

$cd = comdat any

define void @e_c1callee(%class.T* %this, i8* %c) comdat($cd) { ret void }
define void @e_c1caller(%class.T* %this, i8* %c) comdat($cd) {
  call void @e_c1callee(%class.T* %this, i8* %c)
  ret void

; CHECK-LABEL: e_c1caller:
; CHECK: bl e_c1callee
; CHECK-NEXT: nop
}

define weak_odr hidden void @wo_hcallee(%class.T* %this, i8* %c) { ret void }
define void @wo_hcaller(%class.T* %this, i8* %c) {
  call void @wo_hcallee(%class.T* %this, i8* %c)
  ret void

; CHECK-LABEL: wo_hcaller:
; CHECK: bl wo_hcallee
; CHECK-NOT: nop

; SCM-LABEL: wo_hcaller:
; SCM:       bl wo_hcallee
; SCM-NEXT:  nop
}

define weak_odr protected void @wo_pcallee(%class.T* %this, i8* %c) { ret void }
define void @wo_pcaller(%class.T* %this, i8* %c) {
  call void @wo_pcallee(%class.T* %this, i8* %c)
  ret void

; CHECK-LABEL: wo_pcaller:
; CHECK: bl wo_pcallee
; CHECK-NOT: nop

; SCM-LABEL:   wo_pcaller:
; SCM:         bl wo_pcallee
; SCM-NEXT:    nop
}

define weak_odr void @wo_callee(%class.T* %this, i8* %c) { ret void }
define void @wo_caller(%class.T* %this, i8* %c) {
  call void @wo_callee(%class.T* %this, i8* %c)
  ret void

; CHECK-LABEL: wo_caller:
; CHECK: bl wo_callee
; CHECK-NEXT: nop
}

define weak protected void @w_pcallee(i8* %ptr) { ret void }
define void @w_pcaller(i8* %ptr) {
  call void @w_pcallee(i8* %ptr)
  ret void

; CHECK-LABEL: w_pcaller:
; CHECK: bl w_pcallee
; CHECK-NOT: nop

; SCM-LABEL: w_pcaller:
; SCM:       bl w_pcallee
; SCM-NEXT:  nop
}

define weak hidden void @w_hcallee(i8* %ptr) { ret void }
define void @w_hcaller(i8* %ptr) {
  call void @w_hcallee(i8* %ptr)
  ret void

; CHECK-LABEL: w_hcaller:
; CHECK: bl w_hcallee
; CHECK-NOT: nop

; SCM-LABEL: w_hcaller:
; SCM:       bl w_hcallee
; SCM-NEXT:  nop
}

define weak void @w_callee(i8* %ptr) { ret void }
define void @w_caller(i8* %ptr) {
  call void @w_callee(i8* %ptr)
  ret void

; CHECK-LABEL: w_caller:
; CHECK: bl w_callee
; CHECK-NEXT: nop
}

