; RUN: llc < %s -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s
; RUN: llc < %s -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 | FileCheck %s
; RUN: llc < %s -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 | FileCheck %s
; RUN: llc < %s -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s
; RUN: llc < %s -relocation-model=pic -function-sections -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s -check-prefix=CHECK-FS
; RUN: llc < %s -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu | FileCheck %s
; RUN: llc < %s -relocation-model=pic -function-sections -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu | FileCheck %s -check-prefix=CHECK-FS
; RUN: llc < %s -relocation-model=pic -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN: -code-model=small -mcpu=pwr8 | FileCheck %s -check-prefix=SCM

%class.T = type { [2 x i8] }

define void @e_callee(ptr %this, ptr %c) { ret void }
define void @e_caller(ptr %this, ptr %c) {
  call void @e_callee(ptr %this, ptr %c)
  ret void

; CHECK-LABEL: e_caller:
; CHECK: bl e_callee
; CHECK-NEXT: nop

; CHECK-FS-LABEL: e_caller:
; CHECK-FS: bl e_callee
; CHECK-FS-NEXT: nop
}

define void @e_scallee(ptr %this, ptr %c) section "different" { ret void }
define void @e_scaller(ptr %this, ptr %c) {
  call void @e_scallee(ptr %this, ptr %c)
  ret void

; CHECK-LABEL: e_scaller:
; CHECK: bl e_scallee
; CHECK-NEXT: nop
}

define void @e_s2callee(ptr %this, ptr %c) { ret void }
define void @e_s2caller(ptr %this, ptr %c) section "different" {
  call void @e_s2callee(ptr %this, ptr %c)
  ret void

; CHECK-LABEL: e_s2caller:
; CHECK: bl e_s2callee
; CHECK-NEXT: nop
}

$cd1 = comdat any
$cd2 = comdat any

define void @e_ccallee(ptr %this, ptr %c) comdat($cd1) { ret void }
define void @e_ccaller(ptr %this, ptr %c) comdat($cd2) {
  call void @e_ccallee(ptr %this, ptr %c)
  ret void

; CHECK-LABEL: e_ccaller:
; CHECK: bl e_ccallee
; CHECK-NEXT: nop
}

$cd = comdat any

define void @e_c1callee(ptr %this, ptr %c) comdat($cd) { ret void }
define void @e_c1caller(ptr %this, ptr %c) comdat($cd) {
  call void @e_c1callee(ptr %this, ptr %c)
  ret void

; CHECK-LABEL: e_c1caller:
; CHECK: bl e_c1callee
; CHECK-NEXT: nop
}

define weak_odr hidden void @wo_hcallee(ptr %this, ptr %c) { ret void }
define void @wo_hcaller(ptr %this, ptr %c) {
  call void @wo_hcallee(ptr %this, ptr %c)
  ret void

; CHECK-LABEL: wo_hcaller:
; CHECK: bl wo_hcallee
; CHECK-NEXT: nop

; SCM-LABEL: wo_hcaller:
; SCM:       bl wo_hcallee
; SCM-NEXT:  nop
}

define weak_odr protected void @wo_pcallee(ptr %this, ptr %c) { ret void }
define void @wo_pcaller(ptr %this, ptr %c) {
  call void @wo_pcallee(ptr %this, ptr %c)
  ret void

; CHECK-LABEL: wo_pcaller:
; CHECK: bl wo_pcallee
; CHECK-NEXT: nop

; SCM-LABEL:   wo_pcaller:
; SCM:         bl wo_pcallee
; SCM-NEXT:    nop
}

define weak_odr void @wo_callee(ptr %this, ptr %c) { ret void }
define void @wo_caller(ptr %this, ptr %c) {
  call void @wo_callee(ptr %this, ptr %c)
  ret void

; CHECK-LABEL: wo_caller:
; CHECK: bl wo_callee
; CHECK-NEXT: nop
}

define weak protected void @w_pcallee(ptr %ptr) { ret void }
define void @w_pcaller(ptr %ptr) {
  call void @w_pcallee(ptr %ptr)
  ret void

; CHECK-LABEL: w_pcaller:
; CHECK: bl w_pcallee
; CHECK-NEXT: nop

; SCM-LABEL: w_pcaller:
; SCM:       bl w_pcallee
; SCM-NEXT:  nop
}

define weak hidden void @w_hcallee(ptr %ptr) { ret void }
define void @w_hcaller(ptr %ptr) {
  call void @w_hcallee(ptr %ptr)
  ret void

; CHECK-LABEL: w_hcaller:
; CHECK: bl w_hcallee
; CHECK-NEXT: nop

; SCM-LABEL: w_hcaller:
; SCM:       bl w_hcallee
; SCM-NEXT:  nop
}

define weak void @w_callee(ptr %ptr) { ret void }
define void @w_caller(ptr %ptr) {
  call void @w_callee(ptr %ptr)
  ret void

; CHECK-LABEL: w_caller:
; CHECK: bl w_callee
; CHECK-NEXT: nop
}

