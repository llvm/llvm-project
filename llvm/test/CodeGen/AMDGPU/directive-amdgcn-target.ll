; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx600 < %s | FileCheck --check-prefixes=GFX600 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=tahiti < %s | FileCheck --check-prefixes=GFX600 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx601 < %s | FileCheck --check-prefixes=GFX601 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=pitcairn < %s | FileCheck --check-prefixes=GFX601 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=verde < %s | FileCheck --check-prefixes=GFX601 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx602 < %s | FileCheck --check-prefixes=GFX602 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hainan < %s | FileCheck --check-prefixes=GFX602 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=oland < %s | FileCheck --check-prefixes=GFX602 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 < %s | FileCheck --check-prefixes=GFX700 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=kaveri < %s | FileCheck --check-prefixes=GFX700 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx701 < %s | FileCheck --check-prefixes=GFX701 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii < %s | FileCheck --check-prefixes=GFX701 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx702 < %s | FileCheck --check-prefixes=GFX702 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx703 < %s | FileCheck --check-prefixes=GFX703 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=kabini < %s | FileCheck --check-prefixes=GFX703 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=mullins < %s | FileCheck --check-prefixes=GFX703 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx704 < %s | FileCheck --check-prefixes=GFX704 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=bonaire < %s | FileCheck --check-prefixes=GFX704 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx705 < %s | FileCheck --check-prefixes=GFX705 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa --amdhsa-code-object-version=3 -mcpu=gfx801 < %s | FileCheck --check-prefixes=GFX801 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa --amdhsa-code-object-version=3 -mcpu=carrizo < %s | FileCheck --check-prefixes=GFX801 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 < %s | FileCheck --check-prefixes=GFX802 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=iceland < %s | FileCheck --check-prefixes=GFX802 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=tonga < %s | FileCheck --check-prefixes=GFX802 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 < %s | FileCheck --check-prefixes=GFX803 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji < %s | FileCheck --check-prefixes=GFX803 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=polaris10 < %s | FileCheck --check-prefixes=GFX803 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=polaris11 < %s | FileCheck --check-prefixes=GFX803 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx805 < %s | FileCheck --check-prefixes=GFX805 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=tongapro < %s | FileCheck --check-prefixes=GFX805 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx810 < %s | FileCheck --check-prefixes=GFX810 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=stoney < %s | FileCheck --check-prefixes=GFX810 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck --check-prefixes=GFX900 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx902 < %s | FileCheck --check-prefixes=GFX902 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx904 < %s | FileCheck --check-prefixes=GFX904 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 < %s | FileCheck --check-prefixes=GFX906 %s

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=+xnack < %s | FileCheck --check-prefixes=XNACK-GFX900 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx902 -mattr=-xnack < %s | FileCheck --check-prefixes=NO-XNACK-GFX902 %s

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx904 -mattr=+sramecc < %s | FileCheck --check-prefixes=SRAM-ECC-GFX904 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=+sramecc < %s | FileCheck --check-prefixes=SRAM-ECC-GFX906 %s

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx904 -mattr=+sramecc,+xnack < %s | FileCheck --check-prefixes=SRAM-ECC-XNACK-GFX904 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=+sramecc,+xnack < %s | FileCheck --check-prefixes=SRAM-ECC-XNACK-GFX906 %s

; FIXME: With the default attributes these directives are not accurate for
; xnack and sramecc. Subsequent Target-ID patches will address this.

; GFX600: .amdgcn_target "amdgcn-amd-amdhsa--gfx600"
; GFX601: .amdgcn_target "amdgcn-amd-amdhsa--gfx601"
; GFX602: .amdgcn_target "amdgcn-amd-amdhsa--gfx602"
; GFX700: .amdgcn_target "amdgcn-amd-amdhsa--gfx700"
; GFX701: .amdgcn_target "amdgcn-amd-amdhsa--gfx701"
; GFX702: .amdgcn_target "amdgcn-amd-amdhsa--gfx702"
; GFX703: .amdgcn_target "amdgcn-amd-amdhsa--gfx703"
; GFX704: .amdgcn_target "amdgcn-amd-amdhsa--gfx704"
; GFX705: .amdgcn_target "amdgcn-amd-amdhsa--gfx705"
; GFX801: .amdgcn_target "amdgcn-amd-amdhsa--gfx801"
; GFX802: .amdgcn_target "amdgcn-amd-amdhsa--gfx802"
; GFX803: .amdgcn_target "amdgcn-amd-amdhsa--gfx803"
; GFX805: .amdgcn_target "amdgcn-amd-amdhsa--gfx805"
; GFX810: .amdgcn_target "amdgcn-amd-amdhsa--gfx810"
; GFX900: .amdgcn_target "amdgcn-amd-amdhsa--gfx900"
; GFX902: .amdgcn_target "amdgcn-amd-amdhsa--gfx902"
; GFX904: .amdgcn_target "amdgcn-amd-amdhsa--gfx904"
; GFX906: .amdgcn_target "amdgcn-amd-amdhsa--gfx906"

; XNACK-GFX900: .amdgcn_target "amdgcn-amd-amdhsa--gfx900+xnack"
; NO-XNACK-GFX902: .amdgcn_target "amdgcn-amd-amdhsa--gfx902"

; SRAM-ECC-GFX904: .amdgcn_target "amdgcn-amd-amdhsa--gfx904+sramecc"
; SRAM-ECC-GFX906: "amdgcn-amd-amdhsa--gfx906+sramecc"

; SRAM-ECC-XNACK-GFX904: .amdgcn_target "amdgcn-amd-amdhsa--gfx904+xnack+sramecc"
; SRAM-ECC-XNACK-GFX906: .amdgcn_target "amdgcn-amd-amdhsa--gfx906+xnack+sramecc"

define amdgpu_kernel void @directive_amdgcn_target() {
  ret void
}
