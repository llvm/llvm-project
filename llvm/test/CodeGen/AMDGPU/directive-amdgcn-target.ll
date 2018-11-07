; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx600 -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX600 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=tahiti -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX600 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx601 -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX601 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hainan -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX601 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=oland -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX601 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=pitcairn -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX601 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=verde -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX601 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX700 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=kaveri -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX700 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx701 -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX701 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX701 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx702 -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX702 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx703 -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX703 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=kabini -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX703 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=mullins -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX703 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx704 -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX704 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=bonaire -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX704 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx801 -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX801 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=carrizo -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX801 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX802 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=iceland -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX802 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=tonga -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX802 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX803 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX803 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=polaris10 -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX803 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=polaris11 -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX803 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx810 -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX810 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=stoney -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX810 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX900 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx902 -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX902 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx904 -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX904 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=+code-object-v3 < %s | FileCheck --check-prefixes=GFX906 %s

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=+code-object-v3,+xnack < %s | FileCheck --check-prefixes=XNACK-GFX900 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx902 -mattr=+code-object-v3,-xnack < %s | FileCheck --check-prefixes=NO-XNACK-GFX902 %s

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx904 -mattr=+code-object-v3,+sram-ecc < %s | FileCheck --check-prefixes=SRAM-ECC-GFX904 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=+code-object-v3,-sram-ecc < %s | FileCheck --check-prefixes=NO-SRAM-ECC-GFX906 %s

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx904 -mattr=+code-object-v3,+sram-ecc,+xnack < %s | FileCheck --check-prefixes=SRAM-ECC-XNACK-GFX904 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=+code-object-v3,+xnack < %s | FileCheck --check-prefixes=XNACK-GFX906 %s

; GFX600: .amdgcn_target "amdgcn-amd-amdhsa--gfx600"
; GFX601: .amdgcn_target "amdgcn-amd-amdhsa--gfx601"
; GFX700: .amdgcn_target "amdgcn-amd-amdhsa--gfx700"
; GFX701: .amdgcn_target "amdgcn-amd-amdhsa--gfx701"
; GFX702: .amdgcn_target "amdgcn-amd-amdhsa--gfx702"
; GFX703: .amdgcn_target "amdgcn-amd-amdhsa--gfx703"
; GFX704: .amdgcn_target "amdgcn-amd-amdhsa--gfx704"
; GFX801: .amdgcn_target "amdgcn-amd-amdhsa--gfx801+xnack"
; GFX802: .amdgcn_target "amdgcn-amd-amdhsa--gfx802"
; GFX803: .amdgcn_target "amdgcn-amd-amdhsa--gfx803"
; GFX810: .amdgcn_target "amdgcn-amd-amdhsa--gfx810+xnack"
; GFX900: .amdgcn_target "amdgcn-amd-amdhsa--gfx900"
; GFX902: .amdgcn_target "amdgcn-amd-amdhsa--gfx902+xnack"
; GFX904: .amdgcn_target "amdgcn-amd-amdhsa--gfx904"
; GFX906: .amdgcn_target "amdgcn-amd-amdhsa--gfx906+sram-ecc"

; XNACK-GFX900: .amdgcn_target "amdgcn-amd-amdhsa--gfx900+xnack"
; NO-XNACK-GFX902: .amdgcn_target "amdgcn-amd-amdhsa--gfx902"

; SRAM-ECC-GFX904: .amdgcn_target "amdgcn-amd-amdhsa--gfx904+sram-ecc"
; NO-SRAM-ECC-GFX906: "amdgcn-amd-amdhsa--gfx906"

; SRAM-ECC-XNACK-GFX904: .amdgcn_target "amdgcn-amd-amdhsa--gfx904+xnack+sram-ecc"
; XNACK-GFX906: .amdgcn_target "amdgcn-amd-amdhsa--gfx906+xnack+sram-ecc"

define amdgpu_kernel void @directive_amdgcn_target() {
  ret void
}
