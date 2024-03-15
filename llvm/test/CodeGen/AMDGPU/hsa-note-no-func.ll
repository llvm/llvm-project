; RUN: llc < %s -mtriple=amdgcn-- -mcpu=gfx600 | FileCheck --check-prefixes=NONHSA-SI600 %s
; RUN: llc < %s -mtriple=amdgcn-- -mcpu=gfx601 | FileCheck --check-prefixes=NONHSA-SI601 %s
; RUN: llc < %s -mtriple=amdgcn-- -mcpu=gfx602 | FileCheck --check-prefixes=NONHSA-SI602 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx700 | FileCheck --check-prefix=HSA-CI700 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri | FileCheck --check-prefix=HSA-CI700 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx701 | FileCheck --check-prefix=HSA-CI701 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=hawaii | FileCheck --check-prefix=HSA-CI701 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx702 | FileCheck --check-prefix=HSA-CI702 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx703 | FileCheck --check-prefix=HSA-CI703 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kabini | FileCheck --check-prefix=HSA-CI703 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=mullins | FileCheck --check-prefix=HSA-CI703 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx704 | FileCheck --check-prefix=HSA-CI704 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=bonaire | FileCheck --check-prefix=HSA-CI704 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx705 | FileCheck --check-prefix=HSA-CI705 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx801 | FileCheck --check-prefix=HSA-VI801 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=carrizo -mattr=-flat-for-global | FileCheck --check-prefix=HSA-VI801 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx802 | FileCheck --check-prefix=HSA-VI802 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=iceland -mattr=-flat-for-global | FileCheck --check-prefix=HSA-VI802 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=tonga -mattr=-flat-for-global | FileCheck --check-prefix=HSA-VI802 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx803 | FileCheck --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=fiji -mattr=-flat-for-global | FileCheck --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=polaris10 | FileCheck --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=polaris11 | FileCheck --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx805 | FileCheck --check-prefix=HSA-VI805 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=tongapro | FileCheck --check-prefix=HSA-VI805 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx810 | FileCheck --check-prefix=HSA-VI810 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=stoney | FileCheck --check-prefix=HSA-VI810 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx900 -mattr=-xnack | FileCheck --check-prefix=HSA-GFX900 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx900 | FileCheck --check-prefix=HSA-GFX901 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx902 -mattr=-xnack | FileCheck --check-prefix=HSA-GFX902 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx902 | FileCheck --check-prefix=HSA-GFX903 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx904 -mattr=-xnack | FileCheck --check-prefix=HSA-GFX904 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx904 | FileCheck --check-prefix=HSA-GFX905 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx906 -mattr=-xnack | FileCheck --check-prefix=HSA-GFX906 %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=gfx906 | FileCheck --check-prefix=HSA-GFX907 %s

; NONHSA-SI600: .amd_amdgpu_isa "amdgcn-unknown-unknown--gfx600"
; NONHSA-SI601: .amd_amdgpu_isa "amdgcn-unknown-unknown--gfx601"
; NONHSA-SI602: .amd_amdgpu_isa "amdgcn-unknown-unknown--gfx602"
; HSA-CI700: .amdgcn_target "amdgcn-unknown-amdhsa--gfx700"
; HSA-CI701: .amdgcn_target "amdgcn-unknown-amdhsa--gfx701"
; HSA-CI702: .amdgcn_target "amdgcn-unknown-amdhsa--gfx702"
; HSA-CI703: .amdgcn_target "amdgcn-unknown-amdhsa--gfx703"
; HSA-CI704: .amdgcn_target "amdgcn-unknown-amdhsa--gfx704"
; HSA-CI705: .amdgcn_target "amdgcn-unknown-amdhsa--gfx705"
; HSA-VI801: .amdgcn_target "amdgcn-unknown-amdhsa--gfx801"
; HSA-VI802: .amdgcn_target "amdgcn-unknown-amdhsa--gfx802"
; HSA-VI803: .amdgcn_target "amdgcn-unknown-amdhsa--gfx803"
; HSA-VI805: .amdgcn_target "amdgcn-unknown-amdhsa--gfx805"
; HSA-VI810: .amdgcn_target "amdgcn-unknown-amdhsa--gfx810"
; HSA-GFX900: .amdgcn_target "amdgcn-unknown-amdhsa--gfx900:xnack-"
; HSA-GFX901: .amdgcn_target "amdgcn-unknown-amdhsa--gfx900"
; HSA-GFX902: .amdgcn_target "amdgcn-unknown-amdhsa--gfx902:xnack-"
; HSA-GFX903: .amdgcn_target "amdgcn-unknown-amdhsa--gfx902"
; HSA-GFX904: .amdgcn_target "amdgcn-unknown-amdhsa--gfx904:xnack-"
; HSA-GFX905: .amdgcn_target "amdgcn-unknown-amdhsa--gfx904"
; HSA-GFX906: .amdgcn_target "amdgcn-unknown-amdhsa--gfx906:xnack-"
; HSA-GFX907: .amdgcn_target "amdgcn-unknown-amdhsa--gfx906"

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
