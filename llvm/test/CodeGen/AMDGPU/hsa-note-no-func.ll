; RUN: llc < %s -mtriple=amdgpu6.00-- | FileCheck --check-prefixes=NONHSA-SI600 %s
; RUN: llc < %s -mtriple=amdgpu6.01-- | FileCheck --check-prefixes=NONHSA-SI601 %s
; RUN: llc < %s -mtriple=amdgpu6.02-- | FileCheck --check-prefixes=NONHSA-SI602 %s
; RUN: llc < %s -mtriple=amdgpu7.00--amdhsa | FileCheck --check-prefix=HSA-CI700 %s
; RUN: llc < %s -mtriple=amdgpu7.00--amdhsa | FileCheck --check-prefix=HSA-CI700 %s
; RUN: llc < %s -mtriple=amdgpu7.01--amdhsa | FileCheck --check-prefix=HSA-CI701 %s
; RUN: llc < %s -mtriple=amdgpu7.01--amdhsa | FileCheck --check-prefix=HSA-CI701 %s
; RUN: llc < %s -mtriple=amdgpu7.02--amdhsa | FileCheck --check-prefix=HSA-CI702 %s
; RUN: llc < %s -mtriple=amdgpu7.03--amdhsa | FileCheck --check-prefix=HSA-CI703 %s
; RUN: llc < %s -mtriple=amdgpu7.03--amdhsa | FileCheck --check-prefix=HSA-CI703 %s
; RUN: llc < %s -mtriple=amdgpu7.03--amdhsa | FileCheck --check-prefix=HSA-CI703 %s
; RUN: llc < %s -mtriple=amdgpu7.04--amdhsa | FileCheck --check-prefix=HSA-CI704 %s
; RUN: llc < %s -mtriple=amdgpu7.04--amdhsa | FileCheck --check-prefix=HSA-CI704 %s
; RUN: llc < %s -mtriple=amdgpu7.05--amdhsa | FileCheck --check-prefix=HSA-CI705 %s
; RUN: llc < %s -mtriple=amdgpu8.01--amdhsa | FileCheck --check-prefix=HSA-VI801 %s
; RUN: llc < %s -mtriple=amdgpu8.01--amdhsa -mattr=-flat-for-global | FileCheck --check-prefix=HSA-VI801 %s
; RUN: llc < %s -mtriple=amdgpu8.02--amdhsa | FileCheck --check-prefix=HSA-VI802 %s
; RUN: llc < %s -mtriple=amdgpu8.02--amdhsa -mattr=-flat-for-global | FileCheck --check-prefix=HSA-VI802 %s
; RUN: llc < %s -mtriple=amdgpu8.02--amdhsa -mattr=-flat-for-global | FileCheck --check-prefix=HSA-VI802 %s
; RUN: llc < %s -mtriple=amdgpu8.03--amdhsa | FileCheck --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgpu8.03--amdhsa -mattr=-flat-for-global | FileCheck --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgpu8.03--amdhsa | FileCheck --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgpu8.03--amdhsa | FileCheck --check-prefix=HSA-VI803 %s
; RUN: llc < %s -mtriple=amdgpu8.05--amdhsa | FileCheck --check-prefix=HSA-VI805 %s
; RUN: llc < %s -mtriple=amdgpu8.05--amdhsa | FileCheck --check-prefix=HSA-VI805 %s
; RUN: llc < %s -mtriple=amdgpu8.10--amdhsa | FileCheck --check-prefix=HSA-VI810 %s
; RUN: llc < %s -mtriple=amdgpu8.10--amdhsa | FileCheck --check-prefix=HSA-VI810 %s
; RUN: llc < %s -mtriple=amdgpu9.00--amdhsa --amdgpu-xnack=false | FileCheck --check-prefix=HSA-GFX900 %s
; RUN: llc < %s -mtriple=amdgpu9.00--amdhsa | FileCheck --check-prefix=HSA-GFX901 %s
; RUN: llc < %s -mtriple=amdgpu9.02--amdhsa --amdgpu-xnack=false | FileCheck --check-prefix=HSA-GFX902 %s
; RUN: llc < %s -mtriple=amdgpu9.02--amdhsa | FileCheck --check-prefix=HSA-GFX903 %s
; RUN: llc < %s -mtriple=amdgpu9.04--amdhsa --amdgpu-xnack=false | FileCheck --check-prefix=HSA-GFX904 %s
; RUN: llc < %s -mtriple=amdgpu9.04--amdhsa | FileCheck --check-prefix=HSA-GFX905 %s
; RUN: llc < %s -mtriple=amdgpu9.06--amdhsa --amdgpu-xnack=false | FileCheck --check-prefix=HSA-GFX906 %s
; RUN: llc < %s -mtriple=amdgpu9.06--amdhsa | FileCheck --check-prefix=HSA-GFX907 %s

; NONHSA-SI600: .amd_amdgpu_isa "amdgpu6.00-unknown-unknown-unknown-gfx600"
; NONHSA-SI601: .amd_amdgpu_isa "amdgpu6.01-unknown-unknown-unknown-gfx601"
; NONHSA-SI602: .amd_amdgpu_isa "amdgpu6.02-unknown-unknown-unknown-gfx602"
; HSA-CI700: .amdgcn_target "amdgpu7.00-unknown-amdhsa-unknown-gfx700"
; HSA-CI701: .amdgcn_target "amdgpu7.01-unknown-amdhsa-unknown-gfx701"
; HSA-CI702: .amdgcn_target "amdgpu7.02-unknown-amdhsa-unknown-gfx702"
; HSA-CI703: .amdgcn_target "amdgpu7.03-unknown-amdhsa-unknown-gfx703"
; HSA-CI704: .amdgcn_target "amdgpu7.04-unknown-amdhsa-unknown-gfx704"
; HSA-CI705: .amdgcn_target "amdgpu7.05-unknown-amdhsa-unknown-gfx705"
; HSA-VI801: .amdgcn_target "amdgpu8.01-unknown-amdhsa-unknown-gfx801"
; HSA-VI802: .amdgcn_target "amdgpu8.02-unknown-amdhsa-unknown-gfx802"
; HSA-VI803: .amdgcn_target "amdgpu8.03-unknown-amdhsa-unknown-gfx803"
; HSA-VI805: .amdgcn_target "amdgpu8.05-unknown-amdhsa-unknown-gfx805"
; HSA-VI810: .amdgcn_target "amdgpu8.10-unknown-amdhsa-unknown-gfx810"
; HSA-GFX900: .amdgcn_target "amdgpu9.00-unknown-amdhsa-unknown-gfx900:xnack-"
; HSA-GFX901: .amdgcn_target "amdgpu9.00-unknown-amdhsa-unknown-gfx900"
; HSA-GFX902: .amdgcn_target "amdgpu9.02-unknown-amdhsa-unknown-gfx902:xnack-"
; HSA-GFX903: .amdgcn_target "amdgpu9.02-unknown-amdhsa-unknown-gfx902"
; HSA-GFX904: .amdgcn_target "amdgpu9.04-unknown-amdhsa-unknown-gfx904:xnack-"
; HSA-GFX905: .amdgcn_target "amdgpu9.04-unknown-amdhsa-unknown-gfx904"
; HSA-GFX906: .amdgcn_target "amdgpu9.06-unknown-amdhsa-unknown-gfx906:xnack-"
; HSA-GFX907: .amdgcn_target "amdgpu9.06-unknown-amdhsa-unknown-gfx906"

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
