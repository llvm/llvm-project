; RUN: llc -filetype=obj -march=r600 -mcpu=r600 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-R600,R600 %s
; RUN: llc -filetype=obj -march=r600 -mcpu=r630 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-R600,R630 %s
; RUN: llc -filetype=obj -march=r600 -mcpu=rs880 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-R600,RS880 %s
; RUN: llc -filetype=obj -march=r600 -mcpu=rv670 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-R600,RV670 %s
; RUN: llc -filetype=obj -march=r600 -mcpu=rv710 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-R600,RV710 %s
; RUN: llc -filetype=obj -march=r600 -mcpu=rv730 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-R600,RV730 %s
; RUN: llc -filetype=obj -march=r600 -mcpu=rv770 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-R600,RV770 %s
; RUN: llc -filetype=obj -march=r600 -mcpu=cedar < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-R600,CEDAR %s
; RUN: llc -filetype=obj -march=r600 -mcpu=cypress < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-R600,CYPRESS %s
; RUN: llc -filetype=obj -march=r600 -mcpu=juniper < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-R600,JUNIPER %s
; RUN: llc -filetype=obj -march=r600 -mcpu=redwood < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-R600,REDWOOD %s
; RUN: llc -filetype=obj -march=r600 -mcpu=sumo < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-R600,SUMO %s
; RUN: llc -filetype=obj -march=r600 -mcpu=barts < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-R600,BARTS %s
; RUN: llc -filetype=obj -march=r600 -mcpu=caicos < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-R600,CAICOS %s
; RUN: llc -filetype=obj -march=r600 -mcpu=cayman < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-R600,CAYMAN %s
; RUN: llc -filetype=obj -march=r600 -mcpu=turks < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-R600,TURKS %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx600 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX600 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=tahiti < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX600 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx601 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX601 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=hainan < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX601 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=oland < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX601 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=pitcairn < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX601 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=verde < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX601 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx700 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX700 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=kaveri < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX700 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx701 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX701 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=hawaii < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX701 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx702 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX702 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx703 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX703 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=kabini < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX703 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=mullins < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX703 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx704 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX704 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=bonaire < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX704 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx801 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX801 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=carrizo < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX801 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx802 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX802 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=iceland < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX802 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=tonga < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX802 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx803 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX803 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=fiji < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX803 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=polaris10 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX803 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=polaris11 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX803 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx810 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX810 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=stoney < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX810 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx900 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX900 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx902 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX902 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx904 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX904 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx906 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX906 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx909 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX909 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx1010 < %s | llvm-readobj --file-headers - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1010 %s

; ARCH-R600: Arch: r600
; ARCH-GCN:  Arch: amdgcn

; ALL:         Flags [
; R600:          EF_AMDGPU_MACH_R600_R600     (0x1)
; R630:          EF_AMDGPU_MACH_R600_R630     (0x2)
; RS880:         EF_AMDGPU_MACH_R600_RS880    (0x3)
; RV670:         EF_AMDGPU_MACH_R600_RV670    (0x4)
; RV710:         EF_AMDGPU_MACH_R600_RV710    (0x5)
; RV730:         EF_AMDGPU_MACH_R600_RV730    (0x6)
; RV770:         EF_AMDGPU_MACH_R600_RV770    (0x7)
; CEDAR:         EF_AMDGPU_MACH_R600_CEDAR    (0x8)
; CYPRESS:       EF_AMDGPU_MACH_R600_CYPRESS  (0x9)
; JUNIPER:       EF_AMDGPU_MACH_R600_JUNIPER  (0xA)
; REDWOOD:       EF_AMDGPU_MACH_R600_REDWOOD  (0xB)
; SUMO:          EF_AMDGPU_MACH_R600_SUMO     (0xC)
; BARTS:         EF_AMDGPU_MACH_R600_BARTS    (0xD)
; CAICOS:        EF_AMDGPU_MACH_R600_CAICOS   (0xE)
; CAYMAN:        EF_AMDGPU_MACH_R600_CAYMAN   (0xF)
; TURKS:         EF_AMDGPU_MACH_R600_TURKS    (0x10)
; GFX600:        EF_AMDGPU_MACH_AMDGCN_GFX600 (0x20)
; GFX601:        EF_AMDGPU_MACH_AMDGCN_GFX601 (0x21)
; GFX700:        EF_AMDGPU_MACH_AMDGCN_GFX700 (0x22)
; GFX701:        EF_AMDGPU_MACH_AMDGCN_GFX701 (0x23)
; GFX702:        EF_AMDGPU_MACH_AMDGCN_GFX702 (0x24)
; GFX703:        EF_AMDGPU_MACH_AMDGCN_GFX703 (0x25)
; GFX704:        EF_AMDGPU_MACH_AMDGCN_GFX704 (0x26)
; GFX801:        EF_AMDGPU_MACH_AMDGCN_GFX801 (0x28)
; GFX801-NEXT:   EF_AMDGPU_XNACK              (0x100)
; GFX802:        EF_AMDGPU_MACH_AMDGCN_GFX802 (0x29)
; GFX803:        EF_AMDGPU_MACH_AMDGCN_GFX803 (0x2A)
; GFX810:        EF_AMDGPU_MACH_AMDGCN_GFX810 (0x2B)
; GFX810-NEXT:   EF_AMDGPU_XNACK              (0x100)
; GFX900:        EF_AMDGPU_MACH_AMDGCN_GFX900 (0x2C)
; GFX902:        EF_AMDGPU_MACH_AMDGCN_GFX902 (0x2D)
; GFX902-NEXT:   EF_AMDGPU_XNACK              (0x100)
; GFX904:        EF_AMDGPU_MACH_AMDGCN_GFX904 (0x2E)
; GFX906:        EF_AMDGPU_MACH_AMDGCN_GFX906 (0x2F)
; GFX909:        EF_AMDGPU_MACH_AMDGCN_GFX909 (0x31)
; GFX1010:       EF_AMDGPU_MACH_AMDGCN_GFX1010 (0x33)
; ALL:         ]

define amdgpu_kernel void @elf_header() {
  ret void
}
