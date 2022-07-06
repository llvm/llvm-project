; RUN: llc -filetype=obj -march=r600 -mcpu=r600 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,R600 %s
; RUN: llc -filetype=obj -march=r600 -mcpu=r630 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,R630 %s
; RUN: llc -filetype=obj -march=r600 -mcpu=rs880 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,RS880 %s
; RUN: llc -filetype=obj -march=r600 -mcpu=rv670 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,RV670 %s
; RUN: llc -filetype=obj -march=r600 -mcpu=rv710 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,RV710 %s
; RUN: llc -filetype=obj -march=r600 -mcpu=rv730 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,RV730 %s
; RUN: llc -filetype=obj -march=r600 -mcpu=rv770 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,RV770 %s
; RUN: llc -filetype=obj -march=r600 -mcpu=cedar < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,CEDAR %s
; RUN: llc -filetype=obj -march=r600 -mcpu=cypress < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,CYPRESS %s
; RUN: llc -filetype=obj -march=r600 -mcpu=juniper < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,JUNIPER %s
; RUN: llc -filetype=obj -march=r600 -mcpu=redwood < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,REDWOOD %s
; RUN: llc -filetype=obj -march=r600 -mcpu=sumo < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,SUMO %s
; RUN: llc -filetype=obj -march=r600 -mcpu=barts < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,BARTS %s
; RUN: llc -filetype=obj -march=r600 -mcpu=caicos < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,CAICOS %s
; RUN: llc -filetype=obj -march=r600 -mcpu=cayman < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,CAYMAN %s
; RUN: llc -filetype=obj -march=r600 -mcpu=turks < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,TURKS %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx600 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX600 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=tahiti < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX600 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx601 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX601 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=pitcairn < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX601 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=verde < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX601 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx602 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX602 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=hainan < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX602 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=oland < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX602 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx700 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX700 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=kaveri < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX700 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx701 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX701 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=hawaii < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX701 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx702 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX702 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx703 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX703 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=kabini < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX703 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=mullins < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX703 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx704 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX704 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=bonaire < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX704 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx705 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX705 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx801 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX801 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=carrizo < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX801 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx802 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX802 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=iceland < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX802 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=tonga < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX802 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx803 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX803 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=fiji < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX803 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=polaris10 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX803 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=polaris11 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX803 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx805 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX805 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=tongapro < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX805 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx810 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX810 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=stoney < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX810 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx900 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX900 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx902 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX902 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx904 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX904 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx906 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX906 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx908 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX908 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx909 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX909 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx90a < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX90A %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx90c < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX90C %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx940 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX940 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx1010 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1010 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx1011 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1011 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx1012 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1012 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx1013 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1013 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx1030 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1030 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx1031 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1031 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx1032 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1032 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx1033 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1033 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx1034 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1034 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx1035 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1035 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx1036 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1036 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx1100 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1100 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx1101 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1101 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx1102 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1102 %s
; RUN: llc -filetype=obj -march=amdgcn -mcpu=gfx1103 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1103 %s

; FIXME: With the default attributes the eflags are not accurate for
; xnack and sramecc. Subsequent Target-ID patches will address this.

; ARCH-R600: Format: elf32-amdgpu
; ARCH-R600: Arch: r600
; ARCH-R600: AddressSize: 32bit

; ARCH-GCN: Format: elf64-amdgpu
; ARCH-GCN: Arch: amdgcn
; ARCH-GCN: AddressSize: 64bit

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
; GFX602:        EF_AMDGPU_MACH_AMDGCN_GFX602 (0x3A)
; GFX700:        EF_AMDGPU_MACH_AMDGCN_GFX700 (0x22)
; GFX701:        EF_AMDGPU_MACH_AMDGCN_GFX701 (0x23)
; GFX702:        EF_AMDGPU_MACH_AMDGCN_GFX702 (0x24)
; GFX703:        EF_AMDGPU_MACH_AMDGCN_GFX703 (0x25)
; GFX704:        EF_AMDGPU_MACH_AMDGCN_GFX704 (0x26)
; GFX705:        EF_AMDGPU_MACH_AMDGCN_GFX705 (0x3B)
; GFX801:        EF_AMDGPU_MACH_AMDGCN_GFX801 (0x28)
; GFX802:        EF_AMDGPU_MACH_AMDGCN_GFX802 (0x29)
; GFX803:        EF_AMDGPU_MACH_AMDGCN_GFX803 (0x2A)
; GFX805:        EF_AMDGPU_MACH_AMDGCN_GFX805 (0x3C)
; GFX810:        EF_AMDGPU_MACH_AMDGCN_GFX810 (0x2B)
; GFX900:        EF_AMDGPU_MACH_AMDGCN_GFX900 (0x2C)
; GFX902:        EF_AMDGPU_MACH_AMDGCN_GFX902 (0x2D)
; GFX904:        EF_AMDGPU_MACH_AMDGCN_GFX904 (0x2E)
; GFX906:        EF_AMDGPU_MACH_AMDGCN_GFX906 (0x2F)
; GFX908:        EF_AMDGPU_MACH_AMDGCN_GFX908 (0x30)
; GFX909:        EF_AMDGPU_MACH_AMDGCN_GFX909 (0x31)
; GFX90A:        EF_AMDGPU_MACH_AMDGCN_GFX90A (0x3F)
; GFX90C:        EF_AMDGPU_MACH_AMDGCN_GFX90C (0x32)
; GFX940:        EF_AMDGPU_MACH_AMDGCN_GFX940 (0x40)
; GFX1010:       EF_AMDGPU_MACH_AMDGCN_GFX1010 (0x33)
; GFX1011:       EF_AMDGPU_MACH_AMDGCN_GFX1011 (0x34)
; GFX1012:       EF_AMDGPU_MACH_AMDGCN_GFX1012 (0x35)
; GFX1013:       EF_AMDGPU_MACH_AMDGCN_GFX1013 (0x42)
; GFX1030:       EF_AMDGPU_MACH_AMDGCN_GFX1030 (0x36)
; GFX1031:       EF_AMDGPU_MACH_AMDGCN_GFX1031 (0x37)
; GFX1032:       EF_AMDGPU_MACH_AMDGCN_GFX1032 (0x38)
; GFX1033:       EF_AMDGPU_MACH_AMDGCN_GFX1033 (0x39)
; GFX1034:       EF_AMDGPU_MACH_AMDGCN_GFX1034 (0x3E)
; GFX1035:       EF_AMDGPU_MACH_AMDGCN_GFX1035 (0x3D)
; GFX1036:       EF_AMDGPU_MACH_AMDGCN_GFX1036 (0x45)
; GFX1100:       EF_AMDGPU_MACH_AMDGCN_GFX1100 (0x41)
; GFX1101:       EF_AMDGPU_MACH_AMDGCN_GFX1101 (0x46)
; GFX1102:       EF_AMDGPU_MACH_AMDGCN_GFX1102 (0x47)
; GFX1103:       EF_AMDGPU_MACH_AMDGCN_GFX1103 (0x44)
; ALL:         ]

define amdgpu_kernel void @elf_header() {
  ret void
}
