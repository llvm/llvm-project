; RUN: llc -filetype=obj -mtriple=r600 -mcpu=r600 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,R600 %s
; RUN: llc -filetype=obj -mtriple=r600 -mcpu=r630 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,R630 %s
; RUN: llc -filetype=obj -mtriple=r600 -mcpu=rs880 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,RS880 %s
; RUN: llc -filetype=obj -mtriple=r600 -mcpu=rv670 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,RV670 %s
; RUN: llc -filetype=obj -mtriple=r600 -mcpu=rv710 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,RV710 %s
; RUN: llc -filetype=obj -mtriple=r600 -mcpu=rv730 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,RV730 %s
; RUN: llc -filetype=obj -mtriple=r600 -mcpu=rv770 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,RV770 %s
; RUN: llc -filetype=obj -mtriple=r600 -mcpu=cedar < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,CEDAR %s
; RUN: llc -filetype=obj -mtriple=r600 -mcpu=cypress < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,CYPRESS %s
; RUN: llc -filetype=obj -mtriple=r600 -mcpu=juniper < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,JUNIPER %s
; RUN: llc -filetype=obj -mtriple=r600 -mcpu=redwood < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,REDWOOD %s
; RUN: llc -filetype=obj -mtriple=r600 -mcpu=sumo < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,SUMO %s
; RUN: llc -filetype=obj -mtriple=r600 -mcpu=barts < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,BARTS %s
; RUN: llc -filetype=obj -mtriple=r600 -mcpu=caicos < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,CAICOS %s
; RUN: llc -filetype=obj -mtriple=r600 -mcpu=cayman < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,CAYMAN %s
; RUN: llc -filetype=obj -mtriple=r600 -mcpu=turks < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-R600,TURKS %s
; RUN: llc -filetype=obj -mtriple=amdgpu6.00 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX600 %s
; RUN: llc -filetype=obj -mtriple=amdgpu6.00 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX600 %s
; RUN: llc -filetype=obj -mtriple=amdgpu6.01 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX601 %s
; RUN: llc -filetype=obj -mtriple=amdgpu6.01 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX601 %s
; RUN: llc -filetype=obj -mtriple=amdgpu6.01 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX601 %s
; RUN: llc -filetype=obj -mtriple=amdgpu6.02 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX602 %s
; RUN: llc -filetype=obj -mtriple=amdgpu6.02 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX602 %s
; RUN: llc -filetype=obj -mtriple=amdgpu6.02 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX602 %s
; RUN: llc -filetype=obj -mtriple=amdgpu7.00 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX700 %s
; RUN: llc -filetype=obj -mtriple=amdgpu7.00 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX700 %s
; RUN: llc -filetype=obj -mtriple=amdgpu7.01 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX701 %s
; RUN: llc -filetype=obj -mtriple=amdgpu7.01 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX701 %s
; RUN: llc -filetype=obj -mtriple=amdgpu7.02 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX702 %s
; RUN: llc -filetype=obj -mtriple=amdgpu7.03 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX703 %s
; RUN: llc -filetype=obj -mtriple=amdgpu7.03 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX703 %s
; RUN: llc -filetype=obj -mtriple=amdgpu7.03 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX703 %s
; RUN: llc -filetype=obj -mtriple=amdgpu7.04 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX704 %s
; RUN: llc -filetype=obj -mtriple=amdgpu7.04 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX704 %s
; RUN: llc -filetype=obj -mtriple=amdgpu7.05 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX705 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.01 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX801 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.01 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX801 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.02 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX802 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.02 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX802 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.02 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX802 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.03 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX803 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.03 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX803 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.03 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX803 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.03 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX803 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.05 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX805 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.05 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX805 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.10 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX810 %s
; RUN: llc -filetype=obj -mtriple=amdgpu8.10 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX810 %s
; RUN: llc -filetype=obj -mtriple=amdgpu9.00 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX900 %s
; RUN: llc -filetype=obj -mtriple=amdgpu9.02 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX902 %s
; RUN: llc -filetype=obj -mtriple=amdgpu9.04 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX904 %s
; RUN: llc -filetype=obj -mtriple=amdgpu9.06 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX906 %s
; RUN: llc -filetype=obj -mtriple=amdgpu9.08 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX908 %s
; RUN: llc -filetype=obj -mtriple=amdgpu9.09 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX909 %s
; RUN: llc -filetype=obj -mtriple=amdgpu9.0a < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX90A %s
; RUN: llc -filetype=obj -mtriple=amdgpu9.0c < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX90C %s
; RUN: llc -filetype=obj -mtriple=amdgpu9.42 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX942 %s
; RUN: llc -filetype=obj -mtriple=amdgpu9.50 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX950 %s
; RUN: llc -filetype=obj -mtriple=amdgpu10.10 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1010 %s
; RUN: llc -filetype=obj -mtriple=amdgpu10.11 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1011 %s
; RUN: llc -filetype=obj -mtriple=amdgpu10.12 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1012 %s
; RUN: llc -filetype=obj -mtriple=amdgpu10.13 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1013 %s
; RUN: llc -filetype=obj -mtriple=amdgpu10.30 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1030 %s
; RUN: llc -filetype=obj -mtriple=amdgpu10.31 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1031 %s
; RUN: llc -filetype=obj -mtriple=amdgpu10.32 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1032 %s
; RUN: llc -filetype=obj -mtriple=amdgpu10.33 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1033 %s
; RUN: llc -filetype=obj -mtriple=amdgpu10.34 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1034 %s
; RUN: llc -filetype=obj -mtriple=amdgpu10.35 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1035 %s
; RUN: llc -filetype=obj -mtriple=amdgpu10.36 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1036 %s
; RUN: llc -filetype=obj -mtriple=amdgpu11.00 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1100 %s
; RUN: llc -filetype=obj -mtriple=amdgpu11.01 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1101 %s
; RUN: llc -filetype=obj -mtriple=amdgpu11.02 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1102 %s
; RUN: llc -filetype=obj -mtriple=amdgpu11.03 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1103 %s
; RUN: llc -filetype=obj -mtriple=amdgpu11.50 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1150 %s
; RUN: llc -filetype=obj -mtriple=amdgpu11.51 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1151 %s
; RUN: llc -filetype=obj -mtriple=amdgpu11.52 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1152 %s
; RUN: llc -filetype=obj -mtriple=amdgpu11.53 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1153 %s
; RUN: llc -filetype=obj -mtriple=amdgpu11.54 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1154 %s
; RUN: llc -filetype=obj -mtriple=amdgpu11.70 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1170 %s
; RUN: llc -filetype=obj -mtriple=amdgpu11.71 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1171 %s
; RUN: llc -filetype=obj -mtriple=amdgpu11.72 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1172 %s
; RUN: llc -filetype=obj -mtriple=amdgpu12.00 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1200 %s
; RUN: llc -filetype=obj -mtriple=amdgpu12.01 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1201 %s
; RUN: llc -filetype=obj -mtriple=amdgpu12.50 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1250 %s
; RUN: llc -filetype=obj -mtriple=amdgpu12.51 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1251 %s
; RUN: llc -filetype=obj -mtriple=amdgpu13.10 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX1310 %s

; RUN: llc -filetype=obj --amdhsa-code-object-version=6 -mtriple=amdgpu9 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX9_GENERIC %s
; RUN: llc -filetype=obj --amdhsa-code-object-version=6 -mtriple=amdgpu9.4 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX9_4_GENERIC %s
; RUN: llc -filetype=obj --amdhsa-code-object-version=6 -mtriple=amdgpu10.1 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX10_1_GENERIC %s
; RUN: llc -filetype=obj --amdhsa-code-object-version=6 -mtriple=amdgpu10.3 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX10_3_GENERIC %s
; RUN: llc -filetype=obj --amdhsa-code-object-version=6 -mtriple=amdgpu11 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX11_GENERIC %s
; RUN: llc -filetype=obj --amdhsa-code-object-version=6 -mtriple=amdgpu11.7 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX11_7_GENERIC %s
; RUN: llc -filetype=obj --amdhsa-code-object-version=6 -mtriple=amdgpu12 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX12_GENERIC %s
; RUN: llc -filetype=obj --amdhsa-code-object-version=6 -mtriple=amdgpu12.5 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX12_5_GENERIC %s
; RUN: llc -filetype=obj --amdhsa-code-object-version=6 -mtriple=amdgpu13 < %s | llvm-readobj --file-header - | FileCheck --check-prefixes=ALL,ARCH-GCN,GFX13_GENERIC %s

; FIXME: With the default attributes the eflags are not accurate for
; xnack and sramecc. Subsequent Target-ID patches will address this.

; ARCH-R600: Format: elf32-amdgpu
; ARCH-R600: Arch: r600
; ARCH-R600: AddressSize: 32bit

; ARCH-GCN: Format: elf64-amdgpu
; ARCH-GCN: Arch: amdgpu
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
; GFX942:        EF_AMDGPU_MACH_AMDGCN_GFX942 (0x4C)
; GFX950:        EF_AMDGPU_MACH_AMDGCN_GFX950 (0x4F)
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
; GFX1150:       EF_AMDGPU_MACH_AMDGCN_GFX1150 (0x43)
; GFX1151:       EF_AMDGPU_MACH_AMDGCN_GFX1151 (0x4A)
; GFX1152:       EF_AMDGPU_MACH_AMDGCN_GFX1152 (0x55)
; GFX1153:       EF_AMDGPU_MACH_AMDGCN_GFX1153 (0x58)
; GFX1154:       EF_AMDGPU_MACH_AMDGCN_GFX1154 (0x57)
; GFX1170:       EF_AMDGPU_MACH_AMDGCN_GFX1170 (0x5D)
; GFX1171:       EF_AMDGPU_MACH_AMDGCN_GFX1171 (0x5E)
; GFX1172:       EF_AMDGPU_MACH_AMDGCN_GFX1172 (0x5C)
; GFX1200:       EF_AMDGPU_MACH_AMDGCN_GFX1200 (0x48)
; GFX1201:       EF_AMDGPU_MACH_AMDGCN_GFX1201 (0x4E)
; GFX1250:       EF_AMDGPU_MACH_AMDGCN_GFX1250 (0x49)
; GFX1251:       EF_AMDGPU_MACH_AMDGCN_GFX1251 (0x5A)
; GFX1310:       EF_AMDGPU_MACH_AMDGCN_GFX1310 (0x50)

; GFX9_GENERIC:       EF_AMDGPU_MACH_AMDGCN_GFX9_GENERIC (0x51)
; GFX9_4_GENERIC:     EF_AMDGPU_MACH_AMDGCN_GFX9_4_GENERIC (0x5F)
; GFX10_1_GENERIC:    EF_AMDGPU_MACH_AMDGCN_GFX10_1_GENERIC (0x52)
; GFX10_3_GENERIC:    EF_AMDGPU_MACH_AMDGCN_GFX10_3_GENERIC (0x53)
; GFX11_GENERIC:      EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC (0x54)
; GFX11_7_GENERIC:    EF_AMDGPU_MACH_AMDGCN_GFX11_7_GENERIC (0x62)
; GFX12_GENERIC:      EF_AMDGPU_MACH_AMDGCN_GFX12_GENERIC (0x59)
; GFX12_5_GENERIC:    EF_AMDGPU_MACH_AMDGCN_GFX12_5_GENERIC (0x5B)
; GFX13_GENERIC:      EF_AMDGPU_MACH_AMDGCN_GFX13_GENERIC (0x63)
; ALL:         ]

define amdgpu_kernel void @elf_header() {
  ret void
}
