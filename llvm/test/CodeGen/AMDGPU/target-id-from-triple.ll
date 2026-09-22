; RUN: llc -mtriple=amdgpu6-amd-amdhsa < %s | FileCheck -check-prefix=GFX6 %s
; RUN: llc -mtriple=amdgpu6.00-amd-amdhsa < %s | FileCheck -check-prefix=GFX600 %s
; RUN: llc -mtriple=amdgpu6.01-amd-amdhsa < %s | FileCheck -check-prefix=GFX601 %s
; RUN: llc -mtriple=amdgpu6.02-amd-amdhsa < %s | FileCheck -check-prefix=GFX602 %s

; RUN: llc -mtriple=amdgpu7-amd-amdhsa < %s | FileCheck -check-prefix=GFX7 %s
; RUN: llc -mtriple=amdgpu7.00-amd-amdhsa < %s | FileCheck -check-prefix=GFX700 %s
; RUN: llc -mtriple=amdgpu7.01-amd-amdhsa < %s | FileCheck -check-prefix=GFX701 %s
; RUN: llc -mtriple=amdgpu7.02-amd-amdhsa < %s | FileCheck -check-prefix=GFX702 %s
; RUN: llc -mtriple=amdgpu7.03-amd-amdhsa < %s | FileCheck -check-prefix=GFX703 %s
; RUN: llc -mtriple=amdgpu7.04-amd-amdhsa < %s | FileCheck -check-prefix=GFX704 %s
; RUN: llc -mtriple=amdgpu7.05-amd-amdhsa < %s | FileCheck -check-prefix=GFX705 %s

; RUN: llc -mtriple=amdgpu8.01-amd-amdhsa < %s | FileCheck -check-prefix=GFX801 %s
; RUN: llc -mtriple=amdgpu8.02-amd-amdhsa < %s | FileCheck -check-prefix=GFX802 %s
; RUN: llc -mtriple=amdgpu8.03-amd-amdhsa < %s | FileCheck -check-prefix=GFX803 %s
; RUN: llc -mtriple=amdgpu8.05-amd-amdhsa < %s | FileCheck -check-prefix=GFX805 %s
; RUN: llc -mtriple=amdgpu8.10-amd-amdhsa < %s | FileCheck -check-prefix=GFX810 %s

; RUN: llc -mtriple=amdgpu9-amd-amdhsa < %s | FileCheck -check-prefix=GFX9 %s
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa < %s | FileCheck -check-prefix=GFX900 %s
; RUN: llc -mtriple=amdgpu9.02-amd-amdhsa < %s | FileCheck -check-prefix=GFX902 %s
; RUN: llc -mtriple=amdgpu9.04-amd-amdhsa < %s | FileCheck -check-prefix=GFX904 %s
; RUN: llc -mtriple=amdgpu9.06-amd-amdhsa < %s | FileCheck -check-prefix=GFX906 %s
; RUN: llc -mtriple=amdgpu9.08-amd-amdhsa < %s | FileCheck -check-prefix=GFX908 %s
; RUN: llc -mtriple=amdgpu9.09-amd-amdhsa < %s | FileCheck -check-prefix=GFX909 %s
; RUN: llc -mtriple=amdgpu9.0a-amd-amdhsa < %s | FileCheck -check-prefix=GFX90A %s
; RUN: llc -mtriple=amdgpu9.0c-amd-amdhsa < %s | FileCheck -check-prefix=GFX90C %s
; RUN: llc -mtriple=amdgpu9.4-amd-amdhsa < %s | FileCheck -check-prefix=GFX94 %s
; RUN: llc -mtriple=amdgpu9.42-amd-amdhsa < %s | FileCheck -check-prefix=GFX942 %s
; RUN: llc -mtriple=amdgpu9.50-amd-amdhsa < %s | FileCheck -check-prefix=GFX950 %s

; RUN: llc -mtriple=amdgpu10.1-amd-amdhsa < %s | FileCheck -check-prefix=GFX101 %s
; RUN: llc -mtriple=amdgpu10.10-amd-amdhsa < %s | FileCheck -check-prefix=GFX1010 %s
; RUN: llc -mtriple=amdgpu10.11-amd-amdhsa < %s | FileCheck -check-prefix=GFX1011 %s
; RUN: llc -mtriple=amdgpu10.12-amd-amdhsa < %s | FileCheck -check-prefix=GFX1012 %s
; RUN: llc -mtriple=amdgpu10.13-amd-amdhsa < %s | FileCheck -check-prefix=GFX1013 %s

; RUN: llc -mtriple=amdgpu10.3-amd-amdhsa < %s | FileCheck -check-prefix=GFX103 %s
; RUN: llc -mtriple=amdgpu10.30-amd-amdhsa < %s | FileCheck -check-prefix=GFX1030 %s
; RUN: llc -mtriple=amdgpu10.31-amd-amdhsa < %s | FileCheck -check-prefix=GFX1031 %s
; RUN: llc -mtriple=amdgpu10.32-amd-amdhsa < %s | FileCheck -check-prefix=GFX1032 %s
; RUN: llc -mtriple=amdgpu10.33-amd-amdhsa < %s | FileCheck -check-prefix=GFX1033 %s
; RUN: llc -mtriple=amdgpu10.34-amd-amdhsa < %s | FileCheck -check-prefix=GFX1034 %s
; RUN: llc -mtriple=amdgpu10.35-amd-amdhsa < %s | FileCheck -check-prefix=GFX1035 %s
; RUN: llc -mtriple=amdgpu10.36-amd-amdhsa < %s | FileCheck -check-prefix=GFX1036 %s

; RUN: llc -mtriple=amdgpu11-amd-amdhsa < %s | FileCheck -check-prefix=GFX11 %s
; RUN: llc -mtriple=amdgpu11.00-amd-amdhsa < %s | FileCheck -check-prefix=GFX1100 %s
; RUN: llc -mtriple=amdgpu11.01-amd-amdhsa < %s | FileCheck -check-prefix=GFX1101 %s
; RUN: llc -mtriple=amdgpu11.02-amd-amdhsa < %s | FileCheck -check-prefix=GFX1102 %s
; RUN: llc -mtriple=amdgpu11.03-amd-amdhsa < %s | FileCheck -check-prefix=GFX1103 %s
; RUN: llc -mtriple=amdgpu11.50-amd-amdhsa < %s | FileCheck -check-prefix=GFX1150 %s
; RUN: llc -mtriple=amdgpu11.51-amd-amdhsa < %s | FileCheck -check-prefix=GFX1151 %s
; RUN: llc -mtriple=amdgpu11.52-amd-amdhsa < %s | FileCheck -check-prefix=GFX1152 %s
; RUN: llc -mtriple=amdgpu11.53-amd-amdhsa < %s | FileCheck -check-prefix=GFX1153 %s
; RUN: llc -mtriple=amdgpu11.54-amd-amdhsa < %s | FileCheck -check-prefix=GFX1154 %s
; RUN: llc -mtriple=amdgpu11.7-amd-amdhsa < %s | FileCheck -check-prefix=GFX11_7 %s
; RUN: llc -mtriple=amdgpu11.70-amd-amdhsa < %s | FileCheck -check-prefix=GFX1170 %s
; RUN: llc -mtriple=amdgpu11.71-amd-amdhsa < %s | FileCheck -check-prefix=GFX1171 %s
; RUN: llc -mtriple=amdgpu11.72-amd-amdhsa < %s | FileCheck -check-prefix=GFX1172 %s

; RUN: llc -mtriple=amdgpu12-amd-amdhsa < %s | FileCheck -check-prefix=GFX12 %s
; RUN: llc -mtriple=amdgpu12.00-amd-amdhsa < %s | FileCheck -check-prefix=GFX1200 %s
; RUN: llc -mtriple=amdgpu12.01-amd-amdhsa < %s | FileCheck -check-prefix=GFX1201 %s

; RUN: llc -mtriple=amdgpu12.5-amd-amdhsa < %s | FileCheck -check-prefix=GFX12_5 %s
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa < %s | FileCheck -check-prefix=GFX1250 %s
; RUN: llc -mtriple=amdgpu12.51-amd-amdhsa < %s | FileCheck -check-prefix=GFX1251 %s

; RUN: llc -mtriple=amdgpu13-amd-amdhsa < %s | FileCheck -check-prefix=GFX13 %s
; RUN: llc -mtriple=amdgpu13.10-amd-amdhsa < %s | FileCheck -check-prefix=GFX1310 %s

; A major-family subarch combined with a more specific (valid) -mcpu keeps
; the specific processor rather than the subarch's generic/default one.
; RUN: llc -mtriple=amdgpu6-amd-amdhsa -mcpu=gfx602 < %s | FileCheck -check-prefix=MAJOR-GFX602 %s
; RUN: llc -mtriple=amdgpu8-amd-amdhsa -mcpu=gfx803 < %s | FileCheck -check-prefix=MAJOR-GFX803 %s
; RUN: llc -mtriple=amdgpu9-amd-amdhsa -mcpu=gfx906 < %s | FileCheck -check-prefix=MAJOR-GFX906 %s
; RUN: llc -mtriple=amdgpu12-amd-amdhsa -mcpu=gfx1201 < %s | FileCheck -check-prefix=MAJOR-GFX1201 %s

; MAJOR-GFX602: .amdgcn_target "amdgpu6-amd-amdhsa-unknown-gfx602"
; MAJOR-GFX803: .amdgcn_target "amdgpu8-amd-amdhsa-unknown-gfx803"
; MAJOR-GFX906: .amdgcn_target "amdgpu9-amd-amdhsa-unknown-gfx906"
; MAJOR-GFX1201: .amdgcn_target "amdgpu12-amd-amdhsa-unknown-gfx1201"

; An unrecognized -mcpu is ignored. The resulting target ID has an
; empty processor field.
; RUN: llc -mtriple=amdgpu6.00-amd-amdhsa -mcpu=invalid < %s | FileCheck -check-prefix=INVALID-CPU-GFX6 %s
; RUN: llc -mtriple=amdgpu9-amd-amdhsa -mcpu=invalid < %s | FileCheck -check-prefix=INVALID-CPU-GFX9 %s
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -mcpu=invalid < %s | FileCheck -check-prefix=INVALID-CPU-GFX1250 %s

; INVALID-CPU-GFX6: .amdgcn_target "amdgpu6.00-amd-amdhsa-unknown-"
; INVALID-CPU-GFX9: .amdgcn_target "amdgpu9-amd-amdhsa-unknown-"
; INVALID-CPU-GFX1250: .amdgcn_target "amdgpu12.50-amd-amdhsa-unknown-"

; GFX6: .amdgcn_target "amdgpu6-amd-amdhsa-unknown-gfx600"
; GFX600: .amdgcn_target "amdgpu6.00-amd-amdhsa-unknown-gfx600"
; GFX601: .amdgcn_target "amdgpu6.01-amd-amdhsa-unknown-gfx601"
; GFX602: .amdgcn_target "amdgpu6.02-amd-amdhsa-unknown-gfx602"

; GFX7: .amdgcn_target "amdgpu7-amd-amdhsa-unknown-gfx700"
; GFX700: .amdgcn_target "amdgpu7.00-amd-amdhsa-unknown-gfx700"
; GFX701: .amdgcn_target "amdgpu7.01-amd-amdhsa-unknown-gfx701"
; GFX702: .amdgcn_target "amdgpu7.02-amd-amdhsa-unknown-gfx702"
; GFX703: .amdgcn_target "amdgpu7.03-amd-amdhsa-unknown-gfx703"
; GFX704: .amdgcn_target "amdgpu7.04-amd-amdhsa-unknown-gfx704"
; GFX705: .amdgcn_target "amdgpu7.05-amd-amdhsa-unknown-gfx705"

; GFX8: .amdgcn_target "amdgpu8-amd-amdhsa-unknown-gfx800"
; GFX801: .amdgcn_target "amdgpu8.01-amd-amdhsa-unknown-gfx801"
; GFX802: .amdgcn_target "amdgpu8.02-amd-amdhsa-unknown-gfx802"
; GFX803: .amdgcn_target "amdgpu8.03-amd-amdhsa-unknown-gfx803"
; GFX805: .amdgcn_target "amdgpu8.05-amd-amdhsa-unknown-gfx805"

; GFX810: .amdgcn_target "amdgpu8.10-amd-amdhsa-unknown-gfx810"

; GFX9: .amdgcn_target "amdgpu9-amd-amdhsa-unknown-gfx9-generic"
; GFX900: .amdgcn_target "amdgpu9.00-amd-amdhsa-unknown-gfx900"
; GFX902: .amdgcn_target "amdgpu9.02-amd-amdhsa-unknown-gfx902"
; GFX904: .amdgcn_target "amdgpu9.04-amd-amdhsa-unknown-gfx904"
; GFX906: .amdgcn_target "amdgpu9.06-amd-amdhsa-unknown-gfx906"
; GFX908: .amdgcn_target "amdgpu9.08-amd-amdhsa-unknown-gfx908"
; GFX909: .amdgcn_target "amdgpu9.09-amd-amdhsa-unknown-gfx909"
; GFX90A: .amdgcn_target "amdgpu9.0a-amd-amdhsa-unknown-gfx90a"
; GFX90C: .amdgcn_target "amdgpu9.0c-amd-amdhsa-unknown-gfx90c"

; GFX94: .amdgcn_target "amdgpu9.4-amd-amdhsa-unknown-gfx9-4-generic"
; GFX942: .amdgcn_target "amdgpu9.42-amd-amdhsa-unknown-gfx942"
; GFX950: .amdgcn_target "amdgpu9.50-amd-amdhsa-unknown-gfx950"

; GFX101: .amdgcn_target "amdgpu10.1-amd-amdhsa-unknown-gfx10-1-generic"
; GFX1010: .amdgcn_target "amdgpu10.10-amd-amdhsa-unknown-gfx1010"
; GFX1011: .amdgcn_target "amdgpu10.11-amd-amdhsa-unknown-gfx1011"
; GFX1012: .amdgcn_target "amdgpu10.12-amd-amdhsa-unknown-gfx1012"
; GFX1013: .amdgcn_target "amdgpu10.13-amd-amdhsa-unknown-gfx1013"


; GFX103: .amdgcn_target "amdgpu10.3-amd-amdhsa-unknown-gfx10-3-generic"
; GFX1030: .amdgcn_target "amdgpu10.30-amd-amdhsa-unknown-gfx1030"
; GFX1031: .amdgcn_target "amdgpu10.31-amd-amdhsa-unknown-gfx1031"
; GFX1032: .amdgcn_target "amdgpu10.32-amd-amdhsa-unknown-gfx1032"
; GFX1033: .amdgcn_target "amdgpu10.33-amd-amdhsa-unknown-gfx1033"
; GFX1034: .amdgcn_target "amdgpu10.34-amd-amdhsa-unknown-gfx1034"
; GFX1035: .amdgcn_target "amdgpu10.35-amd-amdhsa-unknown-gfx1035"
; GFX1036: .amdgcn_target "amdgpu10.36-amd-amdhsa-unknown-gfx1036"

; GFX11: .amdgcn_target "amdgpu11-amd-amdhsa-unknown-gfx11-generic"
; GFX1100: .amdgcn_target "amdgpu11.00-amd-amdhsa-unknown-gfx1100"
; GFX1101: .amdgcn_target "amdgpu11.01-amd-amdhsa-unknown-gfx1101"
; GFX1102: .amdgcn_target "amdgpu11.02-amd-amdhsa-unknown-gfx1102"
; GFX1103: .amdgcn_target "amdgpu11.03-amd-amdhsa-unknown-gfx1103"

; GFX1150: .amdgcn_target "amdgpu11.50-amd-amdhsa-unknown-gfx1150"
; GFX1151: .amdgcn_target "amdgpu11.51-amd-amdhsa-unknown-gfx1151"
; GFX1152: .amdgcn_target "amdgpu11.52-amd-amdhsa-unknown-gfx1152"
; GFX1153: .amdgcn_target "amdgpu11.53-amd-amdhsa-unknown-gfx1153"
; GFX1154: .amdgcn_target "amdgpu11.54-amd-amdhsa-unknown-gfx1154"

; GFX11_7: .amdgcn_target "amdgpu11.7-amd-amdhsa-unknown-gfx11-7-generic"
; GFX1170: .amdgcn_target "amdgpu11.70-amd-amdhsa-unknown-gfx1170"
; GFX1171: .amdgcn_target "amdgpu11.71-amd-amdhsa-unknown-gfx1171"
; GFX1172: .amdgcn_target "amdgpu11.72-amd-amdhsa-unknown-gfx1172"

; GFX12: .amdgcn_target "amdgpu12-amd-amdhsa-unknown-gfx12-generic"
; GFX1200: .amdgcn_target "amdgpu12.00-amd-amdhsa-unknown-gfx1200"
; GFX1201: .amdgcn_target "amdgpu12.01-amd-amdhsa-unknown-gfx1201"

; GFX12_5: .amdgcn_target "amdgpu12.5-amd-amdhsa-unknown-gfx12-5-generic"
; GFX1250: .amdgcn_target "amdgpu12.50-amd-amdhsa-unknown-gfx1250"
; GFX1251: .amdgcn_target "amdgpu12.51-amd-amdhsa-unknown-gfx1251"

; GFX13: .amdgcn_target "amdgpu13-amd-amdhsa-unknown-gfx13-generic"
; GFX1310: .amdgcn_target "amdgpu13.10-amd-amdhsa-unknown-gfx1310"

define amdgpu_kernel void @foo() {
  ret void
}
