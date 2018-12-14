; RUN: llc < %s -march=amdgcn -mcpu=fiji -filetype=obj | llvm-readobj -symbols -s -sd - | FileCheck %s

; CHECK: Section {
; CHECK: Name: .AMDGPU.metadata.info_1
; CHECK: Type: SHT_PROGBITS (0x1)
; CHECK: Flags [ (0x0)
; CHECK: Size: 16
; CHECK: SectionData (
; CHECK: 0000: 414D4431 414D4431 414D4431 414D4431  |AMD1AMD1AMD1AMD1|
; CHECK: )
; CHECK: }

; CHECK: Section {
; CHECK: Name: .AMDGPU.metadata.info_2
; CHECK: Type: SHT_PROGBITS (0x1)
; CHECK: Flags [ (0x0)
; CHECK: Size: 16
; CHECK: SectionData (
; CHECK: 0000: 414D4432 414D4432 414D4432 414D4432  |AMD2AMD2AMD2AMD2|
; CHECK: )
; CHECK: }

; CHECK: Section {
; CHECK: Name: .AMDGPU.metadata.info_3
; CHECK: Type: SHT_PROGBITS (0x1)
; CHECK: Flags [ (0x0)
; CHECK: Size: 16
; CHECK: SectionData (
; CHECK: 0000: 414D4433 414D4433 414D4433 414D4433  |AMD3AMD3AMD3AMD3|
; CHECK: )
; CHECK: }

; CHECK: Symbol {
; CHECK: Name: metadata_info_var_1
; CHECK: Size: 16
; CHECK: Binding: Local
; CHECK: Section: .AMDGPU.metadata.info_1
; CHECK: }

; CHECK: Symbol {
; CHECK: Name: metadata_info_var_2
; CHECK: Size: 16
; CHECK: Binding: Global
; CHECK: Section: .AMDGPU.metadata.info_2
; CHECK: }

; CHECK: Symbol {
; CHECK: Name: metadata_info_var_3
; CHECK: Size: 16
; CHECK: Binding: Global
; CHECK: Section: .AMDGPU.metadata.info_3
; CHECK: }

@metadata_info_var_1 = internal global [4 x i32][i32 826559809, i32 826559809, i32 826559809, i32 826559809], align 1, section ".AMDGPU.metadata.info_1"
@metadata_info_var_2 = constant [4 x i32][i32 843337025, i32 843337025, i32 843337025, i32 843337025], align 1, section ".AMDGPU.metadata.info_2"
@metadata_info_var_3 = global [4 x i32][i32 860114241, i32 860114241, i32 860114241, i32 860114241], align 1, section ".AMDGPU.metadata.info_3"
