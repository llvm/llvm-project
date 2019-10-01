@ RUN: llvm-mc -triple armv7-elf -filetype asm -o - %s | FileCheck %s
@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s \
@ RUN:   | llvm-readobj --arm-attributes - | FileCheck %s --check-prefix=CHECK-OBJ
.eabi_attribute  Tag_CPU_arch, 2
@CHECK:   .eabi_attribute 6, 2
@CHECK-OBJ: Tag: 6
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: CPU_arch
@CHECK-OBJ-NEXT: Description: ARM v4T

.eabi_attribute  Tag_THUMB_ISA_use, 2
@CHECK:   .eabi_attribute 9, 2
@CHECK-OBJ: Tag: 9
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: THUMB_ISA_use
@CHECK-OBJ-NEXT: Description: Thumb-2

.eabi_attribute  Tag_FP_arch, 2
@CHECK:   .eabi_attribute 10, 2
@CHECK-OBJ: Tag: 10
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: FP_arch
@CHECK-OBJ-NEXT: Description: VFPv2

.eabi_attribute  Tag_WMMX_arch, 2
@CHECK:   .eabi_attribute 11, 2
@CHECK-OBJ: Tag: 11
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: WMMX_arch
@CHECK-OBJ-NEXT: Description: WMMXv2

.eabi_attribute  Tag_Advanced_SIMD_arch, 2
@CHECK:   .eabi_attribute 12, 2
@CHECK-OBJ: Tag: 12
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: Advanced_SIMD_arch
@CHECK-OBJ-NEXT: Description: NEONv2+FMA

.eabi_attribute  Tag_PCS_config, 2
@CHECK:   .eabi_attribute 13, 2
@CHECK-OBJ: Tag: 13
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: PCS_config
@CHECK-OBJ-NEXT: Description: Linux Application

.eabi_attribute  Tag_ABI_PCS_R9_use, 2
@CHECK:   .eabi_attribute 14, 2
@CHECK-OBJ: Tag: 14
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: ABI_PCS_R9_use
@CHECK-OBJ-NEXT: Description: TLS

.eabi_attribute  Tag_ABI_PCS_RW_data, 2
@CHECK:   .eabi_attribute 15, 2
@CHECK-OBJ: Tag: 15
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: ABI_PCS_RW_data
@CHECK-OBJ-NEXT: Description: SB-relative

.eabi_attribute  Tag_ABI_PCS_RO_data, 2
@CHECK:   .eabi_attribute 16, 2
@CHECK-OBJ: Tag: 16
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: ABI_PCS_RO_data
@CHECK-OBJ-NEXT: Description: Not Permitted

.eabi_attribute  Tag_ABI_PCS_GOT_use, 2
@CHECK:   .eabi_attribute 17, 2
@CHECK-OBJ: Tag: 17
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: ABI_PCS_GOT_use
@CHECK-OBJ-NEXT: Description: GOT-Indirect

.eabi_attribute  Tag_ABI_PCS_wchar_t, 2
@CHECK:   .eabi_attribute 18, 2
@CHECK-OBJ: Tag: 18
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: ABI_PCS_wchar_t
@CHECK-OBJ-NEXT: Description: 2-byte

.eabi_attribute  Tag_ABI_FP_denormal, 2
@CHECK:   .eabi_attribute 20, 2
@CHECK-OBJ: Tag: 20
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: ABI_FP_denormal
@CHECK-OBJ-NEXT: Description: Sign Only

.eabi_attribute  Tag_ABI_FP_number_model, 2
@CHECK:   .eabi_attribute 23, 2
@CHECK-OBJ: Tag: 23
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: ABI_FP_number_model
@CHECK-OBJ-NEXT: Description: RTABI

.eabi_attribute  Tag_ABI_align_needed, 2
@CHECK:   .eabi_attribute 24, 2
@CHECK-OBJ: Tag: 24
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: ABI_align_needed
@CHECK-OBJ-NEXT: Description: 4-byte alignment

.eabi_attribute  Tag_ABI_align_preserved, 2
@CHECK:   .eabi_attribute 25, 2
@CHECK-OBJ: Tag: 25
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: ABI_align_preserved
@CHECK-OBJ-NEXT: Description: 8-byte data and code alignment

.eabi_attribute  Tag_ABI_enum_size, 2
@CHECK:   .eabi_attribute 26, 2
@CHECK-OBJ: Tag: 26
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: ABI_enum_size
@CHECK-OBJ-NEXT: Description: Int32

.eabi_attribute  Tag_ABI_HardFP_use, 2
@CHECK:   .eabi_attribute 27, 2
@CHECK-OBJ: Tag: 27
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: ABI_HardFP_use
@CHECK-OBJ-NEXT: Description: Reserved

.eabi_attribute  Tag_ABI_VFP_args, 2
@CHECK:   .eabi_attribute 28, 2
@CHECK-OBJ: Tag: 28
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: ABI_VFP_args
@CHECK-OBJ-NEXT: Description: Custom

.eabi_attribute  Tag_ABI_WMMX_args, 2
@CHECK:   .eabi_attribute 29, 2
@CHECK-OBJ: Tag: 29
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: ABI_WMMX_args
@CHECK-OBJ-NEXT: Description: Custom

.eabi_attribute  Tag_ABI_optimization_goals, 2
@CHECK:   .eabi_attribute 30, 2
@CHECK-OBJ: Tag: 30
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: ABI_optimization_goals
@CHECK-OBJ-NEXT: Description: Aggressive Speed

.eabi_attribute  Tag_ABI_FP_optimization_goals, 2
@CHECK:   .eabi_attribute 31, 2
@CHECK-OBJ: Tag: 31
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: ABI_FP_optimization_goals
@CHECK-OBJ-NEXT: Description: Aggressive Speed

.eabi_attribute  Tag_compatibility, 2, ""
@CHECK:   .eabi_attribute 32, 2
@CHECK-OBJ: Tag: 32
@CHECK-OBJ-NEXT: Value: 2,
@CHECK-OBJ-NEXT: TagName: compatibility
@CHECK-OBJ-NEXT: Description: AEABI Non-Conformant

.eabi_attribute  Tag_ABI_FP_16bit_format, 2
@CHECK:   .eabi_attribute 38, 2
@CHECK-OBJ: Tag: 38
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: ABI_FP_16bit_format
@CHECK-OBJ-NEXT: Description: VFPv3

.eabi_attribute  Tag_DIV_use, 2
@CHECK:   .eabi_attribute 44, 2
@CHECK-OBJ: Tag: 44
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: DIV_use
@CHECK-OBJ-NEXT: Description: Permitted

.eabi_attribute  Tag_Virtualization_use, 2
@CHECK:   .eabi_attribute 68, 2
@CHECK-OBJ: Tag: 68
@CHECK-OBJ-NEXT: Value: 2
@CHECK-OBJ-NEXT: TagName: Virtualization_use
@CHECK-OBJ-NEXT: Description: Virtualization Extensions

