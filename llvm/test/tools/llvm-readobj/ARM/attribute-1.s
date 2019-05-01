@ RUN: llvm-mc -triple armv7-elf -filetype asm -o - %s | FileCheck %s
@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s \
@ RUN:   | llvm-readobj --arm-attributes - | FileCheck %s --check-prefix=CHECK-OBJ
.eabi_attribute  Tag_CPU_arch, 1
@CHECK:   .eabi_attribute 6, 1
@CHECK-OBJ: Tag: 6
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: CPU_arch
@CHECK-OBJ-NEXT: Description: ARM v4

.eabi_attribute  Tag_ARM_ISA_use, 1
@CHECK:   .eabi_attribute 8, 1
@CHECK-OBJ: Tag: 8
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: ARM_ISA_use
@CHECK-OBJ-NEXT: Description: Permitted

.eabi_attribute  Tag_THUMB_ISA_use, 1
@CHECK:   .eabi_attribute 9, 1
@CHECK-OBJ: Tag: 9
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: THUMB_ISA_use
@CHECK-OBJ-NEXT: Description: Thumb-1

.eabi_attribute  Tag_FP_arch, 1
@CHECK:   .eabi_attribute 10, 1
@CHECK-OBJ: Tag: 10
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: FP_arch
@CHECK-OBJ-NEXT: Description: VFPv1

.eabi_attribute  Tag_WMMX_arch, 1
@CHECK:   .eabi_attribute 11, 1
@CHECK-OBJ: Tag: 11
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: WMMX_arch
@CHECK-OBJ-NEXT: Description: WMMXv1

.eabi_attribute  Tag_Advanced_SIMD_arch, 1
@CHECK:   .eabi_attribute 12, 1
@CHECK-OBJ: Tag: 12
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: Advanced_SIMD_arch
@CHECK-OBJ-NEXT: Description: NEONv1

.eabi_attribute  Tag_PCS_config, 1
@CHECK:   .eabi_attribute 13, 1
@CHECK-OBJ: Tag: 13
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: PCS_config
@CHECK-OBJ-NEXT: Description: Bare Platform

.eabi_attribute  Tag_ABI_PCS_R9_use, 1
@CHECK:   .eabi_attribute 14, 1
@CHECK-OBJ: Tag: 14
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: ABI_PCS_R9_use
@CHECK-OBJ-NEXT: Description: Static Base

.eabi_attribute  Tag_ABI_PCS_RW_data, 1
@CHECK:   .eabi_attribute 15, 1
@CHECK-OBJ: Tag: 15
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: ABI_PCS_RW_data
@CHECK-OBJ-NEXT: Description: PC-relative

.eabi_attribute  Tag_ABI_PCS_RO_data, 1
@CHECK:   .eabi_attribute 16, 1
@CHECK-OBJ: Tag: 16
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: ABI_PCS_RO_data
@CHECK-OBJ-NEXT: Description: PC-relative

.eabi_attribute  Tag_ABI_PCS_GOT_use, 1
@CHECK:   .eabi_attribute 17, 1
@CHECK-OBJ: Tag: 17
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: ABI_PCS_GOT_use
@CHECK-OBJ-NEXT: Description: Direct

.eabi_attribute  Tag_ABI_FP_rounding, 1
@CHECK:   .eabi_attribute 19, 1
@CHECK-OBJ: Tag: 19
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: ABI_FP_rounding
@CHECK-OBJ-NEXT: Description: Runtime

.eabi_attribute  Tag_ABI_FP_denormal, 1
@CHECK:   .eabi_attribute 20, 1
@CHECK-OBJ: Tag: 20
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: ABI_FP_denormal
@CHECK-OBJ-NEXT: Description: IEEE-754

.eabi_attribute  Tag_ABI_FP_exceptions, 1
@CHECK:   .eabi_attribute 21, 1
@CHECK-OBJ: Tag: 21
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: ABI_FP_exceptions
@CHECK-OBJ-NEXT: Description: IEEE-754

.eabi_attribute  Tag_ABI_FP_user_exceptions, 1
@CHECK:   .eabi_attribute 22, 1
@CHECK-OBJ: Tag: 22
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: ABI_FP_user_exceptions
@CHECK-OBJ-NEXT: Description: IEEE-754

.eabi_attribute  Tag_ABI_FP_number_model, 1
@CHECK:   .eabi_attribute 23, 1
@CHECK-OBJ: Tag: 23
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: ABI_FP_number_model
@CHECK-OBJ-NEXT: Description: Finite Only

.eabi_attribute  Tag_ABI_align_needed, 1
@CHECK:   .eabi_attribute 24, 1
@CHECK-OBJ: Tag: 24
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: ABI_align_needed
@CHECK-OBJ-NEXT: Description: 8-byte alignment

.eabi_attribute  Tag_ABI_align_preserved, 1
@CHECK:   .eabi_attribute 25, 1
@CHECK-OBJ: Tag: 25
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: ABI_align_preserved
@CHECK-OBJ-NEXT: Description: 8-byte data alignment

.eabi_attribute  Tag_ABI_enum_size, 1
@CHECK:   .eabi_attribute 26, 1
@CHECK-OBJ: Tag: 26
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: ABI_enum_size
@CHECK-OBJ-NEXT: Description: Packed

.eabi_attribute  Tag_ABI_HardFP_use, 1
@CHECK:   .eabi_attribute 27, 1
@CHECK-OBJ: Tag: 27
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: ABI_HardFP_use
@CHECK-OBJ-NEXT: Description: Single-Precision

.eabi_attribute  Tag_ABI_VFP_args, 1
@CHECK:   .eabi_attribute 28, 1
@CHECK-OBJ: Tag: 28
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: ABI_VFP_args
@CHECK-OBJ-NEXT: Description: AAPCS VFP

.eabi_attribute  Tag_ABI_WMMX_args, 1
@CHECK:   .eabi_attribute 29, 1
@CHECK-OBJ: Tag: 29
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: ABI_WMMX_args
@CHECK-OBJ-NEXT: Description: iWMMX

.eabi_attribute  Tag_ABI_optimization_goals, 1
@CHECK:   .eabi_attribute 30, 1
@CHECK-OBJ: Tag: 30
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: ABI_optimization_goals
@CHECK-OBJ-NEXT: Description: Speed

.eabi_attribute  Tag_ABI_FP_optimization_goals, 1
@CHECK:   .eabi_attribute 31, 1
@CHECK-OBJ: Tag: 31
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: ABI_FP_optimization_goals
@CHECK-OBJ-NEXT: Description: Speed

.eabi_attribute  Tag_compatibility, 1, ""
@CHECK:   .eabi_attribute 32, 1
@CHECK-OBJ: Tag: 32
@CHECK-OBJ-NEXT: Value: 1,
@CHECK-OBJ-NEXT: TagName: compatibility
@CHECK-OBJ-NEXT: Description: AEABI Conformant

.eabi_attribute  Tag_CPU_unaligned_access, 1
@CHECK:   .eabi_attribute 34, 1
@CHECK-OBJ: Tag: 34
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: CPU_unaligned_access
@CHECK-OBJ-NEXT: Description: v6-style

.eabi_attribute  Tag_FP_HP_extension, 1
@CHECK:   .eabi_attribute 36, 1
@CHECK-OBJ: Tag: 36
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: FP_HP_extension
@CHECK-OBJ-NEXT: Description: Permitted

.eabi_attribute  Tag_ABI_FP_16bit_format, 1
@CHECK:   .eabi_attribute 38, 1
@CHECK-OBJ: Tag: 38
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: ABI_FP_16bit_format
@CHECK-OBJ-NEXT: Description: IEEE-754

.eabi_attribute  Tag_MPextension_use, 1
@CHECK:   .eabi_attribute 42, 1
@CHECK-OBJ: Tag: 42
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: MPextension_use
@CHECK-OBJ-NEXT: Description: Permitted

.eabi_attribute  Tag_DIV_use, 1
@CHECK:   .eabi_attribute 44, 1
@CHECK-OBJ: Tag: 44
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: DIV_use
@CHECK-OBJ-NEXT: Description: Not Permitted

.eabi_attribute  Tag_DSP_extension, 1
@CHECK:   .eabi_attribute 46, 1
@CHECK-OBJ: Tag: 46
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: DSP_extension
@CHECK-OBJ-NEXT: Description: Permitted

.eabi_attribute  Tag_Virtualization_use, 1
@CHECK:   .eabi_attribute 68, 1
@CHECK-OBJ: Tag: 68
@CHECK-OBJ-NEXT: Value: 1
@CHECK-OBJ-NEXT: TagName: Virtualization_use
@CHECK-OBJ-NEXT: Description: TrustZone

