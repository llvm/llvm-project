@ RUN: llvm-mc -triple armv7-elf -filetype asm -o - %s | FileCheck %s
@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s \
@ RUN:   | llvm-readobj -arm-attributes - | FileCheck %s --check-prefix=CHECK-OBJ
.eabi_attribute  Tag_CPU_arch, 0
@CHECK:   .eabi_attribute 6, 0
@CHECK-OBJ: Tag: 6
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: CPU_arch
@CHECK-OBJ-NEXT: Description: Pre-v4

.eabi_attribute  Tag_CPU_arch_profile, 0
@CHECK:   .eabi_attribute 7, 0
@CHECK-OBJ: Tag: 7
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: CPU_arch_profile
@CHECK-OBJ-NEXT: Description: None

.eabi_attribute  Tag_ARM_ISA_use, 0
@CHECK:   .eabi_attribute 8, 0
@CHECK-OBJ: Tag: 8
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: ARM_ISA_use
@CHECK-OBJ-NEXT: Description: Not Permitted

.eabi_attribute  Tag_THUMB_ISA_use, 0
@CHECK:   .eabi_attribute 9, 0
@CHECK-OBJ: Tag: 9
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: THUMB_ISA_use
@CHECK-OBJ-NEXT: Description: Not Permitted

.eabi_attribute  Tag_FP_arch, 0
@CHECK:   .eabi_attribute 10, 0
@CHECK-OBJ: Tag: 10
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: FP_arch
@CHECK-OBJ-NEXT: Description: Not Permitted

.eabi_attribute  Tag_WMMX_arch, 0
@CHECK:   .eabi_attribute 11, 0
@CHECK-OBJ: Tag: 11
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: WMMX_arch
@CHECK-OBJ-NEXT: Description: Not Permitted

.eabi_attribute  Tag_Advanced_SIMD_arch, 0
@CHECK:   .eabi_attribute 12, 0
@CHECK-OBJ: Tag: 12
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: Advanced_SIMD_arch
@CHECK-OBJ-NEXT: Description: Not Permitted

.eabi_attribute  Tag_PCS_config, 0
@CHECK:   .eabi_attribute 13, 0
@CHECK-OBJ: Tag: 13
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: PCS_config
@CHECK-OBJ-NEXT: Description: None

.eabi_attribute  Tag_ABI_PCS_R9_use, 0
@CHECK:   .eabi_attribute 14, 0
@CHECK-OBJ: Tag: 14
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: ABI_PCS_R9_use
@CHECK-OBJ-NEXT: Description: v6

.eabi_attribute  Tag_ABI_PCS_RW_data, 0
@CHECK:   .eabi_attribute 15, 0
@CHECK-OBJ: Tag: 15
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: ABI_PCS_RW_data
@CHECK-OBJ-NEXT: Description: Absolute

.eabi_attribute  Tag_ABI_PCS_RO_data, 0
@CHECK:   .eabi_attribute 16, 0
@CHECK-OBJ: Tag: 16
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: ABI_PCS_RO_data
@CHECK-OBJ-NEXT: Description: Absolute

.eabi_attribute  Tag_ABI_PCS_GOT_use, 0
@CHECK:   .eabi_attribute 17, 0
@CHECK-OBJ: Tag: 17
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: ABI_PCS_GOT_use
@CHECK-OBJ-NEXT: Description: Not Permitted

.eabi_attribute  Tag_ABI_PCS_wchar_t, 0
@CHECK:   .eabi_attribute 18, 0
@CHECK-OBJ: Tag: 18
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: ABI_PCS_wchar_t
@CHECK-OBJ-NEXT: Description: Not Permitted

.eabi_attribute  Tag_ABI_FP_rounding, 0
@CHECK:   .eabi_attribute 19, 0
@CHECK-OBJ: Tag: 19
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: ABI_FP_rounding
@CHECK-OBJ-NEXT: Description: IEEE-754

.eabi_attribute  Tag_ABI_FP_denormal, 0
@CHECK:   .eabi_attribute 20, 0
@CHECK-OBJ: Tag: 20
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: ABI_FP_denormal
@CHECK-OBJ-NEXT: Description: Unsupported

.eabi_attribute  Tag_ABI_FP_exceptions, 0
@CHECK:   .eabi_attribute 21, 0
@CHECK-OBJ: Tag: 21
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: ABI_FP_exceptions
@CHECK-OBJ-NEXT: Description: Not Permitted

.eabi_attribute  Tag_ABI_FP_user_exceptions, 0
@CHECK:   .eabi_attribute 22, 0
@CHECK-OBJ: Tag: 22
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: ABI_FP_user_exceptions
@CHECK-OBJ-NEXT: Description: Not Permitted

.eabi_attribute  Tag_ABI_FP_number_model, 0
@CHECK:   .eabi_attribute 23, 0
@CHECK-OBJ: Tag: 23
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: ABI_FP_number_model
@CHECK-OBJ-NEXT: Description: Not Permitted

.eabi_attribute  Tag_ABI_align_needed, 0
@CHECK:   .eabi_attribute 24, 0
@CHECK-OBJ: Tag: 24
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: ABI_align_needed
@CHECK-OBJ-NEXT: Description: Not Permitted

.eabi_attribute  Tag_ABI_align_preserved, 0
@CHECK:   .eabi_attribute 25, 0
@CHECK-OBJ: Tag: 25
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: ABI_align_preserved
@CHECK-OBJ-NEXT: Description: Not Required

.eabi_attribute  Tag_ABI_enum_size, 0
@CHECK:   .eabi_attribute 26, 0
@CHECK-OBJ: Tag: 26
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: ABI_enum_size
@CHECK-OBJ-NEXT: Description: Not Permitted

.eabi_attribute  Tag_ABI_HardFP_use, 0
@CHECK:   .eabi_attribute 27, 0
@CHECK-OBJ: Tag: 27
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: ABI_HardFP_use
@CHECK-OBJ-NEXT: Description: Tag_FP_arch

.eabi_attribute  Tag_ABI_VFP_args, 0
@CHECK:   .eabi_attribute 28, 0
@CHECK-OBJ: Tag: 28
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: ABI_VFP_args
@CHECK-OBJ-NEXT: Description: AAPCS

.eabi_attribute  Tag_ABI_WMMX_args, 0
@CHECK:   .eabi_attribute 29, 0
@CHECK-OBJ: Tag: 29
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: ABI_WMMX_args
@CHECK-OBJ-NEXT: Description: AAPCS

.eabi_attribute  Tag_ABI_optimization_goals, 0
@CHECK:   .eabi_attribute 30, 0
@CHECK-OBJ: Tag: 30
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: ABI_optimization_goals
@CHECK-OBJ-NEXT: Description: None

.eabi_attribute  Tag_ABI_FP_optimization_goals, 0
@CHECK:   .eabi_attribute 31, 0
@CHECK-OBJ: Tag: 31
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: ABI_FP_optimization_goals
@CHECK-OBJ-NEXT: Description: None

.eabi_attribute  Tag_compatibility, 0, "ARM"
@CHECK:   .eabi_attribute 32, 0
@CHECK-OBJ: Tag: 32
@CHECK-OBJ-NEXT: Value: 0, ARM
@CHECK-OBJ-NEXT: TagName: compatibility
@CHECK-OBJ-NEXT: Description: No Specific Requirements

.eabi_attribute  Tag_CPU_unaligned_access, 0
@CHECK:   .eabi_attribute 34, 0
@CHECK-OBJ: Tag: 34
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: CPU_unaligned_access
@CHECK-OBJ-NEXT: Description: Not Permitted

.eabi_attribute  Tag_FP_HP_extension, 0
@CHECK:   .eabi_attribute 36, 0
@CHECK-OBJ: Tag: 36
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: FP_HP_extension
@CHECK-OBJ-NEXT: Description: If Available

.eabi_attribute  Tag_ABI_FP_16bit_format, 0
@CHECK:   .eabi_attribute 38, 0
@CHECK-OBJ: Tag: 38
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: ABI_FP_16bit_format
@CHECK-OBJ-NEXT: Description: Not Permitted

.eabi_attribute  Tag_MPextension_use, 0
@CHECK:   .eabi_attribute 42, 0
@CHECK-OBJ: Tag: 42
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: MPextension_use
@CHECK-OBJ-NEXT: Description: Not Permitted

.eabi_attribute  Tag_DIV_use, 0
@CHECK:   .eabi_attribute 44, 0
@CHECK-OBJ: Tag: 44
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: DIV_use
@CHECK-OBJ-NEXT: Description: If Available

.eabi_attribute  Tag_DSP_extension, 0
@CHECK:   .eabi_attribute 46, 0
@CHECK-OBJ: Tag: 46
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: DSP_extension
@CHECK-OBJ-NEXT: Description: Not Permitted

.eabi_attribute  Tag_Virtualization_use, 0
@CHECK:   .eabi_attribute 68, 0
@CHECK-OBJ: Tag: 68
@CHECK-OBJ-NEXT: Value: 0
@CHECK-OBJ-NEXT: TagName: Virtualization_use
@CHECK-OBJ-NEXT: Description: Not Permitted

