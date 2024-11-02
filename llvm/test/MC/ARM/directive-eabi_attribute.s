@ RUN: llvm-mc -triple armv7-elf -filetype asm -o - %s | FileCheck %s
@ RUN: llvm-mc < %s -triple armv7-unknown-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj --arch-specific - | FileCheck %s --check-prefix=CHECK-OBJ

        .syntax unified
        .thumb

        .eabi_attribute Tag_conformance, "2.09"
@ CHECK: .eabi_attribute 67, "2.09"
@ Tag_conformance should be be emitted first in a file-scope
@ sub-subsection of the first public subsection of the attributes
@ section. 2.3.7.4 of ABI Addenda.
@ CHECK-OBJ:        Tag: 67
@ CHECK-OBJ-NEXT:   TagName: conformance
@ CHECK-OBJ-NEXT:   Value: 2.09
	.eabi_attribute Tag_CPU_raw_name, "Cortex-A9"
@ CHECK: .eabi_attribute 4, "Cortex-A9"
@ CHECK-OBJ:        Tag: 4
@ CHECK-OBJ-NEXT:   TagName: CPU_raw_name
@ CHECK-OBJ-NEXT:   Value: Cortex-A9
	.eabi_attribute Tag_CPU_name, "cortex-a9"
@ CHECK: .cpu cortex-a9
@ CHECK-OBJ:        Tag: 5
@ CHECK-OBJ-NEXT:   TagName: CPU_name
@ CHECK-OBJ-NEXT:   Value: cortex-a9
	.eabi_attribute Tag_CPU_arch, 10
@ CHECK: .eabi_attribute 6, 10
@ CHECK-OBJ:        Tag: 6
@ CHECK-OBJ-NEXT:   Value: 10
@ CHECK-OBJ-NEXT:   TagName: CPU_arch
@ CHECK-OBJ-NEXT:   Description: ARM v7
	.eabi_attribute Tag_CPU_arch_profile, 'A'
@ CHECK: .eabi_attribute 7, 65
@ CHECK-OBJ:        Tag: 7
@ CHECK-OBJ-NEXT:   Value: 65
@ CHECK-OBJ-NEXT:   TagName: CPU_arch_profile
@ CHECK-OBJ-NEXT:   Description: Application
	.eabi_attribute Tag_ARM_ISA_use, 0
@ CHECK: .eabi_attribute 8, 0
@ CHECK-OBJ:        Tag: 8
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: ARM_ISA_use
@ CHECK-OBJ-NEXT:   Description: Not Permitted
	.eabi_attribute Tag_THUMB_ISA_use, 2
@ CHECK: .eabi_attribute 9, 2
@ CHECK-OBJ:        Tag: 9
@ CHECK-OBJ-NEXT:   Value: 2
@ CHECK-OBJ-NEXT:   TagName: THUMB_ISA_use
@ CHECK-OBJ-NEXT:   Description: Thumb-2
	.eabi_attribute Tag_FP_arch, 3
@ CHECK: .eabi_attribute 10, 3
@ CHECK-OBJ:        Tag: 10
@ CHECK-OBJ-NEXT:   Value: 3
@ CHECK-OBJ-NEXT:   TagName: FP_arch
@ CHECK-OBJ-NEXT:   Description: VFPv3
	.eabi_attribute Tag_WMMX_arch, 0
@ CHECK: .eabi_attribute 11, 0
@ CHECK-OBJ:        Tag: 11
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: WMMX_arch
@ CHECK-OBJ-NEXT:   Description: Not Permitted
	.eabi_attribute Tag_Advanced_SIMD_arch, 1
@ CHECK: .eabi_attribute 12, 1
@ CHECK-OBJ:        Tag: 12
@ CHECK-OBJ-NEXT:   Value: 1
@ CHECK-OBJ-NEXT:   TagName: Advanced_SIMD_arch
@ CHECK-OBJ-NEXT:   Description: NEONv1
	.eabi_attribute Tag_PCS_config, 2
@ CHECK: .eabi_attribute 13, 2
@ CHECK-OBJ:        Tag: 13
@ CHECK-OBJ-NEXT:   Value: 2
@ CHECK-OBJ-NEXT:   TagName: PCS_config
@ CHECK-OBJ-NEXT:   Description: Linux Application
	.eabi_attribute Tag_ABI_PCS_R9_use, 0
@ CHECK: .eabi_attribute 14, 0
@ CHECK-OBJ:        Tag: 14
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: ABI_PCS_R9_use
@ CHECK-OBJ-NEXT:   Description: v6
	.eabi_attribute Tag_ABI_PCS_RW_data, 0
@ CHECK: .eabi_attribute 15, 0
@ CHECK-OBJ:        Tag: 15
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: ABI_PCS_RW_data
@ CHECK-OBJ-NEXT:   Description: Absolute
	.eabi_attribute Tag_ABI_PCS_RO_data, 0
@ CHECK: .eabi_attribute 16, 0
@ CHECK-OBJ:        Tag: 16
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: ABI_PCS_RO_data
@ CHECK-OBJ-NEXT:   Description: Absolute
	.eabi_attribute Tag_ABI_PCS_GOT_use, 0
@ CHECK: .eabi_attribute 17, 0
@ CHECK-OBJ:        Tag: 17
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: ABI_PCS_GOT_use
@ CHECK-OBJ-NEXT:   Description: Not Permitted
	.eabi_attribute Tag_ABI_PCS_wchar_t, 4
@ CHECK: .eabi_attribute 18, 4
@ CHECK-OBJ:        Tag: 18
@ CHECK-OBJ-NEXT:   Value: 4
@ CHECK-OBJ-NEXT:   TagName: ABI_PCS_wchar_t
@ CHECK-OBJ-NEXT:   Description: 4-byte
	.eabi_attribute Tag_ABI_FP_rounding, 1
@ CHECK: .eabi_attribute 19, 1
@ CHECK-OBJ:        Tag: 19
@ CHECK-OBJ-NEXT:   Value: 1
@ CHECK-OBJ-NEXT:   TagName: ABI_FP_rounding
@ CHECK-OBJ-NEXT:   Description: Runtime
	.eabi_attribute Tag_ABI_FP_denormal, 2
@ CHECK: .eabi_attribute 20, 2
@ CHECK-OBJ:        Tag: 20
@ CHECK-OBJ-NEXT:   Value: 2
@ CHECK-OBJ-NEXT:   TagName: ABI_FP_denormal
@ CHECK-OBJ-NEXT:   Description: Sign Only
	.eabi_attribute Tag_ABI_FP_exceptions, 1
@ CHECK: .eabi_attribute 21, 1
@ CHECK-OBJ:        Tag: 21
@ CHECK-OBJ-NEXT:   Value: 1
@ CHECK-OBJ-NEXT:   TagName: ABI_FP_exceptions
@ CHECK-OBJ-NEXT:   Description: IEEE-754
	.eabi_attribute Tag_ABI_FP_user_exceptions, 1
@ CHECK: .eabi_attribute 22, 1
@ CHECK-OBJ:        Tag: 22
@ CHECK-OBJ-NEXT:   Value: 1
@ CHECK-OBJ-NEXT:   TagName: ABI_FP_user_exceptions
@ CHECK-OBJ-NEXT:   Description: IEEE-754
	.eabi_attribute Tag_ABI_FP_number_model, 3
@ CHECK: .eabi_attribute 23, 3
@ CHECK-OBJ:        Tag: 23
@ CHECK-OBJ-NEXT:   Value: 3
@ CHECK-OBJ-NEXT:   TagName: ABI_FP_number_model
@ CHECK-OBJ-NEXT:   Description: IEEE-754
	.eabi_attribute Tag_ABI_align_needed, 1
@ CHECK: .eabi_attribute 24, 1
@ CHECK-OBJ:        Tag: 24
@ CHECK-OBJ-NEXT:   Value: 1
@ CHECK-OBJ-NEXT:   TagName: ABI_align_needed
@ CHECK-OBJ-NEXT:   Description: 8-byte alignment
	.eabi_attribute Tag_ABI_align_preserved, 2
@ CHECK: .eabi_attribute 25, 2
@ CHECK-OBJ:        Tag: 25
@ CHECK-OBJ-NEXT:   Value: 2
@ CHECK-OBJ-NEXT:   TagName: ABI_align_preserved
@ CHECK-OBJ-NEXT:   Description: 8-byte data and code alignment
	.eabi_attribute Tag_ABI_enum_size, 3
@ CHECK: .eabi_attribute 26, 3
@ CHECK-OBJ:        Tag: 26
@ CHECK-OBJ-NEXT:   Value: 3
@ CHECK-OBJ-NEXT:   TagName: ABI_enum_size
@ CHECK-OBJ-NEXT:   Description: External Int32
	.eabi_attribute Tag_ABI_HardFP_use, 0
@ CHECK: .eabi_attribute 27, 0
@ CHECK-OBJ:        Tag: 27
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: ABI_HardFP_use
@ CHECK-OBJ-NEXT:   Description: Tag_FP_arch
	.eabi_attribute Tag_ABI_VFP_args, 1
@ CHECK: .eabi_attribute 28, 1
@ CHECK-OBJ:        Tag: 28
@ CHECK-OBJ-NEXT:   Value: 1
@ CHECK-OBJ-NEXT:   TagName: ABI_VFP_args
@ CHECK-OBJ-NEXT:   Description: AAPCS VFP
	.eabi_attribute Tag_ABI_WMMX_args, 0
@ CHECK: .eabi_attribute 29, 0
@ CHECK-OBJ:        Tag: 29
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: ABI_WMMX_args
@ CHECK-OBJ-NEXT:   Description: AAPCS
	.eabi_attribute Tag_ABI_FP_optimization_goals, 1
@ CHECK: .eabi_attribute 31, 1
@ CHECK-OBJ:        Tag: 31
@ CHECK-OBJ-NEXT:   Value: 1
@ CHECK-OBJ-NEXT:   TagName: ABI_FP_optimization_goals
@ CHECK-OBJ-NEXT:   Description: Speed
	.eabi_attribute Tag_compatibility, 1, "aeabi"
@ CHECK: .eabi_attribute 32, 1, "aeabi"
@ CHECK-OBJ:        Tag: 32
@ CHECK-OBJ-NEXT:   Value: 1, aeabi
@ CHECK-OBJ-NEXT:   TagName: compatibility
@ CHECK-OBJ-NEXT:   Description: AEABI Conformant
	.eabi_attribute Tag_CPU_unaligned_access, 0
@ CHECK: .eabi_attribute 34, 0
@ CHECK-OBJ:        Tag: 34
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: CPU_unaligned_access
@ CHECK-OBJ-NEXT:   Description: Not Permitted
	.eabi_attribute Tag_FP_HP_extension, 0
@ CHECK: .eabi_attribute 36, 0
@ CHECK-OBJ:        Tag: 36
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: FP_HP_extension
@ CHECK-OBJ-NEXT:   Description: If Available
	.eabi_attribute Tag_ABI_FP_16bit_format, 0
@ CHECK: .eabi_attribute 38, 0
@ CHECK-OBJ:        Tag: 38
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: ABI_FP_16bit_format
@ CHECK-OBJ-NEXT:   Description: Not Permitte
	.eabi_attribute Tag_MPextension_use, 0
@ CHECK: .eabi_attribute 42, 0
@ CHECK-OBJ:        Tag: 42
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: MPextension_use
@ CHECK-OBJ-NEXT:   Description: Not Permitted
	.eabi_attribute Tag_DIV_use, 0
@ CHECK: .eabi_attribute 44, 0
@ CHECK-OBJ:        Tag: 44
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: DIV_use
@ CHECK-OBJ-NEXT:   Description: If Available
	.eabi_attribute Tag_DSP_extension, 0
@ CHECK: .eabi_attribute 46, 0
@ CHECK-OBJ:        Tag: 46
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: DSP_extension
@ CHECK-OBJ-NEXT:   Description: Not Permitted
  .eabi_attribute Tag_PAC_extension, 0
@ CHECK: .eabi_attribute 50, 0
@ CHECK-OBJ:        Tag: 50
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: PAC_extension
@ CHECK-OBJ-NEXT:   Description: Not Permitted
  .eabi_attribute Tag_BTI_extension, 0
@ CHECK: .eabi_attribute 52, 0
@ CHECK-OBJ:        Tag: 52
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: BTI_extension
@ CHECK-OBJ-NEXT:   Description: Not Permitted
	.eabi_attribute Tag_nodefaults, 0
@ CHECK: .eabi_attribute 64, 0
@ CHECK-OBJ:        Tag: 64
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: nodefaults
@ CHECK-OBJ-NEXT:   Description: Unspecified Tags UNDEFINED
	.eabi_attribute Tag_also_compatible_with, "\006\017"
@ CHECK: .eabi_attribute 65, "\006\017"
@ CHECK-OBJ:        Tag: 65
@ CHECK-OBJ-NEXT:   TagName: also_compatible_with
@ CHECK-OBJ-NEXT:   Value: \006\017
@ CHECK-OBJ-NEXT:   Description: Tag_CPU_arch = 15 (ARM v8-R)
	.eabi_attribute Tag_T2EE_use, 0
@ CHECK: .eabi_attribute 66, 0
@ CHECK-OBJ:        Tag: 66
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: T2EE_use
@ CHECK-OBJ-NEXT:   Description: Not Permitted
	.eabi_attribute Tag_Virtualization_use, 0
@ CHECK: .eabi_attribute 68, 0
@ CHECK-OBJ:        Tag: 68
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: Virtualization_use
@ CHECK-OBJ-NEXT:   Description: Not Permitted
  .eabi_attribute Tag_BTI_use, 0
@ CHECK: .eabi_attribute 74, 0
@ CHECK-OBJ:        Tag: 74
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: BTI_use
@ CHECK-OBJ-NEXT:   Description: Not Used
  .eabi_attribute Tag_PACRET_use, 0
@ CHECK: .eabi_attribute 76, 0
@ CHECK-OBJ:        Tag: 76
@ CHECK-OBJ-NEXT:   Value: 0
@ CHECK-OBJ-NEXT:   TagName: PACRET_use
@ CHECK-OBJ-NEXT:   Description: Not Used


@ ===--- Compatibility Checks ---===

	.eabi_attribute Tag_ABI_align8_needed, 1
@ CHECK: .eabi_attribute 24, 1
	.eabi_attribute Tag_ABI_align8_preserved, 2
@ CHECK: .eabi_attribute 25, 2

@ ===--- GNU AS Compatibility Checks ---===

	.eabi_attribute 2 * 2 + 1, "cortex-a9"
@ CHECK: .cpu cortex-a9
	.eabi_attribute 2 * 2 + 2, 5 * 2
@ CHECK: .eabi_attribute 6, 10
