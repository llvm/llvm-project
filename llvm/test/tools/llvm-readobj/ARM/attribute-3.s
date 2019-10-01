@ RUN: llvm-mc -triple armv7-elf -filetype asm -o - %s | FileCheck %s
@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s \
@ RUN:   | llvm-readobj --arm-attributes - | FileCheck %s --check-prefix=CHECK-OBJ
.eabi_attribute  Tag_CPU_arch, 3
@CHECK:   .eabi_attribute 6, 3
@CHECK-OBJ: Tag: 6
@CHECK-OBJ-NEXT: Value: 3
@CHECK-OBJ-NEXT: TagName: CPU_arch
@CHECK-OBJ-NEXT: Description: ARM v5T

.eabi_attribute  Tag_FP_arch, 3
@CHECK:   .eabi_attribute 10, 3
@CHECK-OBJ: Tag: 10
@CHECK-OBJ-NEXT: Value: 3
@CHECK-OBJ-NEXT: TagName: FP_arch
@CHECK-OBJ-NEXT: Description: VFPv3

.eabi_attribute  Tag_Advanced_SIMD_arch, 3
@CHECK:   .eabi_attribute 12, 3
@CHECK-OBJ: Tag: 12
@CHECK-OBJ-NEXT: Value: 3
@CHECK-OBJ-NEXT: TagName: Advanced_SIMD_arch
@CHECK-OBJ-NEXT: Description: ARMv8-a NEON

.eabi_attribute  Tag_PCS_config, 3
@CHECK:   .eabi_attribute 13, 3
@CHECK-OBJ: Tag: 13
@CHECK-OBJ-NEXT: Value: 3
@CHECK-OBJ-NEXT: TagName: PCS_config
@CHECK-OBJ-NEXT: Description: Linux DSO

.eabi_attribute  Tag_ABI_PCS_R9_use, 3
@CHECK:   .eabi_attribute 14, 3
@CHECK-OBJ: Tag: 14
@CHECK-OBJ-NEXT: Value: 3
@CHECK-OBJ-NEXT: TagName: ABI_PCS_R9_use
@CHECK-OBJ-NEXT: Description: Unused

.eabi_attribute  Tag_ABI_PCS_RW_data, 3
@CHECK:   .eabi_attribute 15, 3
@CHECK-OBJ: Tag: 15
@CHECK-OBJ-NEXT: Value: 3
@CHECK-OBJ-NEXT: TagName: ABI_PCS_RW_data
@CHECK-OBJ-NEXT: Description: Not Permitted

.eabi_attribute  Tag_ABI_FP_number_model, 3
@CHECK:   .eabi_attribute 23, 3
@CHECK-OBJ: Tag: 23
@CHECK-OBJ-NEXT: Value: 3
@CHECK-OBJ-NEXT: TagName: ABI_FP_number_model
@CHECK-OBJ-NEXT: Description: IEEE-754

.eabi_attribute  Tag_ABI_align_needed, 3
@CHECK:   .eabi_attribute 24, 3
@CHECK-OBJ: Tag: 24
@CHECK-OBJ-NEXT: Value: 3
@CHECK-OBJ-NEXT: TagName: ABI_align_needed
@CHECK-OBJ-NEXT: Description: Reserved

.eabi_attribute  Tag_ABI_align_preserved, 3
@CHECK:   .eabi_attribute 25, 3
@CHECK-OBJ: Tag: 25
@CHECK-OBJ-NEXT: Value: 3
@CHECK-OBJ-NEXT: TagName: ABI_align_preserved
@CHECK-OBJ-NEXT: Description: Reserved

.eabi_attribute  Tag_ABI_enum_size, 3
@CHECK:   .eabi_attribute 26, 3
@CHECK-OBJ: Tag: 26
@CHECK-OBJ-NEXT: Value: 3
@CHECK-OBJ-NEXT: TagName: ABI_enum_size
@CHECK-OBJ-NEXT: Description: External Int32

.eabi_attribute  Tag_ABI_HardFP_use, 3
@CHECK:   .eabi_attribute 27, 3
@CHECK-OBJ: Tag: 27
@CHECK-OBJ-NEXT: Value: 3
@CHECK-OBJ-NEXT: TagName: ABI_HardFP_use
@CHECK-OBJ-NEXT: Description: Tag_FP_arch (deprecated)

.eabi_attribute  Tag_ABI_VFP_args, 3
@CHECK:   .eabi_attribute 28, 3
@CHECK-OBJ: Tag: 28
@CHECK-OBJ-NEXT: Value: 3
@CHECK-OBJ-NEXT: TagName: ABI_VFP_args
@CHECK-OBJ-NEXT: Description: Not Permitted

.eabi_attribute  Tag_ABI_optimization_goals, 3
@CHECK:   .eabi_attribute 30, 3
@CHECK-OBJ: Tag: 30
@CHECK-OBJ-NEXT: Value: 3
@CHECK-OBJ-NEXT: TagName: ABI_optimization_goals
@CHECK-OBJ-NEXT: Description: Size

.eabi_attribute  Tag_ABI_FP_optimization_goals, 3
@CHECK:   .eabi_attribute 31, 3
@CHECK-OBJ: Tag: 31
@CHECK-OBJ-NEXT: Value: 3
@CHECK-OBJ-NEXT: TagName: ABI_FP_optimization_goals
@CHECK-OBJ-NEXT: Description: Size

.eabi_attribute  Tag_Virtualization_use, 3
@CHECK:   .eabi_attribute 68, 3
@CHECK-OBJ: Tag: 68
@CHECK-OBJ-NEXT: Value: 3
@CHECK-OBJ-NEXT: TagName: Virtualization_use
@CHECK-OBJ-NEXT: Description: TrustZone + Virtualization Extensions

