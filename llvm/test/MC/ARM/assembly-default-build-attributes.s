// RUN: llvm-mc -triple armv7a < %s -arm-add-build-attributes | FileCheck %s --check-prefix=v7A
// RUN: llvm-mc -triple armv6m < %s -arm-add-build-attributes | FileCheck %s --check-prefix=v6M
// RUN: llvm-mc -triple armv7m < %s -arm-add-build-attributes | FileCheck %s --check-prefix=v7M
// RUN: llvm-mc -triple armv7a -mcpu=cortex-a15 < %s -arm-add-build-attributes | FileCheck %s --check-prefix=Cortex-A15
// RUN: llvm-mc -triple armv8m.base < %s -arm-add-build-attributes | FileCheck %s --check-prefix=v8M_Baseline
// RUN: llvm-mc -triple armv8m.main < %s -arm-add-build-attributes | FileCheck %s --check-prefix=v8M_Mainline

// This isn't intended to be a through check of the build attributes emitted
// for each target (that's tested elsewhere), but just to check that the
// hardware attributes are emitted by the assembler based on the selected
// target when requested.

// v7A-NOT: .cpu
// v7A: .eabi_attribute 6, 10   @ Tag_CPU_arch
// v7A: .eabi_attribute 7, 65   @ Tag_CPU_arch_profile
// v7A: .eabi_attribute 8, 1    @ Tag_ARM_ISA_use
// v7A: .eabi_attribute 9, 2    @ Tag_THUMB_ISA_use
// v7A: .fpu    neon
// v7A: .eabi_attribute 34, 1   @ Tag_CPU_unaligned_access

// v6M-NOT: .cpu
// v6M: .eabi_attribute 6, 12   @ Tag_CPU_arch
// v6M: .eabi_attribute 7, 77   @ Tag_CPU_arch_profile
// v6M: .eabi_attribute 8, 0    @ Tag_ARM_ISA_use
// v6M: .eabi_attribute 9, 1    @ Tag_THUMB_ISA_use
// v6M: .eabi_attribute 34, 0   @ Tag_CPU_unaligned_access

// v7M-NOT: .cpu
// v7M: .eabi_attribute 6, 10   @ Tag_CPU_arch
// v7M: .eabi_attribute 7, 77   @ Tag_CPU_arch_profile
// v7M: .eabi_attribute 8, 0    @ Tag_ARM_ISA_use
// v7M: .eabi_attribute 9, 2    @ Tag_THUMB_ISA_use
// v7M: .eabi_attribute 34, 1   @ Tag_CPU_unaligned_access

// v8M_Baseline-NOT: .cpu
// v8M_Baseline: .eabi_attribute 6, 16   @ Tag_CPU_arch
// v8M_Baseline: .eabi_attribute 7, 77   @ Tag_CPU_arch_profile
// v8M_Baseline: .eabi_attribute 8, 0    @ Tag_ARM_ISA_use
// v8M_Baseline: .eabi_attribute 9, 3    @ Tag_THUMB_ISA_use
// v8M_Baseline: .eabi_attribute 34, 0   @ Tag_CPU_unaligned_access

// v8M_Mainline-NOT: .cpu
// v8M_Mainline: .eabi_attribute 6, 17   @ Tag_CPU_arch
// v8M_Mainline: .eabi_attribute 7, 77   @ Tag_CPU_arch_profile
// v8M_Mainline: .eabi_attribute 8, 0    @ Tag_ARM_ISA_use
// v8M_Mainline: .eabi_attribute 9, 3    @ Tag_THUMB_ISA_use
// v8M_Mainline: .eabi_attribute 34, 1   @ Tag_CPU_unaligned_access

// Cortex-A15: .cpu    cortex-a15
// Cortex-A15: .eabi_attribute 6, 10   @ Tag_CPU_arch
// Cortex-A15: .eabi_attribute 7, 65   @ Tag_CPU_arch_profile
// Cortex-A15: .eabi_attribute 8, 1    @ Tag_ARM_ISA_use
// Cortex-A15: .eabi_attribute 9, 2    @ Tag_THUMB_ISA_use
// Cortex-A15: .fpu    neon-vfpv4
// Cortex-A15: .eabi_attribute 36, 1   @ Tag_FP_HP_extension
// Cortex-A15: .eabi_attribute 42, 1   @ Tag_MPextension_use
// Cortex-A15: .eabi_attribute 44, 2   @ Tag_DIV_use
// Cortex-A15: .eabi_attribute 34, 1   @ Tag_CPU_unaligned_access
// Cortex-A15: .eabi_attribute 68, 3   @ Tag_Virtualization_use
