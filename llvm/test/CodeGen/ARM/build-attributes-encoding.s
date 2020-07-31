// This tests that ARM attributes are properly encoded.

// RUN: llvm-mc < %s -triple=arm-linux-gnueabi -filetype=obj -o - \
// RUN:   | llvm-readobj -S --sd - | FileCheck %s

// Tag_CPU_name (=5)
.cpu cortex-a8

// Tag_CPU_arch (=6)
.eabi_attribute 6, 10

// Tag_arch_profile (=7)
.eabi_attribute 7, 'A'

// Tag_ARM_ISA_use (=8)
.eabi_attribute 8, 1

// Tag_THUMB_ISA_use (=9)
.eabi_attribute 9, 2

// Tag_FP_arch (=10)
.fpu vfpv3

// Tag_Advanced_SIMD_arch (=12)
.eabi_attribute 12, 2

// Tag_ABI_FP_denormal (=20)
.eabi_attribute 20, 1

// Tag_ABI_FP_exceptions (=21)
.eabi_attribute 21, 1

// Tag_ABI_FP_number_model (=23)
.eabi_attribute 23, 1

// Tag_ABI_align_needed (=24)
.eabi_attribute 24, 1

// Tag_ABI_align_preserved (=25)
.eabi_attribute 25, 1

// Tag_ABI_HardFP_use (=27)
.eabi_attribute 27, 0

// Tag_ABI_VFP_args (=28)
.eabi_attribute 28, 1

// Tag_FP_HP_extension (=36)
.eabi_attribute 36, 1

// Tag_MPextension_use (=42)
.eabi_attribute 42, 1

// Tag_DIV_use (=44)
.eabi_attribute 44, 2

// Tag_DSP_extension (=46)
.eabi_attribute 46, 1

// Tag_Virtualization_use (=68)
.eabi_attribute 68, 3

// Check that values > 128 are encoded properly
.eabi_attribute 110, 160

// Check that tags > 128 are encoded properly
.eabi_attribute 129, "1"
.eabi_attribute 250, 1

// CHECK:        Section {
// CHECK:          Name: .ARM.attributes
// CHECK-NEXT:     Type: SHT_ARM_ATTRIBUTES
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x34
// CHECK-NEXT:     Size: 73
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 41480000 00616561 62690001 3E000000
// CHECK-NEXT:       0010: 05636F72 7465782D 61380006 0A074108
// CHECK-NEXT:       0020: 0109020A 030C0214 01150117 01180119
// CHECK-NEXT:       0030: 011B001C 0124012A 012C022E 0144036E
// CHECK-NEXT:       0040: A0018101 3100FA01 01
// CHECK-NEXT:     )
