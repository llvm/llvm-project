; Check that MachO ARM CPU Subtypes are respected

; RUN: llc -mtriple=armv4t-apple-darwin -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V4T

; RUN: llc -mtriple=armv5-apple-darwin -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V5
; RUN: llc -mtriple=armv5e-apple-darwin -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V5
; RUN: llc -mtriple=armv5t-apple-darwin -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V5
; RUN: llc -mtriple=armv5te-apple-darwin -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V5
; RUN: llc -mtriple=armv5tej-apple-darwin -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V5

; RUN: llc -mtriple=armv6-apple-darwin -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V6
; RUN: llc -mtriple=armv6k-apple-darwin -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V6
; RUN: llc -mtriple=thumbv6-apple-darwin -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V6
; RUN: llc -mtriple=thumbv6k-apple-darwin -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V6

; RUN: llc -mtriple=armv6m-apple-darwin -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V6M
; RUN: llc -mtriple=thumbv6m-apple-darwin -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V6M

; RUN: llc -mtriple=armv7-apple-darwin -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V7
; RUN: llc -mtriple=thumbv7-apple-darwin -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V7

; RUN: llc -mtriple=thumbv7em-apple-darwin -mcpu=cortex-m4 -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V7EM
; RUN: llc -mtriple=thumbv7em-apple-darwin -mcpu=cortex-m7 -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V7EM

; RUN: llc -mtriple=armv7k-apple-darwin -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V7K
; RUN: llc -mtriple=thumbv7k-apple-darwin -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V7K

; RUN: llc -mtriple=thumbv7m-apple-darwin -mcpu=sc300 -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V7M
; RUN: llc -mtriple=thumbv7m-apple-darwin -mcpu=cortex-m3 -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V7M

; RUN: llc -mtriple=armv7s-apple-darwin -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V7S
; RUN: llc -mtriple=thumbv7s-apple-darwin -filetype=obj -o - < %s \
; RUN: | llvm-readobj --file-headers - | FileCheck %s --check-prefix=CHECK-V7S

define void @_test() {
  ret void
}

; CHECK-V4T:   CpuSubType: CPU_SUBTYPE_ARM_V4T (0x5)
; CHECK-V5:   CpuSubType: CPU_SUBTYPE_ARM_V5 (0x7)
; CHECK-V6:   CpuSubType: CPU_SUBTYPE_ARM_V6 (0x6)
; CHECK-V6M:   CpuSubType: CPU_SUBTYPE_ARM_V6M (0xE)
; CHECK-V7:   CpuSubType: CPU_SUBTYPE_ARM_V7 (0x9)
; CHECK-V7EM:   CpuSubType: CPU_SUBTYPE_ARM_V7EM (0x10)
; CHECK-V7K:   CpuSubType: CPU_SUBTYPE_ARM_V7K (0xC)
; CHECK-V7M:   CpuSubType: CPU_SUBTYPE_ARM_V7M (0xF)
; CHECK-V7S:   CpuSubType: CPU_SUBTYPE_ARM_V7S (0xB)
