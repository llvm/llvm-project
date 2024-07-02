; RUN: llc --mtriple=loongarch32 -mattr=+d --filetype=obj %s -o %t-la32
; RUN: llvm-readelf -h %t-la32 | FileCheck %s --check-prefixes=ILP32,ABI-D --match-full-lines

; RUN: llc --mtriple=loongarch32 --filetype=obj %s --target-abi=ilp32s -o %t-ilp32s
; RUN: llvm-readelf -h %t-ilp32s | FileCheck %s --check-prefixes=ILP32,ABI-S --match-full-lines

; RUN: llc --mtriple=loongarch32 -mattr=+f --filetype=obj %s --target-abi=ilp32f -o %t-ilp32f
; RUN: llvm-readelf -h %t-ilp32f | FileCheck %s --check-prefixes=ILP32,ABI-F --match-full-lines

; RUN: llc --mtriple=loongarch32 -mattr=+d --filetype=obj %s --target-abi=ilp32d -o %t-ilp32d
; RUN: llvm-readelf -h %t-ilp32d | FileCheck %s --check-prefixes=ILP32,ABI-D --match-full-lines

; RUN: llc --mtriple=loongarch64 -mattr=+d --filetype=obj %s -o %t-la64
; RUN: llvm-readelf -h %t-la64 | FileCheck %s --check-prefixes=LP64,ABI-D --match-full-lines

; RUN: llc --mtriple=loongarch64 --filetype=obj %s --target-abi=lp64s -o %t-lp64s
; RUN: llvm-readelf -h %t-lp64s | FileCheck %s --check-prefixes=LP64,ABI-S --match-full-lines

; RUN: llc --mtriple=loongarch64 -mattr=+f --filetype=obj %s --target-abi=lp64f -o %t-lp64f
; RUN: llvm-readelf -h %t-lp64f | FileCheck %s --check-prefixes=LP64,ABI-F --match-full-lines

; RUN: llc --mtriple=loongarch64 --filetype=obj %s --mattr=+d --target-abi=lp64d -o %t-lp64d
; RUN: llvm-readelf -h %t-lp64d | FileCheck %s --check-prefixes=LP64,ABI-D --match-full-lines

; LP64: Class: ELF64
; ILP32: Class: ELF32

; ABI-S: Flags: 0x41, SOFT-FLOAT, OBJ-v1
; ABI-F: Flags: 0x42, SINGLE-FLOAT, OBJ-v1
; ABI-D: Flags: 0x43, DOUBLE-FLOAT, OBJ-v1

define void @foo() {
  ret void
}
