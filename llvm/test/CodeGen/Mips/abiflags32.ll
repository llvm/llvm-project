; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux \
; RUN:     -mcpu=mips32 %s -o - | FileCheck %s
; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux \
; RUN:     -mcpu=mips32r2 -mattr=fp64 %s -o - | FileCheck -check-prefix=CHECK-64 %s
; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux \
; RUN:     -mcpu=mips64 -target-abi n32 %s -o - | FileCheck -check-prefix=CHECK-64n %s
; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux -mcpu=mips32 \
; RUN:     -mattr=soft-float %s -o - | FileCheck  -check-prefix=SOFT %s
; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux -mcpu=mips32r6 \
; RUN:     -mattr=soft-float %s -o - | FileCheck  -check-prefix=SOFT %s
; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux -mcpu=mips64 \
; RUN:     -mattr=soft-float -target-abi n64 %s -o - | FileCheck  -check-prefix=SOFT %s
; RUN: llc -mtriple=mipsel -mcpu=mips32r2 -target-abi=o32 -mattr=+fp64,+single-float < %s | FileCheck %s --check-prefix=SINGLE
; RUN: llc -mtriple=mipsel -mcpu=mips32r6 -target-abi=o32 -mattr=+fp64,+single-float < %s | FileCheck %s --check-prefix=SINGLE
; RUN: llc -mtriple=mipsel -mcpu=mips32r6 -target-abi=o32 -mattr=+fp64,+single-float -filetype=obj < %s | llvm-readobj -A - | FileCheck %s --check-prefix=SINGLE-OBJ
; RUN: llc -mtriple=mipsel -mcpu=mips32r6 -target-abi=o32 -mattr=+fp64,+single-float,+soft-float < %s | FileCheck %s --check-prefix=SOFT

; CHECK: .nan    legacy
; We don't emit '.module fp=32' for compatibility with binutils 2.24 which
; doesn't accept .module.
; CHECK-NOT: .module fp=32

; CHECK-64: .nan    legacy
; We do emit '.module fp=64' though since it contradicts the default value.
; CHECK-64: .module fp=64

; CHECK-64n: .nan    legacy
; We don't emit '.module fp=64' for compatibility with binutils 2.24 which
; doesn't accept .module.
; CHECK-64n-NOT: .module fp=64

; SOFT: .module softfloat

; Single-float is not an fp=32/64/xx ABI. Enabling fp64 must not try
; to print SINGLE as one of those .module fp values.
; SINGLE: .nan
; SINGLE-NOT: .module fp=
; SINGLE: .text
; SINGLE-OBJ: FP ABI: Hard float (single precision) (0x2)
; SINGLE-OBJ: CPR1 size: 32
