; Check the emission of directives at the start of an asm file.

; ### O32 ABI ###
; RUN: llc -filetype=asm -mtriple mips-unknown-linux -mcpu=mips32 \
; RUN: -relocation-model=static %s -o - | \
; RUN:   FileCheck -check-prefixes=CHECK-STATIC-O32,CHECK-STATIC-O32-NLEGACY %s

; RUN: llc -filetype=asm -mtriple mips-unknown-linux -mcpu=mips32 \
; RUN: -relocation-model=pic %s -o - | \
; RUN:   FileCheck -check-prefixes=CHECK-PIC-O32,CHECK-PIC-O32-NLEGACY %s

; RUN: llc -filetype=asm -mtriple mips-unknown-linux -mcpu=mips32 \
; RUN: -relocation-model=static -mattr=+nan2008 %s -o - | \
; RUN:   FileCheck -check-prefixes=CHECK-STATIC-O32,CHECK-STATIC-O32-N2008 %s

; RUN: llc -filetype=asm -mtriple mips-unknown-linux -mcpu=mips32 \
; RUN: -relocation-model=pic -mattr=+nan2008 %s -o - | \
; RUN:   FileCheck -check-prefixes=CHECK-PIC-O32,CHECK-PIC-O32-N2008 %s

; ### N32 ABI ###
; RUN: llc -filetype=asm -mtriple mips64-unknown-linux -mcpu=mips64 \
; RUN: -relocation-model=static -target-abi n32 %s -o - | \
; RUN:   FileCheck -check-prefixes=CHECK-STATIC-N32,CHECK-STATIC-N32-NLEGACY %s

; RUN: llc -filetype=asm -mtriple mips64-unknown-linux -mcpu=mips64 \
; RUN: -relocation-model=pic -target-abi n32 %s -o - | \
; RUN:   FileCheck -check-prefixes=CHECK-PIC-N32,CHECK-PIC-N32-NLEGACY %s

; RUN: llc -filetype=asm -mtriple mips64-unknown-linux -mcpu=mips64 \
; RUN: -relocation-model=static -target-abi n32 -mattr=+nan2008 %s -o - | \
; RUN:   FileCheck -check-prefixes=CHECK-STATIC-N32,CHECK-STATIC-N32-N2008 %s

; RUN: llc -filetype=asm -mtriple mips64-unknown-linux -mcpu=mips64 \
; RUN: -relocation-model=pic -target-abi n32 -mattr=+nan2008 %s -o - | \
; RUN:   FileCheck -check-prefixes=CHECK-PIC-N32,CHECK-PIC-N32-N2008 %s

; ### N64 ABI ###
; RUN: llc -filetype=asm -mtriple mips64-unknown-linux -mcpu=mips64 \
; RUN: -relocation-model=static -target-abi n64 %s -o - | \
; RUN:   FileCheck -check-prefixes=CHECK-STATIC-N64,CHECK-STATIC-N64-NLEGACY %s

; RUN: llc -filetype=asm -mtriple mips64-unknown-linux -mcpu=mips64 \
; RUN: -relocation-model=pic -target-abi n64 %s -o - | \
; RUN:   FileCheck -check-prefixes=CHECK-PIC-N64,CHECK-PIC-N64-NLEGACY %s

; RUN: llc -filetype=asm -mtriple mips64-unknown-linux -mcpu=mips64 \
; RUN: -relocation-model=static -target-abi n64 -mattr=+nan2008 %s -o - | \
; RUN:   FileCheck -check-prefixes=CHECK-STATIC-N64,CHECK-STATIC-N64-N2008 %s

; RUN: llc -filetype=asm -mtriple mips64-unknown-linux -mcpu=mips64 \
; RUN: -relocation-model=pic -target-abi n64 -mattr=+nan2008 %s -o - | \
; RUN:   FileCheck -check-prefixes=CHECK-PIC-N64,CHECK-PIC-N64-N2008 %s

; CHECK-STATIC-O32: .abicalls
; CHECK-STATIC-O32: .option pic0
; CHECK-STATIC-O32: .section .mdebug.abi32
; CHECK-STATIC-O32-NLEGACY: .nan legacy
; CHECK-STATIC-O32-N2008: .nan 2008

; CHECK-PIC-O32: .abicalls
; CHECK-PIC-O32-NOT: .option pic0
; CHECK-PIC-O32: .section .mdebug.abi32
; CHECK-PIC-O32-NLEGACY: .nan legacy
; CHECK-PIC-O32-N2008: .nan 2008

; CHECK-STATIC-N32: .abicalls
; CHECK-STATIC-N32: .option pic0
; CHECK-STATIC-N32: .section .mdebug.abiN32
; CHECK-STATIC-N32-NLEGACY: .nan legacy
; CHECK-STATIC-N32-N2008: .nan 2008

; CHECK-PIC-N32: .abicalls
; CHECK-PIC-N32-NOT: .option pic0
; CHECK-PIC-N32: .section .mdebug.abiN32
; CHECK-PIC-N32-NLEGACY: .nan legacy
; CHECK-PIC-N32-N2008: .nan 2008

; CHECK-STATIC-N64: .abicalls
; CHECK-STATIC-N64-NOT: .option pic0
; CHECK-STATIC-N64: .section .mdebug.abi64
; CHECK-STATIC-N64-NLEGACY: .nan legacy
; CHECK-STATIC-N64-N2008: .nan 2008

; CHECK-PIC-N64: .abicalls
; CHECK-PIC-N64-NOT: .option pic0
; CHECK-PIC-N64: .section .mdebug.abi64
; CHECK-PIC-N64-NLEGACY: .nan legacy
; CHECK-PIC-N64-N2008: .nan 2008
