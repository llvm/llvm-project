; Check the emission of directives at the start of an asm file.

; ### O32 ABI ###
; RUN: llc -filetype=asm -mtriple mips-unknown-linux -mcpu=mips32 \
; RUN: -relocation-model=static %s -o - | \
; RUN:   FileCheck -DNAN=legacy -DABI=32 -check-prefixes=CHECK,STATIC-O32 %s

; RUN: llc -filetype=asm -mtriple mips-unknown-linux -mcpu=mips32 \
; RUN: -relocation-model=pic %s -o - | \
; RUN:   FileCheck -DNAN=legacy -DABI=32 -check-prefixes=CHECK,PIC-O32 %s

; RUN: llc -filetype=asm -mtriple mips-unknown-linux -mcpu=mips32 \
; RUN: -relocation-model=static -mattr=+nan2008 %s -o - | \
; RUN:   FileCheck -DNAN=2008 -DABI=32 -check-prefixes=CHECK,STATIC-O32 %s

; RUN: llc -filetype=asm -mtriple mips-unknown-linux -mcpu=mips32 \
; RUN: -relocation-model=pic -mattr=+nan2008 %s -o - | \
; RUN:   FileCheck -DNAN=2008 -DABI=32 -check-prefixes=CHECK,PIC-O32 %s

; ### N32 ABI ###
; RUN: llc -filetype=asm -mtriple mips64-unknown-linux -mcpu=mips64 \
; RUN: -relocation-model=static -target-abi n32 %s -o - | \
; RUN:   FileCheck -DNAN=legacy -DABI=N32 -check-prefixes=CHECK,STATIC-N32 %s

; RUN: llc -filetype=asm -mtriple mips64-unknown-linux -mcpu=mips64 \
; RUN: -relocation-model=pic -target-abi n32 %s -o - | \
; RUN:   FileCheck -DNAN=legacy -DABI=N32 -check-prefixes=CHECK,PIC-N32 %s

; RUN: llc -filetype=asm -mtriple mips64-unknown-linux -mcpu=mips64 \
; RUN: -relocation-model=static -target-abi n32 -mattr=+nan2008 %s -o - | \
; RUN:   FileCheck -DNAN=2008 -DABI=N32 -check-prefixes=CHECK,STATIC-N32 %s

; RUN: llc -filetype=asm -mtriple mips64-unknown-linux -mcpu=mips64 \
; RUN: -relocation-model=pic -target-abi n32 -mattr=+nan2008 %s -o - | \
; RUN:   FileCheck -DNAN=2008 -DABI=N32 -check-prefixes=CHECK,PIC-N32 %s

; ### N64 ABI ###
; RUN: llc -filetype=asm -mtriple mips64-unknown-linux -mcpu=mips64 \
; RUN: -relocation-model=static -target-abi n64 %s -o - | \
; RUN:   FileCheck -DNAN=legacy -DABI=64 -check-prefixes=CHECK,STATIC-N64 %s

; RUN: llc -filetype=asm -mtriple mips64-unknown-linux -mcpu=mips64 \
; RUN: -relocation-model=pic -target-abi n64 %s -o - | \
; RUN:   FileCheck -DNAN=legacy -DABI=64 -check-prefixes=CHECK,PIC-N64 %s

; RUN: llc -filetype=asm -mtriple mips64-unknown-linux -mcpu=mips64 \
; RUN: -relocation-model=static -target-abi n64 -mattr=+nan2008 %s -o - | \
; RUN:   FileCheck -DNAN=2008 -DABI=64 -check-prefixes=CHECK,STATIC-N64 %s

; RUN: llc -filetype=asm -mtriple mips64-unknown-linux -mcpu=mips64 \
; RUN: -relocation-model=pic -target-abi n64 -mattr=+nan2008 %s -o - | \
; RUN:   FileCheck -DNAN=2008 -DABI=64 -check-prefixes=CHECK,PIC-N64 %s

; STATIC-O32: .abicalls
; STATIC-O32: .option pic0

; PIC-O32: .abicalls
; PIC-O32-NOT: .option pic0

; STATIC-N32: .abicalls
; STATIC-N32: .option pic0

; PIC-N32: .abicalls
; PIC-N32-NOT: .option pic0

; STATIC-N64-NOT: .abicalls
; STATIC-N64-NOT: .option pic0

; PIC-N64: .abicalls
; PIC-N64-NOT: .option pic0

; CHECK: .section .mdebug.abi[[ABI]]
; CHECK: .nan [[NAN]]
; CHECK: .text
; CHECK: .file
