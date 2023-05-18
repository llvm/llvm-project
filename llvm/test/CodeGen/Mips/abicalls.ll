; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux -mcpu=mips32 -relocation-model=static %s -o - | FileCheck -check-prefixes=ABICALLS,STATIC %s
; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux -mcpu=mips32 -relocation-model=pic %s -o - | FileCheck -check-prefixes=ABICALLS,PIC %s
; RUN: llc -filetype=asm -mtriple mips64el-unknown-linux -mcpu=mips4 -relocation-model=static %s -o - | FileCheck -check-prefixes=N64-STATIC %s
; RUN: llc -filetype=asm -mtriple mips64el-unknown-linux -mcpu=mips64 -relocation-model=static %s -o - | FileCheck -check-prefixes=N64-STATIC %s
; RUN: llc -filetype=asm -mtriple mips64el-unknown-linux -mcpu=mips4 -relocation-model=static -mattr=+sym32 %s -o - | FileCheck -check-prefixes=ABICALLS,STATIC %s
; RUN: llc -filetype=asm -mtriple mips64el-unknown-linux -mcpu=mips64 -relocation-model=static -mattr=+sym32 %s -o - | FileCheck -check-prefixes=ABICALLS,STATIC %s
; RUN: llc -filetype=asm -mtriple mips64el-unknown-linux -mcpu=mips4 -relocation-model=pic %s -o - | FileCheck -check-prefixes=ABICALLS %s
; RUN: llc -filetype=asm -mtriple mips64el-unknown-linux -mcpu=mips64 -relocation-model=pic %s -o - | FileCheck -check-prefixes=ABICALLS %s


; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux -mcpu=mips32 -mattr noabicalls -relocation-model=static %s -o - | FileCheck -implicit-check-not='.abicalls' -implicit-check-not='pic0' %s

; ABICALLS: .abicalls

; STATIC: pic0
; PIC-NOT: pic0

; N64-STATIC-NOT: .abicalls
; N64-STATIC-NOT: .pic0
