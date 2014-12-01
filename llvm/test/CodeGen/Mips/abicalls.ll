; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux -mcpu=mips32 -relocation-model=static %s -o - | FileCheck -check-prefix=ABICALLS -check-prefix=STATIC %s
; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux -mcpu=mips32 %s -o - | FileCheck -check-prefix=ABICALLS -check-prefix=PIC %s
; RUN: llc -filetype=asm -mtriple mips64el-unknown-linux -mcpu=mips4 -relocation-model=static %s -o - | FileCheck -check-prefix=ABICALLS -check-prefix=PIC %s
; RUN: llc -filetype=asm -mtriple mips64el-unknown-linux -mcpu=mips64 -relocation-model=static %s -o - | FileCheck -check-prefix=ABICALLS -check-prefix=PIC %s

; RUN: llc -filetype=asm -mtriple mipsel-unknown-linux -mcpu=mips32 -mattr noabicalls -relocation-model=static %s -o - | FileCheck -implicit-check-not='.abicalls' -implicit-check-not='pic0' %s

; ABICALLS: .abicalls

; STATIC: pic0
; PIC-NOT: pic0
