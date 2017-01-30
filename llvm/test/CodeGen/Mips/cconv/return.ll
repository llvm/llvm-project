; RUN: llc -mtriple=mips-linux-gnu -relocation-model=static < %s | FileCheck --check-prefixes=ALL,O32 %s
; RUN: llc -mtriple=mipsel-linux-gnu -relocation-model=static < %s | FileCheck --check-prefixes=ALL,O32 %s

; RUN-TODO: llc -mtriple=mips64-linux-gnu -relocation-model=static -target-abi o32 < %s | FileCheck --check-prefixes=ALL,O32 %s
; RUN-TODO: llc -mtriple=mips64el-linux-gnu -relocation-model=static -target-abi o32 < %s | FileCheck --check-prefixes=ALL,O32 %s

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=static -target-abi n32 < %s | FileCheck --check-prefixes=ALL,N32 %s
; RUN: llc -mtriple=mips64el-linux-gnu -relocation-model=static -target-abi n32 < %s | FileCheck --check-prefixes=ALL,N32 %s

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=static -target-abi n64 < %s | FileCheck --check-prefixes=ALL,N64 %s
; RUN: llc -mtriple=mips64el-linux-gnu -relocation-model=static -target-abi n64 < %s | FileCheck --check-prefixes=ALL,N64 %s

; Test the integer returns for all ABI's and byte orders as specified by
; section 5 of MD00305 (MIPS ABIs Described).

; We only test Linux because other OS's use different relocations and I don't
; know if this is correct.

@byte = global i8 zeroinitializer
@word = global i32 zeroinitializer
@dword = global i64 zeroinitializer
@float = global float zeroinitializer
@double = global double zeroinitializer

define i8 @reti8() nounwind {
entry:
        %0 = load volatile i8, i8* @byte
        ret i8 %0
}

; ALL-LABEL: reti8:
; O32-DAG:           lui [[R1:\$[0-9]+]], %hi(byte)
; O32-DAG:           lbu $2, %lo(byte)([[R1]])
; N32-DAG:           lui [[R1:\$[0-9]+]], %hi(byte)
; N32-DAG:           lbu $2, %lo(byte)([[R1]])
; N64-DAG:           lui  [[R1:\$[0-9]+]], %highest(byte)
; N64-DAG:           lbu $2, %lo(byte)([[R1]])

define i32 @reti32() nounwind {
entry:
        %0 = load volatile i32, i32* @word
        ret i32 %0
}

; ALL-LABEL: reti32:
; O32-DAG:           lui [[R1:\$[0-9]+]], %hi(word)
; O32-DAG:           lw $2, %lo(word)([[R1]])
; N32-DAG:           lui [[R1:\$[0-9]+]], %hi(word)
; N32-DAG:           lw $2, %lo(word)([[R1]])
; N64-DAG:           lui [[R1:\$[0-9]+]], %highest(word)
; N64-DAG:           lw $2, %lo(word)([[R1]])

define i64 @reti64() nounwind {
entry:
        %0 = load volatile i64, i64* @dword
        ret i64 %0
}

; ALL-LABEL: reti64:
; On O32, we must use v0 and v1 for the return value
; O32-DAG:           lw $2, %lo(dword)([[R1:\$[0-9]+]])
; O32-DAG:           addiu [[R2:\$[0-9]+]], [[R1]], %lo(dword)
; O32-DAG:           lw $3, 4([[R2]])
; N32-DAG:           ld $2, %lo(dword)([[R1:\$[0-9]+]])
; N64-DAG:           lui  [[R1:\$[0-9]+]], %highest(dword)
; N64-DAG:           ld $2, %lo(dword)([[R1]])
