; RUN: llc -mtriple=mips < %s | FileCheck --check-prefixes=ALL,O32 %s
; RUN: llc -mtriple=mipsel < %s | FileCheck --check-prefixes=ALL,O32 %s

; RUN-TODO: llc -mtriple=mips64 -target-abi o32 < %s | FileCheck --check-prefixes=ALL,O32 %s
; RUN-TODO: llc -mtriple=mips64el -target-abi o32 < %s | FileCheck --check-prefixes=ALL,O32 %s

; RUN: llc -mtriple=mips64 -target-abi n32 < %s | FileCheck --check-prefixes=ALL,N32 %s
; RUN: llc -mtriple=mips64el -target-abi n32 < %s | FileCheck --check-prefixes=ALL,N32 %s

; RUN: llc -mtriple=mips64 -target-abi n64 < %s | FileCheck --check-prefixes=ALL,N64 %s
; RUN: llc -mtriple=mips64el -target-abi n64 < %s | FileCheck --check-prefixes=ALL,N64 %s

; Test the stack alignment for all ABI's and byte orders as specified by
; section 5 of MD00305 (MIPS ABIs Described).

define void @local_bytes_1() nounwind {
entry:
        %0 = alloca i8
        ret void
}

; ALL-LABEL: local_bytes_1:
; O32:           addiu $sp, $sp, -8
; O32:           addiu $sp, $sp, 8
; N32:           addiu $sp, $sp, -16
; N32:           addiu $sp, $sp, 16
; N64:           addiu $sp, $sp, -16
; N64:           addiu $sp, $sp, 16
