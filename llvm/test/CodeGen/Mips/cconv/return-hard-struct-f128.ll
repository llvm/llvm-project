; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=static -target-abi n32 < %s | FileCheck --check-prefixes=ALL,N32 %s
; RUN: llc -mtriple=mips64el-linux-gnu -relocation-model=static -target-abi n32 < %s | FileCheck --check-prefixes=ALL,N32 %s

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=static -target-abi n64 < %s | FileCheck --check-prefixes=ALL,N64 %s
; RUN: llc -mtriple=mips64el-linux-gnu -relocation-model=static -target-abi n64 < %s | FileCheck --check-prefixes=ALL,N64 %s

; Test return of {fp128} agrees with de-facto N32/N64 ABI.

@struct_fp128 = global {fp128} zeroinitializer

define inreg {fp128} @ret_struct_fp128() nounwind {
entry:
        %0 = load volatile {fp128}, {fp128}* @struct_fp128
        ret {fp128} %0
}

; ALL-LABEL: ret_struct_fp128:

; O32 generates different IR so we don't test it here. It returns the struct
; indirectly.

; Contrary to the N32/N64 ABI documentation, a struct containing a long double
; is returned in $f0, and $f1 instead of the usual $f0, and $f2. This is to
; match the de facto ABI as implemented by GCC.
; N32-DAG:        lui [[R1:\$[0-9]+]], %hi(struct_fp128)
; N32-DAG:        ld  [[R2:\$[0-9]+]], %lo(struct_fp128)([[R1]])
; N32-DAG:        dmtc1 [[R2]], $f0
; N32-DAG:        addiu [[R3:\$[0-9]+]], [[R1]], %lo(struct_fp128)
; N32-DAG:        ld  [[R4:\$[0-9]+]], 8([[R3]])
; N32-DAG:        dmtc1 [[R4]], $f1

; N64-DAG:        lui  [[R1:\$[0-9]+]], %highest(struct_fp128)
; N64-DAG:        ld  [[R2:\$[0-9]+]], %lo(struct_fp128)([[R1]])
; N64-DAG:        dmtc1 [[R2]], $f0
; N64-DAG:        ld  [[R4:\$[0-9]+]], 8([[R1]])
; N64-DAG:        dmtc1 [[R4]], $f1
