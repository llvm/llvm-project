; RUN: llc -march=mips64 -relocation-model=static -target-abi n32 < %s | FileCheck --check-prefixes=ALL,N32 %s
; RUN: llc -march=mips64el -relocation-model=static -target-abi n32 < %s | FileCheck --check-prefixes=ALL,N32 %s

; RUN: llc -march=mips64 -relocation-model=static -target-abi n64 < %s | FileCheck --check-prefixes=ALL,N64 %s
; RUN: llc -march=mips64el -relocation-model=static -target-abi n64 < %s | FileCheck --check-prefixes=ALL,N64 %s

; COM: Test the fp128 returns for N32/N64 and all byte orders as specified by
; COM: section 5 of MD00305 (MIPS ABIs Described).
;
; O32 is not tested because long double is the same as double on O32.
;
@fp128 = global fp128 zeroinitializer

define fp128 @retldouble() nounwind {
entry:
        %0 = load volatile fp128, fp128* @fp128
        ret fp128 %0
}

; ALL-LABEL: retldouble:
; N32-DAG:           ldc1 $f0, %lo(fp128)([[R1:\$[0-9]+]])
; N32-DAG:           addiu [[R3:\$[0-9]+]], [[R1]], %lo(fp128)
; N32-DAG:           ldc1 $f2, 8([[R3]])

; N64-DAG:           lui [[R2:\$[0-9]+]], %highest(fp128)
; N64-DAG:           ldc1 $f0, %lo(fp128)([[R2]])
; N64-DAG:           ldc1 $f2, 8([[R2]])
