; RUN: llc -mtriple=mips64 -relocation-model=static -target-abi n32 -mattr=single-float < %s \
; RUN:   | FileCheck --check-prefixes=ALL,SYM32 %s
; RUN: llc -mtriple=mips64el -relocation-model=static -target-abi n32 -mattr=single-float < %s \
; RUN:   | FileCheck --check-prefixes=ALL,SYM32 %s

; RUN: llc -mtriple=mips64 -relocation-model=static -target-abi n64 -mattr=single-float < %s \
; RUN:   | FileCheck --check-prefixes=ALL,SYM64 %s
; RUN: llc -mtriple=mips64el -relocation-model=static -target-abi n64 -mattr=single-float < %s \
; RUN:   | FileCheck --check-prefixes=ALL,SYM64 %s

@fp128 = global fp128 zeroinitializer

define fp128 @retldouble() nounwind {
entry:
        %0 = load volatile fp128, ptr @fp128
        ret fp128 %0
}

; ALL-LABEL: retldouble:
; SYM32-DAG:         addiu [[R2:\$[0-9]+]], ${{[0-9]+}}, %lo(fp128)
; SYM64-DAG:         daddiu [[R2:\$[0-9]+]], ${{[0-9]+}}, %lo(fp128)

; ALL-DAG:           ld $2, %lo(fp128)([[R2]])
; ALL-DAG:           ld $3, 8([[R2]])
