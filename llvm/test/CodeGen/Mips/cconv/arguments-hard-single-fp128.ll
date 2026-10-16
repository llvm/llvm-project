; RUN: llc -mtriple=mips64 -relocation-model=static -target-abi n32 -mattr=single-float < %s \
; RUN:   | FileCheck --check-prefixes=ALL,SYM32 %s
; RUN: llc -mtriple=mips64el -relocation-model=static -target-abi n32 -mattr=single-float < %s \
; RUN:   | FileCheck --check-prefixes=ALL,SYM32 %s

; RUN: llc -mtriple=mips64 -relocation-model=static -target-abi n64 -mattr=single-float < %s \
; RUN:   | FileCheck --check-prefixes=ALL,SYM64 %s
; RUN: llc -mtriple=mips64el -relocation-model=static -target-abi n64 -mattr=single-float < %s \
; RUN:   | FileCheck --check-prefixes=ALL,SYM64 %s

@ldoubles = global [11 x fp128] zeroinitializer

define void @ldouble_args(fp128 %a, fp128 %b, fp128 %c, fp128 %d, fp128 %e) nounwind {
entry:
        %0 = getelementptr [11 x fp128], ptr @ldoubles, i32 0, i32 1
        store volatile fp128 %a, ptr %0
        %1 = getelementptr [11 x fp128], ptr @ldoubles, i32 0, i32 2
        store volatile fp128 %b, ptr %1
        %2 = getelementptr [11 x fp128], ptr @ldoubles, i32 0, i32 3
        store volatile fp128 %c, ptr %2
        %3 = getelementptr [11 x fp128], ptr @ldoubles, i32 0, i32 4
        store volatile fp128 %d, ptr %3
        %4 = getelementptr [11 x fp128], ptr @ldoubles, i32 0, i32 5
        store volatile fp128 %e, ptr %4
        ret void
}

; ALL-LABEL: ldouble_args:
; We won't test the way the global address is calculated in this test. This is
; just to get the register number for the other checks.
; SYM32-DAG:         addiu [[R2:\$[0-9]+]], ${{[0-9]+}}, %lo(ldoubles)
; SYM64-DAG:         daddiu [[R2:\$[0-9]+]], ${{[0-9]+}}, %lo(ldoubles)

; The first four arguments are the same in N32/N64.
; ALL-DAG:           sd	$5, 24([[R2]])
; ALL-DAG:           sd	$4, 16([[R2]])
; ALL-DAG:           sd	$7, 40([[R2]])
; ALL-DAG:           sd	$6, 32([[R2]])
; ALL-DAG:           sd	$9, 56([[R2]])
; ALL-DAG:           sd	$8, 48([[R2]])
; ALL-DAG:           sd	$11, 72([[R2]])
; ALL-DAG:           sd	$10, 64([[R2]])
