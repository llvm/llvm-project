; RUN: llc -march=mips64 -relocation-model=static -mattr=+soft-float -target-abi n32 < %s | FileCheck --check-prefixes=ALL,SYM32 %s
; RUN: llc -march=mips64el -relocation-model=static -mattr=+soft-float -target-abi n32 < %s | FileCheck --check-prefixes=ALL,SYM32 %s

; RUN: llc -march=mips64 -relocation-model=static -mattr=+soft-float -target-abi n64 < %s | FileCheck --check-prefixes=ALL,SYM64 %s
; RUN: llc -march=mips64el -relocation-model=static -mattr=+soft-float -target-abi n64 < %s | FileCheck --check-prefixes=ALL,SYM64 %s

; Test the fp128 arguments for all ABI's and byte orders as specified
; by section 2 of the MIPSpro N32 Handbook.
;
; O32 is not tested because long double is the same as double on O32.

@ldoubles = global [11 x fp128] zeroinitializer

define void @ldouble_args(fp128 %a, fp128 %b, fp128 %c, fp128 %d, fp128 %e) nounwind {
entry:
        %0 = getelementptr [11 x fp128], [11 x fp128]* @ldoubles, i32 0, i32 1
        store volatile fp128 %a, fp128* %0
        %1 = getelementptr [11 x fp128], [11 x fp128]* @ldoubles, i32 0, i32 2
        store volatile fp128 %b, fp128* %1
        %2 = getelementptr [11 x fp128], [11 x fp128]* @ldoubles, i32 0, i32 3
        store volatile fp128 %c, fp128* %2
        %3 = getelementptr [11 x fp128], [11 x fp128]* @ldoubles, i32 0, i32 4
        store volatile fp128 %d, fp128* %3
        %4 = getelementptr [11 x fp128], [11 x fp128]* @ldoubles, i32 0, i32 5
        store volatile fp128 %e, fp128* %4
        ret void
}

; ALL-LABEL: ldouble_args:
; We won't test the way the global address is calculated in this test. This is
; just to get the register number for the other checks.
; SYM32-DAG:           addiu [[R2:\$[0-9]+]], ${{[0-9]+}}, %lo(ldoubles)
; SYM64-DAG:           ld [[R2:\$[0-9]]], %got_disp(ldoubles)(

; The first four arguments are the same in N32/N64.
; The first argument is floating point but soft-float is enabled so floating
; point registers are not used.
; ALL-DAG:           sd $4, 16([[R2]])
; ALL-DAG:           sd $5, 24([[R2]])
; ALL-DAG:           sd $6, 32([[R2]])
; ALL-DAG:           sd $7, 40([[R2]])
; ALL-DAG:           sd $8, 48([[R2]])
; ALL-DAG:           sd $9, 56([[R2]])
; ALL-DAG:           sd $10, 64([[R2]])
; ALL-DAG:           sd $11, 72([[R2]])

; N32/N64 have run out of registers and starts using the stack too
; ALL-DAG:           ld [[R3:\$[0-9]+]], 0($sp)
; ALL-DAG:           ld [[R4:\$[0-9]+]], 8($sp)
; ALL-DAG:           sd [[R3]], 80([[R2]])
; ALL-DAG:           sd [[R4]], 88([[R2]])
