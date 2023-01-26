; RUN: llc -mtriple=mips -relocation-model=static < %s | FileCheck --check-prefixes=ALL,SYM32,O32,O32BE %s
; RUN: llc -mtriple=mipsel -relocation-model=static < %s | FileCheck --check-prefixes=ALL,SYM32,O32,O32LE %s

; RUN-TODO: llc -mtriple=mips64 -relocation-model=static -target-abi o32 < %s | FileCheck --check-prefixes=ALL,SYM32,O32 %s
; RUN-TODO: llc -mtriple=mips64el -relocation-model=static -target-abi o32 < %s | FileCheck --check-prefixes=ALL,SYM32,O32 %s

; RUN: llc -mtriple=mips64 -relocation-model=static -target-abi n32 < %s | FileCheck --check-prefixes=ALL,SYM32,NEW %s
; RUN: llc -mtriple=mips64el -relocation-model=static -target-abi n32 < %s | FileCheck --check-prefixes=ALL,SYM32,NEW %s

; RUN: llc -mtriple=mips64 -relocation-model=static -target-abi n64 < %s | FileCheck --check-prefixes=ALL,SYM64,NEW %s
; RUN: llc -mtriple=mips64el -relocation-model=static -target-abi n64 < %s | FileCheck --check-prefixes=ALL,SYM64,NEW %s

; Test the floating point arguments for all ABI's and byte orders as specified
; by section 5 of MD00305 (MIPS ABIs Described).
;
; N32/N64 are identical in this area so their checks have been combined into
; the 'NEW' prefix (the N stands for New).

@bytes = global [11 x i8] zeroinitializer
@dwords = global [11 x i64] zeroinitializer
@floats = global [11 x float] zeroinitializer
@doubles = global [11 x double] zeroinitializer

define void @double_args(double %a, double %b, double %c, double %d, double %e,
                         double %f, double %g, double %h, double %i) nounwind {
entry:
        %0 = getelementptr [11 x double], ptr @doubles, i32 0, i32 1
        store volatile double %a, ptr %0
        %1 = getelementptr [11 x double], ptr @doubles, i32 0, i32 2
        store volatile double %b, ptr %1
        %2 = getelementptr [11 x double], ptr @doubles, i32 0, i32 3
        store volatile double %c, ptr %2
        %3 = getelementptr [11 x double], ptr @doubles, i32 0, i32 4
        store volatile double %d, ptr %3
        %4 = getelementptr [11 x double], ptr @doubles, i32 0, i32 5
        store volatile double %e, ptr %4
        %5 = getelementptr [11 x double], ptr @doubles, i32 0, i32 6
        store volatile double %f, ptr %5
        %6 = getelementptr [11 x double], ptr @doubles, i32 0, i32 7
        store volatile double %g, ptr %6
        %7 = getelementptr [11 x double], ptr @doubles, i32 0, i32 8
        store volatile double %h, ptr %7
        %8 = getelementptr [11 x double], ptr @doubles, i32 0, i32 9
        store volatile double %i, ptr %8
        ret void
}

; ALL-LABEL: double_args:
; We won't test the way the global address is calculated in this test. This is
; just to get the register number for the other checks.
; SYM32-DAG:           addiu [[R2:\$[0-9]+]], ${{[0-9]+}}, %lo(doubles)
; SYM64-DAG:           daddiu [[R2:\$[0-9]]], ${{[0-9]+}}, %lo(doubles)

; The first argument is floating point so floating point registers are used.
; The first argument is the same for O32/N32/N64 but the second argument differs
; by register
; ALL-DAG:           sdc1 $f12, 8([[R2]])
; O32-DAG:           sdc1 $f14, 16([[R2]])
; NEW-DAG:           sdc1 $f13, 16([[R2]])

; O32 has run out of argument registers and starts using the stack
; O32-DAG:           ldc1 [[F1:\$f[0-9]+]], 16($sp)
; O32-DAG:           sdc1 [[F1]], 24([[R2]])
; NEW-DAG:           sdc1 $f14, 24([[R2]])
; O32-DAG:           ldc1 [[F1:\$f[0-9]+]], 24($sp)
; O32-DAG:           sdc1 [[F1]], 32([[R2]])
; NEW-DAG:           sdc1 $f15, 32([[R2]])
; O32-DAG:           ldc1 [[F1:\$f[0-9]+]], 32($sp)
; O32-DAG:           sdc1 [[F1]], 40([[R2]])
; NEW-DAG:           sdc1 $f16, 40([[R2]])
; O32-DAG:           ldc1 [[F1:\$f[0-9]+]], 40($sp)
; O32-DAG:           sdc1 [[F1]], 48([[R2]])
; NEW-DAG:           sdc1 $f17, 48([[R2]])
; O32-DAG:           ldc1 [[F1:\$f[0-9]+]], 48($sp)
; O32-DAG:           sdc1 [[F1]], 56([[R2]])
; NEW-DAG:           sdc1 $f18, 56([[R2]])
; O32-DAG:           ldc1 [[F1:\$f[0-9]+]], 56($sp)
; O32-DAG:           sdc1 [[F1]], 64([[R2]])
; NEW-DAG:           sdc1 $f19, 64([[R2]])

; N32/N64 have run out of registers and start using the stack too
; O32-DAG:           ldc1 [[F1:\$f[0-9]+]], 64($sp)
; O32-DAG:           sdc1 [[F1]], 72([[R2]])
; NEW-DAG:           ldc1 [[F1:\$f[0-9]+]], 0($sp)
; NEW-DAG:           sdc1 [[F1]], 72([[R2]])

define void @float_args(float %a, float %b, float %c, float %d, float %e,
                        float %f, float %g, float %h, float %i) nounwind {
entry:
        %0 = getelementptr [11 x float], ptr @floats, i32 0, i32 1
        store volatile float %a, ptr %0
        %1 = getelementptr [11 x float], ptr @floats, i32 0, i32 2
        store volatile float %b, ptr %1
        %2 = getelementptr [11 x float], ptr @floats, i32 0, i32 3
        store volatile float %c, ptr %2
        %3 = getelementptr [11 x float], ptr @floats, i32 0, i32 4
        store volatile float %d, ptr %3
        %4 = getelementptr [11 x float], ptr @floats, i32 0, i32 5
        store volatile float %e, ptr %4
        %5 = getelementptr [11 x float], ptr @floats, i32 0, i32 6
        store volatile float %f, ptr %5
        %6 = getelementptr [11 x float], ptr @floats, i32 0, i32 7
        store volatile float %g, ptr %6
        %7 = getelementptr [11 x float], ptr @floats, i32 0, i32 8
        store volatile float %h, ptr %7
        %8 = getelementptr [11 x float], ptr @floats, i32 0, i32 9
        store volatile float %i, ptr %8
        ret void
}

; ALL-LABEL: float_args:
; We won't test the way the global address is calculated in this test. This is
; just to get the register number for the other checks.
; SYM32-DAG:           addiu  [[R1:\$[0-9]+]], ${{[0-9]+}}, %lo(floats)
; SYM64-DAG:           daddiu [[R1:\$[0-9]]], ${{[0-9]+}}, %lo(floats)

; The first argument is floating point so floating point registers are used.
; The first argument is the same for O32/N32/N64 but the second argument differs
; by register
; ALL-DAG:           swc1 $f12, 4([[R1]])
; O32-DAG:           swc1 $f14, 8([[R1]])
; NEW-DAG:           swc1 $f13, 8([[R1]])

; O32 has run out of argument registers and (in theory) starts using the stack
; I've yet to find a reference in the documentation about this but GCC uses up
; the remaining two argument slots in the GPR's first. We'll do the same for
; compatibility.
; O32-DAG:           mtc1 $6, $f0
; O32-DAG:           swc1 $f0, 12([[R1]])
; NEW-DAG:           swc1 $f14, 12([[R1]])
; O32-DAG:           mtc1 $7, $f0
; O32-DAG:           swc1 $f0, 16([[R1]])
; NEW-DAG:           swc1 $f15, 16([[R1]])

; O32 is definitely out of registers now and switches to the stack.
; O32-DAG:           lwc1 [[F1:\$f[0-9]+]], 16($sp)
; O32-DAG:           swc1 [[F1]], 20([[R1]])
; NEW-DAG:           swc1 $f16, 20([[R1]])
; O32-DAG:           lwc1 [[F1:\$f[0-9]+]], 20($sp)
; O32-DAG:           swc1 [[F1]], 24([[R1]])
; NEW-DAG:           swc1 $f17, 24([[R1]])
; O32-DAG:           lwc1 [[F1:\$f[0-9]+]], 24($sp)
; O32-DAG:           swc1 [[F1]], 28([[R1]])
; NEW-DAG:           swc1 $f18, 28([[R1]])
; O32-DAG:           lwc1 [[F1:\$f[0-9]+]], 28($sp)
; O32-DAG:           swc1 [[F1]], 32([[R1]])
; NEW-DAG:           swc1 $f19, 32([[R1]])

; N32/N64 have run out of registers and start using the stack too
; O32-DAG:           lwc1 [[F1:\$f[0-9]+]], 32($sp)
; O32-DAG:           swc1 [[F1]], 36([[R1]])
; NEW-DAG:           lwc1 [[F1:\$f[0-9]+]], 0($sp)
; NEW-DAG:           swc1 [[F1]], 36([[R1]])


define void @double_arg2(i8 %a, double %b) nounwind {
entry:
        %0 = getelementptr [11 x i8], ptr @bytes, i32 0, i32 1
        store volatile i8 %a, ptr %0
        %1 = getelementptr [11 x double], ptr @doubles, i32 0, i32 1
        store volatile double %b, ptr %1
        ret void
}

; ALL-LABEL: double_arg2:
; We won't test the way the global address is calculated in this test. This is
; just to get the register number for the other checks.
; SYM32-DAG:           addiu [[R1:\$[0-9]+]], ${{[0-9]+}}, %lo(bytes)
; SYM64-DAG:           daddiu [[R1:\$[0-9]]], ${{[0-9]+}}, %lo(bytes)
; SYM32-DAG:           addiu [[R2:\$[0-9]+]], ${{[0-9]+}}, %lo(doubles)
; SYM64-DAG:           daddiu [[R2:\$[0-9]]], ${{[0-9]+}}, %lo(doubles)

; The first argument is the same in O32/N32/N64.
; ALL-DAG:           sb $4, 1([[R1]])

; The first argument isn't floating point so floating point registers are not
; used in O32, but N32/N64 will still use them.
; The second slot is insufficiently aligned for double on O32 so it is skipped.
; Also, double occupies two slots on O32 and only one for N32/N64.
; O32LE-DAG:           mtc1 $6, [[F1:\$f[0-9]*[02468]+]]
; O32LE-DAG:           mtc1 $7, [[F2:\$f[0-9]*[13579]+]]
; O32BE-DAG:           mtc1 $6, [[F2:\$f[0-9]*[13579]+]]
; O32BE-DAG:           mtc1 $7, [[F1:\$f[0-9]*[02468]+]]
; O32-DAG:           sdc1 [[F1]], 8([[R2]])
; NEW-DAG:           sdc1 $f13, 8([[R2]])

define void @float_arg2(i8 %a, float %b) nounwind {
entry:
        %0 = getelementptr [11 x i8], ptr @bytes, i32 0, i32 1
        store volatile i8 %a, ptr %0
        %1 = getelementptr [11 x float], ptr @floats, i32 0, i32 1
        store volatile float %b, ptr %1
        ret void
}

; ALL-LABEL: float_arg2:
; We won't test the way the global address is calculated in this test. This is
; just to get the register number for the other checks.
; SYM32-DAG:           addiu [[R1:\$[0-9]+]], ${{[0-9]+}}, %lo(bytes)
; SYM64-DAG:           daddiu [[R1:\$[0-9]]], ${{[0-9]+}}, %lo(bytes)
; SYM32-DAG:           addiu [[R2:\$[0-9]+]], ${{[0-9]+}}, %lo(floats)
; SYM64-DAG:           daddiu [[R2:\$[0-9]]], ${{[0-9]+}}, %lo(floats)

; The first argument is the same in O32/N32/N64.
; ALL-DAG:           sb $4, 1([[R1]])

; The first argument isn't floating point so floating point registers are not
; used in O32, but N32/N64 will still use them.
; MD00305 and GCC disagree on this one. MD00305 says that floats are treated
; as 8-byte aligned and occupy two slots on O32. GCC is treating them as 4-byte
; aligned and occupying one slot. We'll use GCC's definition.
; O32-DAG:           mtc1 $5, $f0
; O32-DAG:           swc1 $f0, 4([[R2]])
; NEW-DAG:           swc1 $f13, 4([[R2]])
