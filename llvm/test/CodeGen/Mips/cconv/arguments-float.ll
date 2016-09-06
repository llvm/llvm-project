; RUN: llc -march=mips -relocation-model=static -mattr=+soft-float < %s | FileCheck --check-prefixes=ALL,SYM32,O32,O32BE %s
; RUN: llc -march=mipsel -relocation-model=static -mattr=+soft-float < %s | FileCheck --check-prefixes=ALL,SYM32,O32,O32LE %s

; RUN-TODO: llc -march=mips64 -relocation-model=static -mattr=+soft-float -target-abi o32 < %s | FileCheck --check-prefixes=ALL,SYM32,O32 %s
; RUN-TODO: llc -march=mips64el -relocation-model=static -mattr=+soft-float -target-abi o32 < %s | FileCheck --check-prefixes=ALL,SYM32,O32 %s

; RUN: llc -march=mips64 -relocation-model=static -mattr=+soft-float -target-abi n32 < %s | FileCheck --check-prefixes=ALL,SYM32,NEW %s
; RUN: llc -march=mips64el -relocation-model=static -mattr=+soft-float -target-abi n32 < %s | FileCheck --check-prefixes=ALL,SYM32,NEW %s

; RUN: llc -march=mips64 -relocation-model=static -mattr=+soft-float -target-abi n64 < %s | FileCheck --check-prefixes=ALL,SYM64,NEW %s
; RUN: llc -march=mips64el -relocation-model=static -mattr=+soft-float -target-abi n64 < %s | FileCheck --check-prefixes=ALL,SYM64,NEW %s

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
        %0 = getelementptr [11 x double], [11 x double]* @doubles, i32 0, i32 1
        store volatile double %a, double* %0
        %1 = getelementptr [11 x double], [11 x double]* @doubles, i32 0, i32 2
        store volatile double %b, double* %1
        %2 = getelementptr [11 x double], [11 x double]* @doubles, i32 0, i32 3
        store volatile double %c, double* %2
        %3 = getelementptr [11 x double], [11 x double]* @doubles, i32 0, i32 4
        store volatile double %d, double* %3
        %4 = getelementptr [11 x double], [11 x double]* @doubles, i32 0, i32 5
        store volatile double %e, double* %4
        %5 = getelementptr [11 x double], [11 x double]* @doubles, i32 0, i32 6
        store volatile double %f, double* %5
        %6 = getelementptr [11 x double], [11 x double]* @doubles, i32 0, i32 7
        store volatile double %g, double* %6
        %7 = getelementptr [11 x double], [11 x double]* @doubles, i32 0, i32 8
        store volatile double %h, double* %7
        %8 = getelementptr [11 x double], [11 x double]* @doubles, i32 0, i32 9
        store volatile double %i, double* %8
        ret void
}

; ALL-LABEL: double_args:
; We won't test the way the global address is calculated in this test. This is
; just to get the register number for the other checks.
; SYM32-DAG:           addiu [[R2:\$[0-9]+]], ${{[0-9]+}}, %lo(doubles)
; SYM64-DAG:           ld [[R2:\$[0-9]]], %got_disp(doubles)(

; The first four arguments are the same in O32/N32/N64.
; The first argument is floating point but soft-float is enabled so floating
; point registers are not used.
; O32-DAG:           sw $4, 8([[R2]])
; O32-DAG:           sw $5, 12([[R2]])
; NEW-DAG:           sd $4, 8([[R2]])

; O32-DAG:           sw $6, 16([[R2]])
; O32-DAG:           sw $7, 20([[R2]])
; NEW-DAG:           sd $5, 16([[R2]])

; O32 has run out of argument registers and starts using the stack
; O32-DAG:           lw [[R3:\$([0-9]+|gp)]], 24($sp)
; O32-DAG:           lw [[R4:\$([0-9]+|gp)]], 28($sp)
; O32-DAG:           sw [[R3]], 24([[R2]])
; O32-DAG:           sw [[R4]], 28([[R2]])
; NEW-DAG:           sd $6, 24([[R2]])

; O32-DAG:           lw [[R3:\$([0-9]+|gp)]], 32($sp)
; O32-DAG:           lw [[R4:\$([0-9]+|gp)]], 36($sp)
; O32-DAG:           sw [[R3]], 32([[R2]])
; O32-DAG:           sw [[R4]], 36([[R2]])
; NEW-DAG:           sd $7, 32([[R2]])

; O32-DAG:           lw [[R3:\$([0-9]+|gp)]], 40($sp)
; O32-DAG:           lw [[R4:\$([0-9]+|gp)]], 44($sp)
; O32-DAG:           sw [[R3]], 40([[R2]])
; O32-DAG:           sw [[R4]], 44([[R2]])
; NEW-DAG:           sd $8, 40([[R2]])

; O32-DAG:           lw [[R3:\$([0-9]+|gp)]], 48($sp)
; O32-DAG:           lw [[R4:\$([0-9]+|gp)]], 52($sp)
; O32-DAG:           sw [[R3]], 48([[R2]])
; O32-DAG:           sw [[R4]], 52([[R2]])
; NEW-DAG:           sd $9, 48([[R2]])

; O32-DAG:           lw [[R3:\$([0-9]+|gp)]], 56($sp)
; O32-DAG:           lw [[R4:\$([0-9]+|gp)]], 60($sp)
; O32-DAG:           sw [[R3]], 56([[R2]])
; O32-DAG:           sw [[R4]], 60([[R2]])
; NEW-DAG:           sd $10, 56([[R2]])

; N32/N64 have run out of registers and starts using the stack too
; O32-DAG:           lw [[R3:\$[0-9]+]], 64($sp)
; O32-DAG:           lw [[R4:\$[0-9]+]], 68($sp)
; O32-DAG:           sw [[R3]], 64([[R2]])
; O32-DAG:           sw [[R4]], 68([[R2]])
; NEW-DAG:           ld [[R3:\$[0-9]+]], 0($sp)
; NEW-DAG:           sd $11, 64([[R2]])

define void @float_args(float %a, float %b, float %c, float %d, float %e,
                        float %f, float %g, float %h, float %i, float %j)
                       nounwind {
entry:
        %0 = getelementptr [11 x float], [11 x float]* @floats, i32 0, i32 1
        store volatile float %a, float* %0
        %1 = getelementptr [11 x float], [11 x float]* @floats, i32 0, i32 2
        store volatile float %b, float* %1
        %2 = getelementptr [11 x float], [11 x float]* @floats, i32 0, i32 3
        store volatile float %c, float* %2
        %3 = getelementptr [11 x float], [11 x float]* @floats, i32 0, i32 4
        store volatile float %d, float* %3
        %4 = getelementptr [11 x float], [11 x float]* @floats, i32 0, i32 5
        store volatile float %e, float* %4
        %5 = getelementptr [11 x float], [11 x float]* @floats, i32 0, i32 6
        store volatile float %f, float* %5
        %6 = getelementptr [11 x float], [11 x float]* @floats, i32 0, i32 7
        store volatile float %g, float* %6
        %7 = getelementptr [11 x float], [11 x float]* @floats, i32 0, i32 8
        store volatile float %h, float* %7
        %8 = getelementptr [11 x float], [11 x float]* @floats, i32 0, i32 9
        store volatile float %i, float* %8
        %9 = getelementptr [11 x float], [11 x float]* @floats, i32 0, i32 10
        store volatile float %j, float* %9
        ret void
}

; ALL-LABEL: float_args:
; We won't test the way the global address is calculated in this test. This is
; just to get the register number for the other checks.
; SYM32-DAG:           addiu [[R2:\$[0-9]+]], ${{[0-9]+}}, %lo(floats)
; SYM64-DAG:           ld [[R2:\$[0-9]]], %got_disp(floats)(

; The first four arguments are the same in O32/N32/N64.
; The first argument is floating point but soft-float is enabled so floating
; point registers are not used.
; MD00305 and GCC disagree on this one. MD00305 says that floats are treated
; as 8-byte aligned and occupy two slots on O32. GCC is treating them as 4-byte
; aligned and occupying one slot. We'll use GCC's definition.
; ALL-DAG:           sw $4, 4([[R2]])
; ALL-DAG:           sw $5, 8([[R2]])
; ALL-DAG:           sw $6, 12([[R2]])
; ALL-DAG:           sw $7, 16([[R2]])

; O32 has run out of argument registers and starts using the stack
; O32-DAG:           lw [[R3:\$[0-9]+]], 16($sp)
; O32-DAG:           sw [[R3]], 20([[R2]])
; NEW-DAG:           sw $8, 20([[R2]])

; O32-DAG:           lw [[R3:\$[0-9]+]], 20($sp)
; O32-DAG:           sw [[R3]], 24([[R2]])
; NEW-DAG:           sw $9, 24([[R2]])

; O32-DAG:           lw [[R3:\$[0-9]+]], 24($sp)
; O32-DAG:           sw [[R3]], 28([[R2]])
; NEW-DAG:           sw $10, 28([[R2]])

; O32-DAG:           lw [[R3:\$[0-9]+]], 28($sp)
; O32-DAG:           sw [[R3]], 32([[R2]])
; NEW-DAG:           sw $11, 32([[R2]])

; N32/N64 have run out of registers and start using the stack too
; O32-DAG:           lw [[R3:\$[0-9]+]], 32($sp)
; O32-DAG:           sw [[R3]], 36([[R2]])
; NEW-DAG:           lw [[R3:\$[0-9]+]], 0($sp)
; NEW-DAG:           sw [[R3]], 36([[R2]])

define void @double_arg2(i8 %a, double %b) nounwind {
entry:
        %0 = getelementptr [11 x i8], [11 x i8]* @bytes, i32 0, i32 1
        store volatile i8 %a, i8* %0
        %1 = getelementptr [11 x double], [11 x double]* @doubles, i32 0, i32 1
        store volatile double %b, double* %1
        ret void
}

; ALL-LABEL: double_arg2:
; We won't test the way the global address is calculated in this test. This is
; just to get the register number for the other checks.
; SYM32-DAG:           addiu [[R1:\$[0-9]+]], ${{[0-9]+}}, %lo(bytes)
; SYM64-DAG:           ld [[R1:\$[0-9]]], %got_disp(bytes)(
; SYM32-DAG:           addiu [[R2:\$[0-9]+]], ${{[0-9]+}}, %lo(doubles)
; SYM64-DAG:           ld [[R2:\$[0-9]]], %got_disp(doubles)(

; The first four arguments are the same in O32/N32/N64.
; The first argument isn't floating point so floating point registers are not
; used.
; The second slot is insufficiently aligned for double on O32 so it is skipped.
; Also, double occupies two slots on O32 and only one for N32/N64.
; ALL-DAG:           sb $4, 1([[R1]])
; O32-DAG:           sw $6, 8([[R2]])
; O32-DAG:           sw $7, 12([[R2]])
; NEW-DAG:           sd $5, 8([[R2]])

define void @float_arg2(i8 signext %a, float %b) nounwind {
entry:
        %0 = getelementptr [11 x i8], [11 x i8]* @bytes, i32 0, i32 1
        store volatile i8 %a, i8* %0
        %1 = getelementptr [11 x float], [11 x float]* @floats, i32 0, i32 1
        store volatile float %b, float* %1
        ret void
}

; ALL-LABEL: float_arg2:
; We won't test the way the global address is calculated in this test. This is
; just to get the register number for the other checks.
; SYM32-DAG:           addiu [[R1:\$[0-9]+]], ${{[0-9]+}}, %lo(bytes)
; SYM64-DAG:           ld [[R1:\$[0-9]]], %got_disp(bytes)(
; SYM32-DAG:           addiu [[R2:\$[0-9]+]], ${{[0-9]+}}, %lo(floats)
; SYM64-DAG:           ld [[R2:\$[0-9]]], %got_disp(floats)(

; The first four arguments are the same in O32/N32/N64.
; The first argument isn't floating point so floating point registers are not
; used.
; MD00305 and GCC disagree on this one. MD00305 says that floats are treated
; as 8-byte aligned and occupy two slots on O32. GCC is treating them as 4-byte
; aligned and occupying one slot. We'll use GCC's definition.
; ALL-DAG:           sb $4, 1([[R1]])
; ALL-DAG:           sw $5, 4([[R2]])
