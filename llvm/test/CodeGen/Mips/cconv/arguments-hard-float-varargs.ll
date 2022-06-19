; RUN: llc -mtriple=mips -relocation-model=static < %s \
; RUN:   | FileCheck --check-prefixes=ALL,SYM32,O32,O32BE %s
; RUN: llc -mtriple=mipsel -relocation-model=static < %s \
; RUN:   | FileCheck --check-prefixes=ALL,SYM32,O32,O32LE %s

; RUN-TODO: llc -mtriple=mips64 -relocation-model=static -target-abi o32 < %s \
; RUN-TODO:   | FileCheck --check-prefixes=ALL,SYM32,O32 %s
; RUN-TODO: llc -mtriple=mips64el -relocation-model=static -target-abi o32 < %s \
; RUN-TODO:   | FileCheck --check-prefixes=ALL,SYM32,O32 %s

; RUN: llc -mtriple=mips64 -relocation-model=static -target-abi n32 < %s \
; RUN:   | FileCheck --check-prefixes=ALL,SYM32,N32,NEW,NEWBE %s
; RUN: llc -mtriple=mips64el -relocation-model=static -target-abi n32 < %s \
; RUN:   | FileCheck --check-prefixes=ALL,SYM32,N32,NEW,NEWLE %s

; RUN: llc -mtriple=mips64 -relocation-model=static -target-abi n64 < %s \
; RUN:   | FileCheck --check-prefixes=ALL,SYM64,N64,NEW,NEWBE %s
; RUN: llc -mtriple=mips64el -relocation-model=static -target-abi n64 < %s \
; RUN:   | FileCheck --check-prefixes=ALL,SYM64,N64,NEW,NEWLE %s

; Test the effect of varargs on floating point types in the non-variable part
; of the argument list as specified by section 2 of the MIPSpro N32 Handbook.
;
; N32/N64 are almost identical in this area so many of their checks have been
; combined into the 'NEW' prefix (the N stands for New).
;
; On O32, varargs prevents all FPU argument register usage. This contradicts
; the N32 handbook, but agrees with the SYSV ABI and GCC's behaviour.

@floats = global [11 x float] zeroinitializer
@doubles = global [11 x double] zeroinitializer

define void @double_args(double %a, ...)
                         nounwind {
entry:
        %0 = getelementptr [11 x double], [11 x double]* @doubles, i32 0, i32 1
        store volatile double %a, double* %0

        %ap = alloca i8*
        %ap2 = bitcast i8** %ap to i8*
        call void @llvm.va_start(i8* %ap2)
        %b = va_arg i8** %ap, double
        %1 = getelementptr [11 x double], [11 x double]* @doubles, i32 0, i32 2
        store volatile double %b, double* %1
        call void @llvm.va_end(i8* %ap2)
        ret void
}

; ALL-LABEL: double_args:
; We won't test the way the global address is calculated in this test. This is
; just to get the register number for the other checks.
; SYM32-DAG:         addiu [[R2:\$[0-9]+]], ${{[0-9]+}}, %lo(doubles)
; SYM64-DAG:         daddiu [[R2:\$[0-9]+]], ${{[0-9]+}}, %lo(doubles)

; O32 forbids using floating point registers for the non-variable portion.
; N32/N64 allow it.
; O32BE-DAG:         mtc1 $5, [[FTMP1:\$f[0-9]*[02468]+]]
; O32BE-DAG:         mtc1 $4, [[FTMP2:\$f[0-9]*[13579]+]]
; O32LE-DAG:         mtc1 $4, [[FTMP1:\$f[0-9]*[02468]+]]
; O32LE-DAG:         mtc1 $5, [[FTMP2:\$f[0-9]*[13579]+]]
; O32-DAG:           sdc1 [[FTMP1]], 8([[R2]])
; NEW-DAG:           sdc1 $f12, 8([[R2]])

; The varargs portion is dumped to stack
; O32-DAG:           sw $6, 16($sp)
; O32-DAG:           sw $7, 20($sp)
; NEW-DAG:           sd $5, 8($sp)
; NEW-DAG:           sd $6, 16($sp)
; NEW-DAG:           sd $7, 24($sp)
; NEW-DAG:           sd $8, 32($sp)
; NEW-DAG:           sd $9, 40($sp)
; NEW-DAG:           sd $10, 48($sp)
; NEW-DAG:           sd $11, 56($sp)

; Get the varargs pointer
; O32 has 4 bytes padding, 4 bytes for the varargs pointer, and 8 bytes reserved
; for arguments 1 and 2.
; N32/N64 has 8 bytes for the varargs pointer, and no reserved area.
; O32-DAG:           addiu [[VAPTR:\$[0-9]+]], $sp, 16
; O32-DAG:           sw [[VAPTR]], 4($sp)
; N32-DAG:           addiu [[VAPTR:\$[0-9]+]], $sp, 8
; N32-DAG:           sw [[VAPTR]], 4($sp)
; N64-DAG:           daddiu [[VAPTR:\$[0-9]+]], $sp, 8
; N64-DAG:           sd [[VAPTR]], 0($sp)

; Increment the pointer then get the varargs arg
; LLVM will rebind the load to the stack pointer instead of the varargs pointer
; during lowering. This is fine and doesn't change the behaviour.
; O32-DAG:           addiu [[VAPTR]], [[VAPTR]], 8
; N32-DAG:           addiu [[VAPTR]], [[VAPTR]], 8
; N64-DAG:           daddiu [[VAPTR]], [[VAPTR]], 8
; O32-DAG:           ldc1 [[FTMP1:\$f[0-9]+]], 16($sp)
; NEW-DAG:           ldc1 [[FTMP1:\$f[0-9]+]], 8($sp)
; ALL-DAG:           sdc1 [[FTMP1]], 16([[R2]])

define void @float_args(float %a, ...) nounwind {
entry:
        %0 = getelementptr [11 x float], [11 x float]* @floats, i32 0, i32 1
        store volatile float %a, float* %0

        %ap = alloca i8*
        %ap2 = bitcast i8** %ap to i8*
        call void @llvm.va_start(i8* %ap2)
        %b = va_arg i8** %ap, float
        %1 = getelementptr [11 x float], [11 x float]* @floats, i32 0, i32 2
        store volatile float %b, float* %1
        call void @llvm.va_end(i8* %ap2)
        ret void
}

; ALL-LABEL: float_args:
; We won't test the way the global address is calculated in this test. This is
; just to get the register number for the other checks.
; SYM32-DAG:         addiu [[R2:\$[0-9]+]], ${{[0-9]+}}, %lo(floats)
; SYM64-DAG:         daddiu [[R2:\$[0-9]+]], ${{[0-9]+}}, %lo(floats)

; The first four arguments are the same in O32/N32/N64.
; The non-variable portion should be unaffected.
; O32-DAG:           mtc1 $4, $f0
; O32-DAG:           swc1 $f0, 4([[R2]])
; NEW-DAG:           swc1 $f12, 4([[R2]])

; The varargs portion is dumped to stack
; O32-DAG:           sw $5, 12($sp)
; O32-DAG:           sw $6, 16($sp)
; O32-DAG:           sw $7, 20($sp)
; NEW-DAG:           sd $5, 8($sp)
; NEW-DAG:           sd $6, 16($sp)
; NEW-DAG:           sd $7, 24($sp)
; NEW-DAG:           sd $8, 32($sp)
; NEW-DAG:           sd $9, 40($sp)
; NEW-DAG:           sd $10, 48($sp)
; NEW-DAG:           sd $11, 56($sp)

; Get the varargs pointer
; O32 has 4 bytes padding, 4 bytes for the varargs pointer, and should have 8
; bytes reserved for arguments 1 and 2 (the first float arg) but as discussed in
; arguments-float.ll, GCC doesn't agree with MD00305 and treats floats as 4
; bytes so we only have 12 bytes total.
; N32/N64 has 8 bytes for the varargs pointer, and no reserved area.
; O32-DAG:           addiu [[VAPTR:\$[0-9]+]], $sp, 12
; O32-DAG:           sw [[VAPTR]], 4($sp)
; N32-DAG:           addiu [[VAPTR:\$[0-9]+]], $sp, 8
; N32-DAG:           sw [[VAPTR]], 4($sp)
; N64-DAG:           daddiu [[VAPTR:\$[0-9]+]], $sp, 8
; N64-DAG:           sd [[VAPTR]], 0($sp)

; Increment the pointer then get the varargs arg
; LLVM will rebind the load to the stack pointer instead of the varargs pointer
; during lowering. This is fine and doesn't change the behaviour.
; Also, in big-endian mode the offset must be increased by 4 to retrieve the
; correct half of the argument slot.
;
; O32-DAG:           addiu [[VAPTR]], [[VAPTR]], 4
; N32-DAG:           addiu [[VAPTR]], [[VAPTR]], 8
; N64-DAG:           daddiu [[VAPTR]], [[VAPTR]], 8
; O32-DAG:           lwc1 [[FTMP1:\$f[0-9]+]], 12($sp)
; NEWLE-DAG:         lwc1 [[FTMP1:\$f[0-9]+]], 8($sp)
; NEWBE-DAG:         lwc1 [[FTMP1:\$f[0-9]+]], 12($sp)
; ALL-DAG:           swc1 [[FTMP1]], 8([[R2]])

declare void @llvm.va_start(i8*)
declare void @llvm.va_copy(i8*, i8*)
declare void @llvm.va_end(i8*)
