; RUN: llc -march=mips -relocation-model=static < %s | FileCheck --check-prefixes=ALL,SYM32,O32 %s
; RUN: llc -march=mipsel -relocation-model=static < %s | FileCheck --check-prefixes=ALL,SYM32,O32 %s

; RUN-TODO: llc -march=mips64 -relocation-model=static -target-abi n32 < %s | FileCheck --check-prefixes=ALL,SYM32,O32 %s
; RUN-TODO: llc -march=mips64el -relocation-model=static -target-abi n32 < %s | FileCheck --check-prefixes=ALL,SYM32,O32 %s

; RUN: llc -march=mips64 -relocation-model=static -target-abi n32 < %s | FileCheck --check-prefixes=ALL,SYM32,NEW %s
; RUN: llc -march=mips64el -relocation-model=static -target-abi n32 < %s | FileCheck --check-prefixes=ALL,SYM32,NEW %s

; RUN: llc -march=mips64 -relocation-model=static -target-abi n64 < %s | FileCheck --check-prefixes=ALL,SYM64,NEW %s
; RUN: llc -march=mips64el -relocation-model=static -target-abi n64 < %s | FileCheck --check-prefixes=ALL,SYM64,NEW %s

; Test the integer arguments for all ABI's and byte orders as specified by
; section 5 of MD00305 (MIPS ABIs Described).
;
; N32/N64 are identical in this area so their checks have been combined into
; the 'NEW' prefix (the N stands for New).
;
; Varargs are covered in arguments-hard-float-varargs.ll.

@bytes = global [11 x i8] zeroinitializer
@dwords = global [11 x i64] zeroinitializer
@floats = global [11 x float] zeroinitializer
@doubles = global [11 x double] zeroinitializer

define void @align_to_arg_slots(i8 signext %a, i8 signext %b, i8 signext %c,
                                i8 signext %d, i8 signext %e, i8 signext %f,
                                i8 signext %g, i8 signext %h, i8 signext %i,
                                i8 signext %j) nounwind {
entry:
        %0 = getelementptr [11 x i8], [11 x i8]* @bytes, i32 0, i32 1
        store volatile i8 %a, i8* %0
        %1 = getelementptr [11 x i8], [11 x i8]* @bytes, i32 0, i32 2
        store volatile i8 %b, i8* %1
        %2 = getelementptr [11 x i8], [11 x i8]* @bytes, i32 0, i32 3
        store volatile i8 %c, i8* %2
        %3 = getelementptr [11 x i8], [11 x i8]* @bytes, i32 0, i32 4
        store volatile i8 %d, i8* %3
        %4 = getelementptr [11 x i8], [11 x i8]* @bytes, i32 0, i32 5
        store volatile i8 %e, i8* %4
        %5 = getelementptr [11 x i8], [11 x i8]* @bytes, i32 0, i32 6
        store volatile i8 %f, i8* %5
        %6 = getelementptr [11 x i8], [11 x i8]* @bytes, i32 0, i32 7
        store volatile i8 %g, i8* %6
        %7 = getelementptr [11 x i8], [11 x i8]* @bytes, i32 0, i32 8
        store volatile i8 %h, i8* %7
        %8 = getelementptr [11 x i8], [11 x i8]* @bytes, i32 0, i32 9
        store volatile i8 %i, i8* %8
        %9 = getelementptr [11 x i8], [11 x i8]* @bytes, i32 0, i32 10
        store volatile i8 %j, i8* %9
        ret void
}

; ALL-LABEL: align_to_arg_slots:
; We won't test the way the global address is calculated in this test. This is
; just to get the register number for the other checks.
; SYM32-DAG:           addiu [[R1:\$[0-9]+]], ${{[0-9]+}}, %lo(bytes)
; SYM64-DAG:           daddiu [[R1:\$[0-9]+]], ${{[0-9]+}},  %lo(bytes)

; COM: The first four arguments are the same in O32/N32/N64
; ALL-DAG:           sb $4, 1([[R1]])
; ALL-DAG:           sb $5, 2([[R1]])
; ALL-DAG:           sb $6, 3([[R1]])
; ALL-DAG:           sb $7, 4([[R1]])

; COM: N32/N64 get an extra four arguments in registers
; COM: O32 starts loading from the stack. The addresses start at 16 because space is
; COM: always reserved for the first four arguments.
; O32-DAG:           lw [[R3:\$[0-9]+]], 16($sp)
; O32-DAG:           sb [[R3]], 5([[R1]])
; NEW-DAG:           sb $8, 5([[R1]])
; O32-DAG:           lw [[R3:\$[0-9]+]], 20($sp)
; O32-DAG:           sb [[R3]], 6([[R1]])
; NEW-DAG:           sb $9, 6([[R1]])
; O32-DAG:           lw [[R3:\$[0-9]+]], 24($sp)
; O32-DAG:           sb [[R3]], 7([[R1]])
; NEW-DAG:           sb $10, 7([[R1]])
; O32-DAG:           lw [[R3:\$[0-9]+]], 28($sp)
; O32-DAG:           sb [[R3]], 8([[R1]])
; NEW-DAG:           sb $11, 8([[R1]])

; COM: O32/N32/N64 are accessing the stack at this point.
; COM: Unlike O32, N32/N64 do not reserve space for the arguments.
; COM: increase by 4 for O32 and 8 for N32/N64.
; O32-DAG:           lw [[R3:\$[0-9]+]], 32($sp)
; O32-DAG:           sb [[R3]], 9([[R1]])
; NEW-DAG:           ld [[R3:\$[0-9]+]], 0($sp)
; NEW-DAG:           sb [[R3]], 9([[R1]])
; O32-DAG:           lw [[R3:\$[0-9]+]], 36($sp)
; O32-DAG:           sb [[R3]], 10([[R1]])
; NEW-DAG:           ld [[R3:\$[0-9]+]], 8($sp)
; NEW-DAG:           sb [[R3]], 10([[R1]])

define void @slot_skipping(i8 signext %a, i64 signext %b, i8 signext %c,
                           i8 signext %d, i8 signext %e, i8 signext %f,
                           i8 signext %g, i64 signext %i, i8 signext %j) nounwind {
entry:
        %0 = getelementptr [11 x i8], [11 x i8]* @bytes, i32 0, i32 1
        store volatile i8 %a, i8* %0
        %1 = getelementptr [11 x i64], [11 x i64]* @dwords, i32 0, i32 1
        store volatile i64 %b, i64* %1
        %2 = getelementptr [11 x i8], [11 x i8]* @bytes, i32 0, i32 2
        store volatile i8 %c, i8* %2
        %3 = getelementptr [11 x i8], [11 x i8]* @bytes, i32 0, i32 3
        store volatile i8 %d, i8* %3
        %4 = getelementptr [11 x i8], [11 x i8]* @bytes, i32 0, i32 4
        store volatile i8 %e, i8* %4
        %5 = getelementptr [11 x i8], [11 x i8]* @bytes, i32 0, i32 5
        store volatile i8 %f, i8* %5
        %6 = getelementptr [11 x i8], [11 x i8]* @bytes, i32 0, i32 6
        store volatile i8 %g, i8* %6
        %7 = getelementptr [11 x i64], [11 x i64]* @dwords, i32 0, i32 2
        store volatile i64 %i, i64* %7
        %8 = getelementptr [11 x i8], [11 x i8]* @bytes, i32 0, i32 7
        store volatile i8 %j, i8* %8
        ret void
}

; ALL-LABEL: slot_skipping:
; We won't test the way the global address is calculated in this test. This is
; just to get the register number for the other checks.
; SYM32-DAG:           addiu [[R1:\$[0-9]+]], ${{[0-9]+}}, %lo(bytes)
; SYM64-DAG:           daddiu [[R1:\$[0-9]+]], ${{[0-9]+}},  %lo(bytes)
; SYM32-DAG:           addiu [[R2:\$[0-9]+]], ${{[0-9]+}}, %lo(dwords)
; SYM64-DAG:           daddiu [[R2:\$[0-9]+]], ${{[0-9]+}}, %lo(dwords)

; The first argument is the same in O32/N32/N64.
; ALL-DAG:           sb $4, 1([[R1]])

; COM: The second slot is insufficiently aligned for i64 on O32 so it is skipped.
; COM: Also, i64 occupies two slots on O32 and only one for N32/N64.
; O32-DAG:           sw $6, 8([[R2]])
; O32-DAG:           sw $7, 12([[R2]])
; NEW-DAG:           sd $5, 8([[R2]])

; COM: N32/N64 get an extra four arguments in registers and still have two left from
; COM: the first four.
; COM: O32 starts loading from the stack. The addresses start at 16 because space is
; COM: always reserved for the first four arguments.
; COM: It's not clear why O32 uses lbu for this argument, but it's not wrong so we'll
; COM: accept it for now. The only IR difference is that this argument has
; COM: anyext from i8 and align 8 on it.
; O32-DAG:           lw [[R3:\$[0-9]+]], 16($sp)
; O32-DAG:           sb [[R3]], 2([[R1]])
; NEW-DAG:           sb $6, 2([[R1]])
; O32-DAG:           lw [[R3:\$[0-9]+]], 20($sp)
; O32-DAG:           sb [[R3]], 3([[R1]])
; NEW-DAG:           sb $7, 3([[R1]])
; O32-DAG:           lw [[R3:\$[0-9]+]], 24($sp)
; O32-DAG:           sb [[R3]], 4([[R1]])
; NEW-DAG:           sb $8, 4([[R1]])
; O32-DAG:           lw [[R3:\$[0-9]+]], 28($sp)
; O32-DAG:           sb [[R3]], 5([[R1]])
; NEW-DAG:           sb $9, 5([[R1]])

; O32-DAG:           lw [[R3:\$[0-9]+]], 32($sp)
; O32-DAG:           sb [[R3]], 6([[R1]])
; NEW-DAG:           sb $10, 6([[R1]])

; O32-DAG:           lw [[R3:\$[0-9]+]], 40($sp)
; O32-DAG:           sw [[R3]], 16([[R2]])
; O32-DAG:           lw [[R3:\$[0-9]+]], 44($sp)
; O32-DAG:           sw [[R3]], 20([[R2]])
; NEW-DAG:           sd $11, 16([[R2]])

; COM: O32/N32/N64 are accessing the stack at this point.
; COM: Unlike O32, N32/N64 do not reserve space for the arguments.
; COM: increase by 4 for O32 and 8 for N32/N64.
; O32-DAG:           lw [[R3:\$[0-9]+]], 48($sp)
; O32-DAG:           sb [[R3]], 7([[R1]])
; NEW-DAG:           ld [[R3:\$[0-9]+]], 0($sp)
; NEW-DAG:           sb [[R3]], 7([[R1]])
