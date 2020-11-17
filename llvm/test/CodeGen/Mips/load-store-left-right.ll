; RUN: llc -march=mipsel   -mcpu=mips32              -relocation-model=pic < %s | FileCheck -check-prefixes=ALL,MIPS32,MIPS32-EL %s
; RUN: llc -march=mips     -mcpu=mips32              -relocation-model=pic < %s | FileCheck -check-prefixes=ALL,MIPS32,MIPS32-EB %s
; RUN: llc -march=mipsel   -mcpu=mips32r2            -relocation-model=pic < %s | FileCheck -check-prefixes=ALL,MIPS32,MIPS32-EL %s
; RUN: llc -march=mips     -mcpu=mips32r2            -relocation-model=pic < %s | FileCheck -check-prefixes=ALL,MIPS32,MIPS32-EB %s
; RUN: llc -march=mipsel   -mcpu=mips32r6            -relocation-model=pic < %s | FileCheck -check-prefixes=ALL,MIPS32R6,MIPS32R6-EL %s
; RUN: llc -march=mips     -mcpu=mips32r6            -relocation-model=pic < %s | FileCheck -check-prefixes=ALL,MIPS32R6,MIPS32R6-EB %s
; RUN: llc -march=mips64el -mcpu=mips4    -target-abi=n64 -relocation-model=pic < %s | FileCheck -check-prefixes=ALL,MIPS64,MIPS64-EL %s
; RUN: llc -march=mips64   -mcpu=mips4    -target-abi=n64 -relocation-model=pic < %s | FileCheck -check-prefixes=ALL,MIPS64,MIPS64-EB %s
; RUN: llc -march=mips64el -mcpu=mips64   -target-abi=n64 -relocation-model=pic < %s | FileCheck -check-prefixes=ALL,MIPS64,MIPS64-EL %s
; RUN: llc -march=mips64   -mcpu=mips64   -target-abi=n64 -relocation-model=pic < %s | FileCheck -check-prefixes=ALL,MIPS64,MIPS64-EB %s
; RUN: llc -march=mips64el -mcpu=mips64r2 -target-abi=n64 -relocation-model=pic < %s | FileCheck -check-prefixes=ALL,MIPS64,MIPS64R2-EL %s
; RUN: llc -march=mips64   -mcpu=mips64r2 -target-abi=n64 -relocation-model=pic < %s | FileCheck -check-prefixes=ALL,MIPS64,MIPS64R2-EB %s
; RUN: llc -march=mips64el -mcpu=mips64r6 -target-abi=n64 -relocation-model=pic < %s | FileCheck -check-prefixes=ALL,MIPS64R6 %s
; RUN: llc -march=mips64   -mcpu=mips64r6 -target-abi=n64 -relocation-model=pic < %s | FileCheck -check-prefixes=ALL,MIPS64R6 %s

%struct.SLL = type { i64 }
%struct.SI = type { i32 }
%struct.SUI = type { i32 }

@sll = common global %struct.SLL zeroinitializer, align 1
@si = common global %struct.SI zeroinitializer, align 1
@sui = common global %struct.SUI zeroinitializer, align 1

define i32 @load_SI() nounwind readonly {
entry:
; ALL-LABEL: load_SI:

; MIPS32-EL:     lwl $[[R0:[0-9]+]], 3($[[R1:[0-9]+]])
; MIPS32-EL:     lwr $[[R0]], 0($[[R1]])

; MIPS32-EB:     lwl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; MIPS32-EB:     lwr $[[R0]], 3($[[R1]])

; MIPS32R6:      lw $[[PTR:[0-9]+]], %got(si)(
; MIPS32R6:      lw $2, 0($[[PTR]])

; MIPS64-EL:     lwl $[[R0:[0-9]+]], 3($[[R1:[0-9]+]])
; MIPS64-EL:     lwr $[[R0]], 0($[[R1]])

; MIPS64R2-EL:   lwl $[[R0:[0-9]+]], 3($[[R1:[0-9]+]])
; MIPS64R2-EL:   lwr $[[R0]], 0($[[R1]])

; MIPS64-EB:     lwl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; MIPS64-EB:     lwr $[[R0]], 3($[[R1]])

; MIPS64R2-EB:   lwl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; MIPS64R2-EB:   lwr $[[R0]], 3($[[R1]])

; MIPS64R6:      ld $[[PTR:[0-9]+]], %got_disp(si)(
; MIPS64R6:      lw $2, 0($[[PTR]])

  %0 = load i32, i32* getelementptr inbounds (%struct.SI, %struct.SI* @si, i32 0, i32 0), align 1
  ret i32 %0
}

define void @store_SI(i32 signext %a) nounwind {
entry:
; ALL-LABEL: store_SI:

; MIPS32-EL:     swl $[[R0:[0-9]+]], 3($[[R1:[0-9]+]])
; MIPS32-EL:     swr $[[R0]], 0($[[R1]])

; MIPS32-EB:     swl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; MIPS32-EB:     swr $[[R0]], 3($[[R1]])

; MIPS32R6:      lw $[[PTR:[0-9]+]], %got(si)(
; MIPS32R6:      sw $4, 0($[[PTR]])

; MIPS64-EL:     swl $[[R0:[0-9]+]], 3($[[R1:[0-9]+]])
; MIPS64-EL:     swr $[[R0]], 0($[[R1]])

; MIPS64R2-EL:   swl $[[R0:[0-9]+]], 3($[[R1:[0-9]+]])
; MIPS64R2-EL:   swr $[[R0]], 0($[[R1]])

; MIPS64-EB:     swl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; MIPS64-EB:     swr $[[R0]], 3($[[R1]])

; MIPS64R2-EB:   swl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; MIPS64R2-EB:   swr $[[R0]], 3($[[R1]])

; MIPS64R6:      ld $[[PTR:[0-9]+]], %got_disp(si)(
; MIPS64R6:      sw $4, 0($[[PTR]])

  store i32 %a, i32* getelementptr inbounds (%struct.SI, %struct.SI* @si, i32 0, i32 0), align 1
  ret void
}

define i64 @load_SLL() nounwind readonly {
entry:
; ALL-LABEL: load_SLL:

; MIPS32-EL:     lwl $2, 3($[[R1:[0-9]+]])
; MIPS32-EL:     lwr $2, 0($[[R1]])
; MIPS32-EL:     lwl $3, 7($[[R1:[0-9]+]])
; MIPS32-EL:     lwr $3, 4($[[R1]])

; MIPS32-EB:     lwl $2, 0($[[R1:[0-9]+]])
; MIPS32-EB:     lwr $2, 3($[[R1]])
; MIPS32-EB:     lwl $3, 4($[[R1:[0-9]+]])
; MIPS32-EB:     lwr $3, 7($[[R1]])

; MIPS32R6:      lw $[[PTR:[0-9]+]], %got(sll)(
; MIPS32R6-DAG:  lw $2, 0($[[PTR]])
; MIPS32R6-DAG:  lw $3, 4($[[PTR]])

; MIPS64-EL:     ldl $[[R0:[0-9]+]], 7($[[R1:[0-9]+]])
; MIPS64-EL:     ldr $[[R0]], 0($[[R1]])

; MIPS64R2-EL:   ldl $[[R0:[0-9]+]], 7($[[R1:[0-9]+]])
; MIPS64R2-EL:   ldr $[[R0]], 0($[[R1]])

; MIPS64-EB:     ldl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; MIPS64-EB:     ldr $[[R0]], 7($[[R1]])

; MIPS64R2-EB:   ldl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; MIPS64R2-EB:   ldr $[[R0]], 7($[[R1]])

; MIPS64R6:      ld $[[PTR:[0-9]+]], %got_disp(sll)(
; MIPS64R6:      ld $2, 0($[[PTR]])

  %0 = load i64, i64* getelementptr inbounds (%struct.SLL, %struct.SLL* @sll, i64 0, i32 0), align 1
  ret i64 %0
}

define i64 @load_SI_sext_to_i64() nounwind readonly {
entry:
; ALL-LABEL: load_SI_sext_to_i64:

; MIPS32-EL:     lwl $[[R0:[0-9]+]], 3($[[R1:[0-9]+]])
; MIPS32-EL:     lwr $[[R0]], 0($[[R1]])

; MIPS32-EB:     lwl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; MIPS32-EB:     lwr $[[R0]], 3($[[R1]])

; MIPS32R6:      lw $[[PTR:[0-9]+]], %got(si)(
; MIPS32R6-EL:   lw $2, 0($[[PTR]])
; MIPS32R6-EL:   sra $3, $2, 31
; MIPS32R6-EB:   lw $3, 0($[[PTR]])
; MIPS32R6-EB:   sra $2, $3, 31

; MIPS64-EL:     lwl $[[R0:[0-9]+]], 3($[[R1:[0-9]+]])
; MIPS64-EL:     lwr $[[R0]], 0($[[R1]])

; MIPS64R2-EL:   lwl $[[R0:[0-9]+]], 3($[[R1:[0-9]+]])
; MIPS64R2-EL:   lwr $[[R0]], 0($[[R1]])

; MIPS64-EB:     lwl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; MIPS64-EB:     lwr $[[R0]], 3($[[R1]])

; MIPS64R2-EB:   lwl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; MIPS64R2-EB:   lwr $[[R0]], 3($[[R1]])

; MIPS64R6:      ld $[[PTR:[0-9]+]], %got_disp(si)(
; MIPS64R6:      lw $2, 0($[[PTR]])

  %0 = load i32, i32* getelementptr inbounds (%struct.SI, %struct.SI* @si, i64 0, i32 0), align 1
  %conv = sext i32 %0 to i64
  ret i64 %conv
}

define i64 @load_UI() nounwind readonly {
entry:
; ALL-LABEL: load_UI:

; MIPS32-EL-DAG: lwl $[[R2:2]], 3($[[R1:[0-9]+]])
; MIPS32-EL-DAG: lwr $[[R2]],   0($[[R1]])
; MIPS32-EL-DAG: addiu $3, $zero, 0

; MIPS32-EB-DAG: lwl $[[R2:3]], 0($[[R1:[0-9]+]])
; MIPS32-EB-DAG: lwr $[[R2]],   3($[[R1]])
; MIPS32-EB-DAG: addiu $2, $zero, 0

; MIPS32R6:        lw $[[PTR:[0-9]+]], %got(sui)(
; MIPS32R6-EL-DAG: lw $2, 0($[[PTR]])
; MIPS32R6-EL-DAG: addiu $3, $zero, 0
; MIPS32R6-EB-DAG: lw $3, 0($[[PTR]])
; MIPS32R6-EB-DAG: addiu $2, $zero, 0

; MIPS64-EL-DAG: lwl $[[R0:[0-9]+]], 3($[[R1:[0-9]+]])
; MIPS64-EL-DAG: lwr $[[R0]], 0($[[R1]])
; MIPS64-EL-DAG: daddiu $[[R2:[0-9]+]], $zero, 1
; MIPS64-EL-DAG: dsll   $[[R3:[0-9]+]], $[[R2]], 32
; MIPS64-EL-DAG: daddiu $[[R4:[0-9]+]], $[[R3]], -1
; MIPS64-EL-DAG: and    ${{[0-9]+}}, $[[R0]], $[[R4]]

; MIPS64R2-EL-DAG: lwl $[[R0:[0-9]+]], 3($[[R1:[0-9]+]])
; MIPS64R2-EL-DAG: lwr $[[R0]], 0($[[R1]])
; MIPS64R2-EL-DAG: dext $[[R0]], $[[R0]], 0, 32

; MIPS64-EB:     lwl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; MIPS64-EB:     lwr $[[R0]], 3($[[R1]])

; MIPS64R2-EB:   lwl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; MIPS64R2-EB:   lwr $[[R0]], 3($[[R1]])
; MIPS64R2-EB:   dext $[[R0]], $[[R0]], 0, 32

; MIPS64R6:      ld $[[PTR:[0-9]+]], %got_disp(sui)(
; MIPS64R6:      lwu $2, 0($[[PTR]])

  %0 = load i32, i32* getelementptr inbounds (%struct.SUI, %struct.SUI* @sui, i64 0, i32 0), align 1
  %conv = zext i32 %0 to i64
  ret i64 %conv
}

define void @store_SLL(i64 %a) nounwind {
entry:
; ALL-LABEL: store_SLL:

; MIPS32-EL-DAG: swl $[[A1:4]], 3($[[R1:[0-9]+]])
; MIPS32-EL-DAG: swr $[[A1]],   0($[[R1]])
; MIPS32-EL-DAG: swl $[[A2:5]], 7($[[R1:[0-9]+]])
; MIPS32-EL-DAG: swr $[[A2]],   4($[[R1]])

; MIPS32-EB-DAG: swl $[[A1:4]], 0($[[R1:[0-9]+]])
; MIPS32-EB-DAG: swr $[[A1]],   3($[[R1]])
; MIPS32-EB-DAG: swl $[[A1:5]], 4($[[R1:[0-9]+]])
; MIPS32-EB-DAG: swr $[[A1]],   7($[[R1]])

; MIPS32R6-DAG:  lw $[[PTR:[0-9]+]], %got(sll)(
; MIPS32R6-DAG:  sw $4, 0($[[PTR]])
; MIPS32R6-DAG:  sw $5, 4($[[PTR]])

; MIPS64-EL:     sdl $[[R0:[0-9]+]], 7($[[R1:[0-9]+]])
; MIPS64-EL:     sdr $[[R0]], 0($[[R1]])

; MIPS64R2-EL:   sdl $[[R0:[0-9]+]], 7($[[R1:[0-9]+]])
; MIPS64R2-EL:   sdr $[[R0]], 0($[[R1]])

; MIPS64-EB:     sdl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; MIPS64-EB:     sdr $[[R0]], 7($[[R1]])

; MIPS64R2-EB:   sdl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; MIPS64R2-EB:   sdr $[[R0]], 7($[[R1]])

; MIPS64R6:      ld $[[PTR:[0-9]+]], %got_disp(sll)(
; MIPS64R6:      sd $4, 0($[[PTR]])

  store i64 %a, i64* getelementptr inbounds (%struct.SLL, %struct.SLL* @sll, i64 0, i32 0), align 1
  ret void
}

define void @store_SI_trunc_from_i64(i32 signext %a) nounwind {
entry:
; ALL-LABEL: store_SI_trunc_from_i64:

; MIPS32-EL:     swl $[[R0:[0-9]+]], 3($[[R1:[0-9]+]])
; MIPS32-EL:     swr $[[R0]], 0($[[R1]])

; MIPS32-EB:     swl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; MIPS32-EB:     swr $[[R0]], 3($[[R1]])

; MIPS32R6:      lw $[[PTR:[0-9]+]], %got(si)(
; MIPS32R6:      sw $4, 0($[[PTR]])

; MIPS64-EL:     swl $[[R0:[0-9]+]], 3($[[R1:[0-9]+]])
; MIPS64-EL:     swr $[[R0]], 0($[[R1]])

; MIPS64R2-EL:   swl $[[R0:[0-9]+]], 3($[[R1:[0-9]+]])
; MIPS64R2-EL:   swr $[[R0]], 0($[[R1]])

; MIPS64-EB:     swl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; MIPS64-EB:     swr $[[R0]], 3($[[R1]])

; MIPS64R2-EB:   swl $[[R0:[0-9]+]], 0($[[R1:[0-9]+]])
; MIPS64R2-EB:   swr $[[R0]], 3($[[R1]])

; MIPS64R6:      ld $[[PTR:[0-9]+]], %got_disp(si)(
; MIPS64R6:      sw $4, 0($[[PTR]])

  store i32 %a, i32* getelementptr inbounds (%struct.SI, %struct.SI* @si, i64 0, i32 0), align 1
  ret void
}

;
; Structures are simply concatenations of the members. They are unaffected by
; endianness
;

%struct.S0 = type { i8, i8 }
@struct_s0 = common global %struct.S0 zeroinitializer, align 1
%struct.S1 = type { i16, i16 }
@struct_s1 = common global %struct.S1 zeroinitializer, align 1
%struct.S2 = type { i32, i32 }
@struct_s2 = common global %struct.S2 zeroinitializer, align 1

define void @copy_struct_S0() nounwind {
entry:
; ALL-LABEL: copy_struct_S0:

; MIPS32-EL:     lw $[[PTR:[0-9]+]], %got(struct_s0)(
; MIPS32-EB:     lw $[[PTR:[0-9]+]], %got(struct_s0)(
; MIPS32R6:      lw $[[PTR:[0-9]+]], %got(struct_s0)(
; MIPS64-EL:     ld $[[PTR:[0-9]+]], %got_disp(struct_s0)(
; MIPS64R2-EL:   ld $[[PTR:[0-9]+]], %got_disp(struct_s0)(
; MIPS64-EB:     ld $[[PTR:[0-9]+]], %got_disp(struct_s0)(
; MIPS64R2-EB:   ld $[[PTR:[0-9]+]], %got_disp(struct_s0)(
; MIPS64R6:      ld $[[PTR:[0-9]+]], %got_disp(struct_s0)(

; MIPS32-DAG:       lbu $[[R1:[0-9]+]], 0($[[PTR]])
; MIPS32-DAG:       sb $[[R1]], 2($[[PTR]])
; MIPS32-DAG:       lbu $[[R2:[0-9]+]], 1($[[PTR]])
; MIPS32-DAG:       sb $[[R2]], 3($[[PTR]])

; MIPS32R6:       lhu $[[R1:[0-9]+]], 0($[[PTR]])
; MIPS32R6:       sh $[[R1]], 2($[[PTR]])

; MIPS64-DAG:       lbu $[[R1:[0-9]+]], 0($[[PTR]])
; MIPS64-DAG:       sb $[[R1]], 2($[[PTR]])
; MIPS64-DAG:       lbu $[[R2:[0-9]+]], 1($[[PTR]])
; MIPS64-DAG:       sb $[[R2]], 3($[[PTR]])

  %0 = load %struct.S0, %struct.S0* getelementptr inbounds (%struct.S0, %struct.S0* @struct_s0, i32 0), align 1
  store %struct.S0 %0, %struct.S0* getelementptr inbounds (%struct.S0, %struct.S0* @struct_s0, i32 1), align 1
  ret void
}

define void @copy_struct_S1() nounwind {
entry:
; ALL-LABEL: copy_struct_S1:

; MIPS32-EL:     lw $[[PTR:[0-9]+]], %got(struct_s1)(
; MIPS32-EB:     lw $[[PTR:[0-9]+]], %got(struct_s1)(
; MIPS32-EL-DAG:    lwl $[[R1:[0-9]+]], 3($[[PTR]])
; MIPS32-EL-DAG:    lwr $[[R1]], 0($[[PTR]])
; MIPS32-EL-DAG:    swl $[[R1]], 7($[[PTR]])
; MIPS32-EL-DAG:    swr $[[R1]], 4($[[PTR]])
; MIPS32-EB-DAG:    lwl $[[R1:[0-9]+]], 0($[[PTR]])
; MIPS32-EB-DAG:    lwr $[[R1]], 3($[[PTR]])
; MIPS32-EB-DAG:    swl $[[R1]], 4($[[PTR]])
; MIPS32-EB-DAG:    swr $[[R1]], 7($[[PTR]])

; MIPS32-NOLEFTRIGHT-DAG:    lbu $[[R1:[0-9]+]], 0($[[PTR]])
; MIPS32-NOLEFTRIGHT-DAG:    sb $[[R1]], 4($[[PTR]])
; MIPS32-NOLEFTRIGHT-DAG:    lbu $[[R1:[0-9]+]], 1($[[PTR]])
; MIPS32-NOLEFTRIGHT-DAG:    sb $[[R1]], 5($[[PTR]])
; MIPS32-NOLEFTRIGHT-DAG:    lbu $[[R1:[0-9]+]], 2($[[PTR]])
; MIPS32-NOLEFTRIGHT-DAG:    sb $[[R1]], 6($[[PTR]])
; MIPS32-NOLEFTRIGHT-DAG:    lbu $[[R1:[0-9]+]], 3($[[PTR]])
; MIPS32-NOLEFTRIGHT-DAG:    sb $[[R1]], 7($[[PTR]])

; MIPS32R6:      lw $[[PTR:[0-9]+]], %got(struct_s1)(
; MIPS32R6-DAG:  lw $[[R1:[0-9]+]], 0($[[PTR]])
; MIPS32R6-DAG:  sw $[[R1]], 4($[[PTR]])

; MIPS64-EL:     ld $[[PTR:[0-9]+]], %got_disp(struct_s1)(
; MIPS64R2-EL:   ld $[[PTR:[0-9]+]], %got_disp(struct_s1)(
; MIPS64-EB:     ld $[[PTR:[0-9]+]], %got_disp(struct_s1)(
; MIPS64R2-EB:   ld $[[PTR:[0-9]+]], %got_disp(struct_s1)(

; MIPS64-EL-DAG:    lwl $[[R1:[0-9]+]], 3($[[PTR]])
; MIPS64-EL-DAG:    lwr $[[R1]], 0($[[PTR]])
; MIPS64-EL-DAG:    swl $[[R1]], 7($[[PTR]])
; MIPS64-EL-DAG:    swr $[[R1]], 4($[[PTR]])

; MIPS64R2-EL-DAG:    lwl $[[R1:[0-9]+]], 3($[[PTR]])
; MIPS64R2-EL-DAG:    lwr $[[R1]], 0($[[PTR]])
; MIPS64R2-EL-DAG:    swl $[[R1]], 7($[[PTR]])
; MIPS64R2-EL-DAG:    swr $[[R1]], 4($[[PTR]])

; MIPS64-EB-DAG:    lwl $[[R1:[0-9]+]], 0($[[PTR]])
; MIPS64-EB-DAG:    lwr $[[R1]], 3($[[PTR]])
; MIPS64-EB-DAG:    swl $[[R1]], 4($[[PTR]])
; MIPS64-EB-DAG:    swr $[[R1]], 7($[[PTR]])

; MIPS64R2-EB-DAG:    lwl $[[R1:[0-9]+]], 0($[[PTR]])
; MIPS64R2-EB-DAG:    lwr $[[R1]], 3($[[PTR]])
; MIPS64R2-EB-DAG:    swl $[[R1]], 4($[[PTR]])
; MIPS64R2-EB-DAG:    swr $[[R1]], 7($[[PTR]])

; MIPS64-NOLEFTRIGHT-DAG:    lbu $[[R1:[0-9]+]], 0($[[PTR]])
; MIPS64-NOLEFTRIGHT-DAG:    sb $[[R1]], 4($[[PTR]])
; MIPS64-NOLEFTRIGHT-DAG:    lbu $[[R1:[0-9]+]], 1($[[PTR]])
; MIPS64-NOLEFTRIGHT-DAG:    sb $[[R1]], 5($[[PTR]])
; MIPS64-NOLEFTRIGHT-DAG:    lbu $[[R1:[0-9]+]], 2($[[PTR]])
; MIPS64-NOLEFTRIGHT-DAG:    sb $[[R1]], 6($[[PTR]])
; MIPS64-NOLEFTRIGHT-DAG:    lbu $[[R1:[0-9]+]], 3($[[PTR]])
; MIPS64-NOLEFTRIGHT-DAG:    sb $[[R1]], 7($[[PTR]])

; MIPS64R6:      ld $[[PTR:[0-9]+]], %got_disp(struct_s1)(
; MIPS64R6-DAG:  lw $[[R1:[0-9]+]], 0($[[PTR]])
; MIPS64R6-DAG:  sw $[[R1]], 4($[[PTR]])

  %0 = load %struct.S1, %struct.S1* getelementptr inbounds (%struct.S1, %struct.S1* @struct_s1, i32 0), align 1
  store %struct.S1 %0, %struct.S1* getelementptr inbounds (%struct.S1, %struct.S1* @struct_s1, i32 1), align 1
  ret void
}

define void @copy_struct_S2() nounwind {
entry:
; ALL-LABEL: copy_struct_S2:

; MIPS32-EL:     lw $[[PTR:[0-9]+]], %got(struct_s2)(
; MIPS32-EL-DAG: lwl $[[R1:[0-9]+]], 3($[[PTR]])
; MIPS32-EL-DAG: lwr $[[R1]],        0($[[PTR]])
; MIPS32-EL-DAG: swl $[[R1]],       11($[[PTR]])
; MIPS32-EL-DAG: swr $[[R1]],        8($[[PTR]])
; MIPS32-EL-DAG: lwl $[[R1:[0-9]+]], 7($[[PTR]])
; MIPS32-EL-DAG: lwr $[[R1]],        4($[[PTR]])
; MIPS32-EL-DAG: swl $[[R1]],       15($[[PTR]])
; MIPS32-EL-DAG: swr $[[R1]],       12($[[PTR]])

; MIPS32-EB:     lw $[[PTR:[0-9]+]], %got(struct_s2)(
; MIPS32-EB-DAG: lwl $[[R1:[0-9]+]], 0($[[PTR]])
; MIPS32-EB-DAG: lwr $[[R1]],        3($[[PTR]])
; MIPS32-EB-DAG: swl $[[R1]],        8($[[PTR]])
; MIPS32-EB-DAG: swr $[[R1]],       11($[[PTR]])
; MIPS32-EB-DAG: lwl $[[R1:[0-9]+]], 4($[[PTR]])
; MIPS32-EB-DAG: lwr $[[R1]],        7($[[PTR]])
; MIPS32-EB-DAG: swl $[[R1]],       12($[[PTR]])
; MIPS32-EB-DAG: swr $[[R1]],       15($[[PTR]])

; MIPS32R6:      lw $[[PTR:[0-9]+]], %got(struct_s2)(
; MIPS32R6-DAG:  lw $[[R1:[0-9]+]], 0($[[PTR]])
; MIPS32R6-DAG:  sw $[[R1]],        8($[[PTR]])
; MIPS32R6-DAG:  lw $[[R1:[0-9]+]], 4($[[PTR]])
; MIPS32R6-DAG:  sw $[[R1]],       12($[[PTR]])

; MIPS64-EL:     ld $[[PTR:[0-9]+]], %got_disp(struct_s2)(

; MIPS64-EL-DAG: ldl $[[R1:[0-9]+]], 7($[[PTR]])
; MIPS64-EL-DAG: ldr $[[R1]],        0($[[PTR]])
; MIPS64-EL-DAG: sdl $[[R1]],       15($[[PTR]])
; MIPS64-EL-DAG: sdr $[[R1]],        8($[[PTR]])

; MIPS64R2-EL:     ld $[[PTR:[0-9]+]], %got_disp(struct_s2)(

; MIPS64R2-EL-DAG: ldl $[[R1:[0-9]+]], 7($[[PTR]])
; MIPS64R2-EL-DAG: ldr $[[R1]],        0($[[PTR]])
; MIPS64R2-EL-DAG: sdl $[[R1]],       15($[[PTR]])
; MIPS64R2-EL-DAG: sdr $[[R1]],        8($[[PTR]])

; MIPS64-EB:     ld $[[PTR:[0-9]+]], %got_disp(struct_s2)(
; MIPS64-EB-DAG: ldl $[[R1:[0-9]+]], 0($[[PTR]])
; MIPS64-EB-DAG: ldr $[[R1]],        7($[[PTR]])
; MIPS64-EB-DAG: sdl $[[R1]],        8($[[PTR]])
; MIPS64-EB-DAG: sdr $[[R1]],       15($[[PTR]])

; MIPS64R2-EB:     ld $[[PTR:[0-9]+]], %got_disp(struct_s2)(
; MIPS64R2-EB-DAG: ldl $[[R1:[0-9]+]], 0($[[PTR]])
; MIPS64R2-EB-DAG: ldr $[[R1]],        7($[[PTR]])
; MIPS64R2-EB-DAG: sdl $[[R1]],        8($[[PTR]])
; MIPS64R2-EB-DAG: sdr $[[R1]],       15($[[PTR]])

; MIPS64R6:      ld $[[PTR:[0-9]+]], %got_disp(struct_s2)(
; MIPS64R6-DAG:  ld $[[R1:[0-9]+]], 0($[[PTR]])
; MIPS64R6-DAG:  sd $[[R1]],        8($[[PTR]])

  %0 = load %struct.S2, %struct.S2* getelementptr inbounds (%struct.S2, %struct.S2* @struct_s2, i32 0), align 1
  store %struct.S2 %0, %struct.S2* getelementptr inbounds (%struct.S2, %struct.S2* @struct_s2, i32 1), align 1
  ret void
}

;
; Arrays are simply concatenations of the members. They are unaffected by
; endianness
;

@arr = common global [7 x i8] zeroinitializer, align 1

define void @pass_array_byval() nounwind {
entry:
; ALL-LABEL: pass_array_byval:

; MIPS32-EL:     lw $[[SPTR:[0-9]+]], %got(arr)(
; MIPS32-EL-DAG: lwl $[[R1:4]], 3($[[PTR]])
; MIPS32-EL-DAG: lwr $[[R1]],   0($[[PTR]])
; MIPS32-EL-DAG: lbu $[[R2:[0-9]+]], 4($[[PTR]])
; MIPS32-EL-DAG: lbu $[[R3:[0-9]+]], 5($[[PTR]])
; MIPS32-EL-DAG: sll $[[T0:[0-9]+]], $[[R3]], 8
; MIPS32-EL-DAG: or  $[[T1:[0-9]+]], $[[T0]], $[[R2]]
; MIPS32-EL-DAG: lbu $[[R4:[0-9]+]], 6($[[PTR]])
; MIPS32-EL-DAG: sll $[[T2:[0-9]+]], $[[R4]], 16
; MIPS32-EL-DAG: or  $5, $[[T1]], $[[T2]]

; MIPS32-EB:     lw $[[SPTR:[0-9]+]], %got(arr)(
; MIPS32-EB-DAG: lwl $[[R1:4]], 0($[[PTR]])
; MIPS32-EB-DAG: lwr $[[R1]],   3($[[PTR]])
; MIPS32-EB-DAG: lbu $[[R2:[0-9]+]], 5($[[PTR]])
; MIPS32-EB-DAG: lbu $[[R3:[0-9]+]], 4($[[PTR]])
; MIPS32-EB-DAG: sll $[[T0:[0-9]+]], $[[R3]], 8
; MIPS32-EB-DAG: or  $[[T1:[0-9]+]], $[[T0]], $[[R2]]
; MIPS32-EB-DAG: sll $[[T1]], $[[T1]], 16
; MIPS32-EB-DAG: lbu $[[R4:[0-9]+]], 6($[[PTR]])
; MIPS32-EB-DAG: sll $[[T2:[0-9]+]], $[[R4]], 8
; MIPS32-EB-DAG: or  $5, $[[T1]], $[[T2]]

; MIPS32R6:        lw $[[SPTR:[0-9]+]], %got(arr)(
; MIPS32R6-DAG:    lw $4, 0($[[PTR]])
; MIPS32R6-EL-DAG: lhu $[[R2:[0-9]+]], 4($[[PTR]])
; MIPS32R6-EL-DAG: lbu $[[R3:[0-9]+]], 6($[[PTR]])
; MIPS32R6-EL-DAG: sll $[[T0:[0-9]+]], $[[R3]], 16
; MIPS32R6-EL-DAG: or  $5, $[[R2]], $[[T0]]

; MIPS32R6-EB-DAG: lhu $[[R2:[0-9]+]], 4($[[PTR]])
; MIPS32R6-EB-DAG: lbu $[[R3:[0-9]+]], 6($[[PTR]])
; MIPS32R6-EB-DAG: sll $[[T0:[0-9]+]], $[[R2]], 16
; MIPS32R6-EB-DAG: or  $5, $[[T0]], $[[R3]]

; MIPS64-EL:     ld $[[SPTR:[0-9]+]], %got_disp(arr)(
; MIPS64-EL-DAG: lwl $[[R1:[0-9]+]], 3($[[PTR]])
; MIPS64-EL-DAG: lwr $[[R1]],   0($[[PTR]])

; MIPS64R2-EL:     ld $[[SPTR:[0-9]+]], %got_disp(arr)(
; MIPS64R2-EL-DAG: lwl $[[R1:[0-9]+]], 3($[[PTR]])
; MIPS64R2-EL-DAG: lwr $[[R1]],   0($[[PTR]])

; MIPS64-EB: ld $[[SPTR:[0-9]+]], %got_disp(arr)(
; MIPS64-EB-DAG: lbu  $[[R2:[0-9]+]], 5($[[PTR]])
; MIPS64-EB-DAG: lbu  $[[R3:[0-9]+]], 4($[[PTR]])
; MIPS64-EB-DAG: dsll $[[T0:[0-9]+]], $[[R3]], 8
; MIPS64-EB-DAG: or   $[[T1:[0-9]+]], $[[T0]], $[[R2]]
; MIPS64-EB-DAG: lbu  $[[R4:[0-9]+]], 6($[[PTR]])
; MIPS64-EB-DAG: dsll $[[T1]], $[[T1]], 16
; MIPS64-EB-DAG: lwl  $[[R1:[0-9]+]], 0($[[PTR]])
; MIPS64-EB-DAG: lwr  $[[R1]],   3($[[PTR]])
; MIPS64-EB-DAG: dsll $[[R5:[0-9]+]], $[[R1]], 32
; MIPS64-EB-DAG: or   $[[T3:[0-9]+]], $[[R5]], $[[T1]]
; MIPS64-EB-DAG: dsll $[[T4:[0-9]+]], $[[R4]], 8
; MIPS64-EB-DAG: or   $4, $[[T3]], $[[T4]]

; MIPS64R2-EB: ld $[[SPTR:[0-9]+]], %got_disp(arr)(
; MIPS64R2-EB-DAG: lbu  $[[R1:[0-9]+]], 5($[[PTR]])
; MIPS64R2-EB-DAG: lbu  $[[R2:[0-9]+]], 4($[[PTR]])
; MIPS64R2-EB-DAG: dsll $[[T0:[0-9]+]], $[[R2]], 8
; MIPS64R2-EB-DAG: or   $[[T1:[0-9]+]], $[[T0]], $[[R1]]
; MIPS64R2-EB-DAG: dsll $[[T1]], $[[T1]], 16
; MIPS64R2-EB-DAG: lwl  $[[R3:[0-9]+]], 0($[[PTR]])
; MIPS64R2-EB-DAG: lwr  $[[R3]], 3($[[PTR]])
; MIPS64R2-EB-DAG: dext $[[R3]], $[[R3]], 0, 32
; MIPS64R2-EB-DAG: dsll $[[R3]], $[[R3]], 32
; MIPS64R2-EB-DAG: or   $[[T2:[0-9]+]], $[[R3]], $[[T1]]
; MIPS64R2-EB-DAG: lbu  $[[R4:[0-9]+]], 6($[[PTR]])
; MIPS64R2-EB-DAG: dsll $[[T3:[0-9]+]], $[[R4]], 8
; MIPS64R2-EB-DAG: or   $4, $[[T2]], $[[T3]]

; MIPS64R6:      ld $[[SPTR:[0-9]+]], %got_disp(arr)(

  tail call void @extern_func([7 x i8]* byval @arr) nounwind
  ret void
}

declare void @extern_func([7 x i8]* byval)
