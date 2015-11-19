; RUN: llc -mtriple=mips-linux-gnu -relocation-model=static < %s | FileCheck --check-prefix=ALL --check-prefix=O32 --check-prefix=O32-BE %s
; RUN: llc -mtriple=mipsel-linux-gnu -relocation-model=static < %s | FileCheck --check-prefix=ALL --check-prefix=O32 --check-prefix=O32-LE %s

; RUN-TODO: llc -mtriple=mips64-linux-gnu -relocation-model=static -target-abi o32 < %s | FileCheck --check-prefix=ALL --check-prefix=O32 %s
; RUN-TODO: llc -mtriple=mips64el-linux-gnu -relocation-model=static -target-abi o32 < %s | FileCheck --check-prefix=ALL --check-prefix=O32 %s

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=static -target-abi n32 < %s | FileCheck --check-prefix=ALL --check-prefix=N32 --check-prefix=N32-BE %s
; RUN: llc -mtriple=mips64el-linux-gnu -relocation-model=static -target-abi n32 < %s | FileCheck --check-prefix=ALL --check-prefix=N32 --check-prefix=N32-LE %s

; RUN: llc -mtriple=mips64-linux-gnu -relocation-model=static -target-abi n64 < %s | FileCheck --check-prefix=ALL --check-prefix=N64 --check-prefix=N64-BE %s
; RUN: llc -mtriple=mips64el-linux-gnu -relocation-model=static -target-abi n64 < %s | FileCheck --check-prefix=ALL --check-prefix=N64 --check-prefix=N64-LE %s

; Test struct returns for all ABI's and byte orders.

@struct_byte = global {i8} zeroinitializer
@struct_2byte = global {i8,i8} zeroinitializer
@struct_3xi16 = global {[3 x i16]} zeroinitializer
@struct_6xi32 = global {[6 x i32]} zeroinitializer
@struct_128xi16 = global {[128 x i16]} zeroinitializer

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1)

define inreg {i8} @ret_struct_i8() nounwind {
entry:
        %0 = load volatile {i8}, {i8}* @struct_byte
        ret {i8} %0
}

; ALL-LABEL: ret_struct_i8:
; O32-DAG:           lui [[R1:\$[0-9]+]], %hi(struct_byte)
; O32-DAG:           lbu $2, %lo(struct_byte)([[R1]])

; N32-LE-DAG:        lui [[R1:\$[0-9]+]], %hi(struct_byte)
; N32-LE-DAG:        lb $2, %lo(struct_byte)([[R1]])

; N32-BE-DAG:        lui [[R1:\$[0-9]+]], %hi(struct_byte)
; N32-BE-DAG:        lb [[R2:\$[0-9]+]], %lo(struct_byte)([[R1]])
; N32-BE-DAG:        dsll $2, [[R2]], 56

; N64-LE-DAG:        ld  [[R1:\$[0-9]+]], %got_disp(struct_byte)($1)
; N64-LE-DAG:        lb $2, 0([[R1]])

; N64-BE-DAG:        ld  [[R1:\$[0-9]+]], %got_disp(struct_byte)($1)
; N64-BE-DAG:        lb [[R2:\$[0-9]+]], 0([[R1]])
; N64-BE-DAG:        dsll $2, [[R2]], 56

; This test is based on the way clang currently lowers {i8,i8} to {i16}.
; FIXME: It should probably work for without any lowering too but this doesn't
;        work as expected. Each member gets mapped to a register rather than
;        packed into a single register.
define inreg {i16} @ret_struct_i16() nounwind {
entry:
        %retval = alloca {i8,i8}, align 1
        %0 = bitcast {i8,i8}* %retval to i8*
        call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* getelementptr inbounds ({i8,i8}, {i8,i8}* @struct_2byte, i32 0, i32 0), i64 2, i1 false)
        %1 = bitcast {i8,i8}* %retval to {i16}*
        %2 = load volatile {i16}, {i16}* %1
        ret {i16} %2
}

; ALL-LABEL: ret_struct_i16:
; O32-DAG:           lui [[R1:\$[0-9]+]], %hi(struct_2byte)
; O32-DAG:           lhu [[R2:\$[0-9]+]], %lo(struct_2byte)([[R1]])
; O32-DAG:           sh  [[R2]], 0([[SP:\$sp]])
; O32-DAG:           lhu $2, 0([[SP:\$sp]])

; N32-LE-DAG:        lui [[R1:\$[0-9]+]], %hi(struct_2byte)
; N32-LE-DAG:        lhu [[R2:\$[0-9]+]], %lo(struct_2byte)([[R1]])
; N32-LE-DAG:        sh  [[R2]], 8([[SP:\$sp]])
; N32-LE-DAG:        lh  $2, 8([[SP:\$sp]])

; N32-BE-DAG:        lui [[R1:\$[0-9]+]], %hi(struct_2byte)
; N32-BE-DAG:        lhu [[R2:\$[0-9]+]], %lo(struct_2byte)([[R1]])
; N32-BE-DAG:        sh  [[R2]], 8([[SP:\$sp]])
; N32-BE-DAG:        lh  [[R3:\$[0-9]+]], 8([[SP:\$sp]])
; N32-BE-DAG:        dsll $2, [[R3]], 48

; N64-LE-DAG:        ld  [[R1:\$[0-9]+]], %got_disp(struct_2byte)($1)
; N64-LE-DAG:        lhu [[R2:\$[0-9]+]], 0([[R1]])
; N64-LE-DAG:        sh  [[R2]], 8([[SP:\$sp]])
; N64-LE-DAG:        lh  $2, 8([[SP:\$sp]])

; N64-BE-DAG:        ld  [[R1:\$[0-9]+]], %got_disp(struct_2byte)($1)
; N64-BE-DAG:        lhu [[R2:\$[0-9]+]], 0([[R1]])
; N64-BE-DAG:        sh  [[R2]], 8([[SP:\$sp]])
; N64-BE-DAG:        lh  [[R3:\$[0-9]+]], 8([[SP:\$sp]])
; N64-BE-DAG:        dsll $2, [[R3]], 48

; Ensure that structures bigger than 32-bits but smaller than 64-bits are
; also returned in the upper bits on big endian targets. Previously, these were
; missed by the CCPromoteToType and the shift didn't happen.
define inreg {i48} @ret_struct_3xi16() nounwind {
entry:
        %0 = load volatile i48, i48* bitcast ({[3 x i16]}* @struct_3xi16 to i48*), align 2
        %1 = insertvalue {i48} undef, i48 %0, 0
        ret {i48} %1
}

; ALL-LABEL: ret_struct_3xi16:

; O32-BE-DAG:        lui [[PTR_HI:\$[0-9]+]], %hi(struct_3xi16)
; O32-BE-DAG:        addiu [[PTR_LO:\$[0-9]+]], [[PTR_HI]], %lo(struct_3xi16)
; O32-BE-DAG:        lhu [[R1:\$[0-9]+]], 4([[PTR_LO]])
; O32-BE-DAG:        lw [[R2:\$[0-9]+]], %lo(struct_3xi16)([[PTR_HI]])
; O32-BE-DAG:        sll [[R3:\$[0-9]+]], [[R2]], 16
; O32-BE-DAG:        or  $3, [[R1]], [[R3]]
; O32-BE-DAG:        srl $2, [[R2]], 16

; O32-LE-DAG:        lui [[PTR_HI:\$[0-9]+]], %hi(struct_3xi16)
; O32-LE-DAG:        addiu [[PTR_LO:\$[0-9]+]], [[PTR_HI]], %lo(struct_3xi16)
; O32-LE-DAG:        lhu $3, 4([[PTR_LO]])
; O32-LE-DAG:        lw $2, %lo(struct_3xi16)([[PTR_HI]])

; N32-LE-DAG:        lui [[PTR_HI:\$[0-9]+]], %hi(struct_3xi16)
; N32-LE-DAG:        addiu [[PTR_LO:\$[0-9]+]], [[PTR_HI]], %lo(struct_3xi16)
; N32-LE-DAG:        lh [[R1:\$[0-9]+]], 4([[PTR_LO]])
; N32-LE-DAG:        lwu [[R2:\$[0-9]+]], %lo(struct_3xi16)([[PTR_HI]])
; N32-LE-DAG:        dsll [[R3:\$[0-9]+]], [[R1]], 32
; N32-LE-DAG:        or $2, [[R2]], [[R3]]

; N32-BE-DAG:        lui [[PTR_HI:\$[0-9]+]], %hi(struct_3xi16)
; N32-BE-DAG:        addiu [[PTR_LO:\$[0-9]+]], [[PTR_HI]], %lo(struct_3xi16)
; N32-BE-DAG:        lw [[R1:\$[0-9]+]], %lo(struct_3xi16)([[PTR_HI]])
; N32-BE-DAG:        dsll [[R2:\$[0-9]+]], [[R1]], 16
; N32-BE-DAG:        lhu [[R3:\$[0-9]+]], 4([[PTR_LO]])
; N32-BE-DAG:        or [[R4:\$[0-9]+]], [[R3]], [[R2]]
; N32-BE-DAG:        dsll $2, [[R4]], 16

; N64-LE-DAG:        ld  [[PTR:\$[0-9]+]], %got_disp(struct_3xi16)($1)
; N64-LE-DAG:        lh [[R1:\$[0-9]+]], 4([[PTR]])
; N64-LE-DAG:        lwu [[R2:\$[0-9]+]], 0([[PTR]])
; N64-LE-DAG:        dsll [[R3:\$[0-9]+]], [[R1]], 32
; N64-LE-DAG:        or $2, [[R2]], [[R3]]

; N64-BE-DAG:        ld  [[PTR:\$[0-9]+]], %got_disp(struct_3xi16)($1)
; N64-BE-DAG:        lw [[R1:\$[0-9]+]], 0([[PTR]])
; N64-BE-DAG:        dsll [[R2:\$[0-9]+]], [[R1]], 16
; N64-BE-DAG:        lhu [[R3:\$[0-9]+]], 4([[PTR]])
; N64-BE-DAG:        or [[R4:\$[0-9]+]], [[R3]], [[R2]]
; N32-BE-DAG:        dsll $2, [[R4]], 16

; Ensure that large structures (>128-bit) are returned indirectly.
; We pick an extremely large structure so we don't have to match inlined memcpy's.
define void @ret_struct_128xi16({[128 x i16]}* sret %returnval) {
entry:
        %0 = bitcast {[128 x i16]}* %returnval to i8*
        call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* bitcast ({[128 x i16]}* @struct_128xi16 to i8*), i64 256, i1 false)
        ret void
}

; ALL-LABEL: ret_struct_128xi16:

; sret pointer is already in $4
; O32-DAG:        lui [[PTR:\$[0-9]+]], %hi(struct_128xi16)
; O32-DAG:        addiu $5, [[PTR]], %lo(struct_128xi16)
; O32:            jal memcpy

; sret pointer is already in $4
; N32-DAG:        lui [[PTR_HI:\$[0-9]+]], %hi(struct_128xi16)
; N32-DAG:        addiu [[PTR:\$[0-9]+]], [[PTR_HI]], %lo(struct_128xi16)
; FIXME: This signext isn't necessary. Like integers, pointers are
;        but unlike integers, pointers cannot have the signext attribute.
; N32-DAG:        sll $5, [[PTR]], 0
; N32:            jal memcpy

; sret pointer is already in $4
; N64-DAG:        ld $5, %got_disp(struct_128xi16)(
; N64-DAG:        ld $25, %call16(memcpy)(
; N64:            jalr $25

; Ensure that large structures (>128-bit) are returned indirectly.
; This will generate inlined memcpy's anyway so pick the smallest large
; structure
; This time we let the backend lower the sret argument.
define {[6 x i32]} @ret_struct_6xi32() {
entry:
        %0 = load volatile {[6 x i32]}, {[6 x i32]}* @struct_6xi32, align 2
        ret {[6 x i32]} %0
}

; ALL-LABEL: ret_struct_6xi32:

; sret pointer is already in $4
; O32-DAG:        lui [[PTR_HI:\$[0-9]+]], %hi(struct_6xi32)
; O32-DAG:        addiu [[PTR:\$[0-9]+]], [[PTR_HI]], %lo(struct_6xi32)
; O32-DAG:        lw [[T0:\$[0-9]+]], %lo(struct_6xi32)([[PTR]])
; O32-DAG:        lw [[T1:\$[0-9]+]], 4([[PTR]])
; O32-DAG:        lw [[T2:\$[0-9]+]], 8([[PTR]])
; O32-DAG:        lw [[T3:\$[0-9]+]], 12([[PTR]])
; O32-DAG:        lw [[T4:\$[0-9]+]], 16([[PTR]])
; O32-DAG:        lw [[T5:\$[0-9]+]], 20([[PTR]])
; O32-DAG:        sw [[T0]], 0($4)
; O32-DAG:        sw [[T1]], 4($4)
; O32-DAG:        sw [[T2]], 8($4)
; O32-DAG:        sw [[T3]], 12($4)
; O32-DAG:        sw [[T4]], 16($4)
; O32-DAG:        sw [[T5]], 20($4)

; FIXME: This signext isn't necessary. Like integers, pointers are
;        but unlike integers, pointers cannot have the signext attribute.
;        In this case we don't have anywhere to put the signext either since
;        the sret argument is invented by the backend.
; N32-DAG:        sll [[RET_PTR:\$[0-9]+]], $4, 0
; N32-DAG:        lui [[PTR_HI:\$[0-9]+]], %hi(struct_6xi32)
; N32-DAG:        addiu [[PTR:\$[0-9]+]], [[PTR_HI]], %lo(struct_6xi32)
; N32-DAG:        lw [[T0:\$[0-9]+]], %lo(struct_6xi32)([[PTR]])
; N32-DAG:        lw [[T1:\$[0-9]+]], 4([[PTR]])
; N32-DAG:        lw [[T2:\$[0-9]+]], 8([[PTR]])
; N32-DAG:        lw [[T3:\$[0-9]+]], 12([[PTR]])
; N32-DAG:        lw [[T4:\$[0-9]+]], 16([[PTR]])
; N32-DAG:        lw [[T5:\$[0-9]+]], 20([[PTR]])
; N32-DAG:        sw [[T0]], 0([[RET_PTR]])
; N32-DAG:        sw [[T1]], 4([[RET_PTR]])
; N32-DAG:        sw [[T2]], 8([[RET_PTR]])
; N32-DAG:        sw [[T3]], 12([[RET_PTR]])
; N32-DAG:        sw [[T4]], 16([[RET_PTR]])
; N32-DAG:        sw [[T5]], 20([[RET_PTR]])

; sret pointer is already in $4
; N64-DAG:        ld [[PTR:\$[0-9]+]], %got_disp(struct_6xi32)(
; N64-DAG:        lw [[T0:\$[0-9]+]], 0([[PTR]])
; N64-DAG:        lw [[T1:\$[0-9]+]], 4([[PTR]])
; N64-DAG:        lw [[T2:\$[0-9]+]], 8([[PTR]])
; N64-DAG:        lw [[T3:\$[0-9]+]], 12([[PTR]])
; N64-DAG:        lw [[T4:\$[0-9]+]], 16([[PTR]])
; N64-DAG:        lw [[T5:\$[0-9]+]], 20([[PTR]])
; N64-DAG:        sw [[T0]], 0($4)
; N64-DAG:        sw [[T1]], 4($4)
; N64-DAG:        sw [[T2]], 8($4)
; N64-DAG:        sw [[T3]], 12($4)
; N64-DAG:        sw [[T4]], 16($4)
; N64-DAG:        sw [[T5]], 20($4)
