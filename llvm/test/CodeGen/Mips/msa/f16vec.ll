; RUN: llc -mtriple=mipsel-unknown-linux-gnu -mcpu=mips32r5 -mattr=+fp64,+msa < %s | FileCheck %s --check-prefix=MIPS32
; RUN: llc -mtriple=mips64el-unknown-linux-gnuabi64 -mcpu=mips64r5 -mattr=+fp64,+msa < %s | FileCheck %s --check-prefix=MIPS64

; Test that f16 vectors can be built from scalar f16 values without crashing.
; This is a regression test for a crash in soft-promotion of BUILD_VECTOR
; operands.

define <8 x half> @build_v8f16(half %a, half %b, half %c, half %d, half %e, half %f, half %g, half %h) nounwind {
; MIPS32-LABEL: build_v8f16:
; MIPS32:       # %bb.0:
; MIPS32-NEXT:    insert.h $w0[0], $5
; MIPS32-NEXT:    insert.h $w0[1], $6
; MIPS32-NEXT:    insert.h $w0[2], $7
; MIPS32-NEXT:    lhu $1, 16($sp)
; MIPS32-NEXT:    insert.h $w0[3], $1
; MIPS32-NEXT:    lhu $1, 20($sp)
; MIPS32-NEXT:    insert.h $w0[4], $1
; MIPS32-NEXT:    lhu $1, 24($sp)
; MIPS32-NEXT:    insert.h $w0[5], $1
; MIPS32-NEXT:    lhu $1, 28($sp)
; MIPS32-NEXT:    insert.h $w0[6], $1
; MIPS32-NEXT:    lhu $1, 32($sp)
; MIPS32-NEXT:    insert.h $w0[7], $1
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    st.h $w0, 0($4)
;
; MIPS64-LABEL: build_v8f16:
; MIPS64:       # %bb.0:
; MIPS64-NEXT:    sll $1, $5, 0
; MIPS64-NEXT:    sll $2, $4, 0
; MIPS64-NEXT:    insert.h $w0[0], $2
; MIPS64-NEXT:    insert.h $w0[1], $1
; MIPS64-NEXT:    sll $1, $6, 0
; MIPS64-NEXT:    insert.h $w0[2], $1
; MIPS64-NEXT:    sll $1, $7, 0
; MIPS64-NEXT:    insert.h $w0[3], $1
; MIPS64-NEXT:    sll $1, $8, 0
; MIPS64-NEXT:    insert.h $w0[4], $1
; MIPS64-NEXT:    sll $1, $9, 0
; MIPS64-NEXT:    insert.h $w0[5], $1
; MIPS64-NEXT:    sll $1, $11, 0
; MIPS64-NEXT:    sll $2, $10, 0
; MIPS64-NEXT:    insert.h $w0[6], $2
; MIPS64-NEXT:    insert.h $w0[7], $1
; MIPS64-NEXT:    copy_s.d $2, $w0[0]
; MIPS64-NEXT:    jr $ra
; MIPS64-NEXT:    copy_s.d $3, $w0[1]
  %v0 = insertelement <8 x half> poison, half %a, i32 0
  %v1 = insertelement <8 x half> %v0, half %b, i32 1
  %v2 = insertelement <8 x half> %v1, half %c, i32 2
  %v3 = insertelement <8 x half> %v2, half %d, i32 3
  %v4 = insertelement <8 x half> %v3, half %e, i32 4
  %v5 = insertelement <8 x half> %v4, half %f, i32 5
  %v6 = insertelement <8 x half> %v5, half %g, i32 6
  %v7 = insertelement <8 x half> %v6, half %h, i32 7
  ret <8 x half> %v7
}

define <4 x half> @build_v4f16(half %a, half %b, half %c, half %d) nounwind {
; MIPS32-LABEL: build_v4f16:
; MIPS32:       # %bb.0:
; MIPS32-NEXT:    addiu $sp, $sp, -32
; MIPS32-NEXT:    sw $ra, 28($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    sw $fp, 24($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    move $fp, $sp
; MIPS32-NEXT:    addiu $1, $zero, -16
; MIPS32-NEXT:    and $sp, $sp, $1
; MIPS32-NEXT:    lhu $1, 48($fp)
; MIPS32-NEXT:    sh $1, 6($sp)
; MIPS32-NEXT:    sh $7, 4($sp)
; MIPS32-NEXT:    sh $6, 2($sp)
; MIPS32-NEXT:    sh $5, 0($sp)
; MIPS32-NEXT:    ld.w $w0, 0($sp)
; MIPS32-NEXT:    copy_s.w $1, $w0[0]
; MIPS32-NEXT:    copy_s.w $2, $w0[1]
; MIPS32-NEXT:    sw $2, 4($4)
; MIPS32-NEXT:    sw $1, 0($4)
; MIPS32-NEXT:    move $sp, $fp
; MIPS32-NEXT:    lw $fp, 24($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $ra, 28($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    addiu $sp, $sp, 32
;
; MIPS64-LABEL: build_v4f16:
; MIPS64:       # %bb.0:
; MIPS64-NEXT:    daddiu $sp, $sp, -16
; MIPS64-NEXT:    sh $7, 6($sp)
; MIPS64-NEXT:    sh $6, 4($sp)
; MIPS64-NEXT:    sh $5, 2($sp)
; MIPS64-NEXT:    sh $4, 0($sp)
; MIPS64-NEXT:    ld.d $w0, 0($sp)
; MIPS64-NEXT:    copy_s.d $2, $w0[0]
; MIPS64-NEXT:    jr $ra
; MIPS64-NEXT:    daddiu $sp, $sp, 16
  %v0 = insertelement <4 x half> poison, half %a, i32 0
  %v1 = insertelement <4 x half> %v0, half %b, i32 1
  %v2 = insertelement <4 x half> %v1, half %c, i32 2
  %v3 = insertelement <4 x half> %v2, half %d, i32 3
  ret <4 x half> %v3
}

define <2 x half> @build_v2f16(half %a, half %b) nounwind {
; MIPS32-LABEL: build_v2f16:
; MIPS32:       # %bb.0:
; MIPS32-NEXT:    addiu $sp, $sp, -32
; MIPS32-NEXT:    sw $ra, 28($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    sw $fp, 24($sp) # 4-byte Folded Spill
; MIPS32-NEXT:    move $fp, $sp
; MIPS32-NEXT:    addiu $1, $zero, -16
; MIPS32-NEXT:    and $sp, $sp, $1
; MIPS32-NEXT:    sh $6, 2($sp)
; MIPS32-NEXT:    sh $5, 0($sp)
; MIPS32-NEXT:    ld.w $w0, 0($sp)
; MIPS32-NEXT:    copy_s.w $1, $w0[0]
; MIPS32-NEXT:    sw $1, 0($4)
; MIPS32-NEXT:    move $sp, $fp
; MIPS32-NEXT:    lw $fp, 24($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    lw $ra, 28($sp) # 4-byte Folded Reload
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    addiu $sp, $sp, 32
;
; MIPS64-LABEL: build_v2f16:
; MIPS64:       # %bb.0:
; MIPS64-NEXT:    daddiu $sp, $sp, -16
; MIPS64-NEXT:    sh $6, 2($sp)
; MIPS64-NEXT:    sh $5, 0($sp)
; MIPS64-NEXT:    ld.w $w0, 0($sp)
; MIPS64-NEXT:    copy_s.w $1, $w0[0]
; MIPS64-NEXT:    sw $1, 0($4)
; MIPS64-NEXT:    jr $ra
; MIPS64-NEXT:    daddiu $sp, $sp, 16
  %v0 = insertelement <2 x half> poison, half %a, i32 0
  %v1 = insertelement <2 x half> %v0, half %b, i32 1
  ret <2 x half> %v1
}
