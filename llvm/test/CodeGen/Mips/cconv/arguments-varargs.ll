; RUN: llc -mtriple=mips-linux -relocation-model=static < %s | FileCheck --check-prefix=ALL --check-prefix=O32 --check-prefix=O32-BE %s
; RUN: llc -mtriple=mipsel-linux -relocation-model=static < %s | FileCheck --check-prefix=ALL --check-prefix=O32 --check-prefix=O32-LE %s

; RUN-TODO: llc -march=mips64 -relocation-model=static -mattr=-n64,+o32 < %s | FileCheck --check-prefix=ALL --check-prefix=O32 %s
; RUN-TODO: llc -march=mips64el -relocation-model=static -mattr=-n64,+o32 < %s | FileCheck --check-prefix=ALL --check-prefix=O32 %s

; RUN: llc -mtriple=mips64-linux -relocation-model=static -mattr=-n64,+n32 < %s | FileCheck --check-prefix=ALL --check-prefix=NEW --check-prefix=N32 --check-prefix=NEW-BE %s
; RUN: llc -mtriple=mips64el-linux -relocation-model=static -mattr=-n64,+n32 < %s | FileCheck --check-prefix=ALL --check-prefix=NEW --check-prefix=N32 --check-prefix=NEW-LE %s

; RUN: llc -march=mips64 -relocation-model=static -mattr=-n64,+n64 < %s | FileCheck --check-prefix=ALL --check-prefix=NEW --check-prefix=N64 --check-prefix=NEW-BE %s
; RUN: llc -march=mips64el -relocation-model=static -mattr=-n64,+n64 < %s | FileCheck --check-prefix=ALL --check-prefix=NEW --check-prefix=N64 --check-prefix=NEW-LE %s

@hwords = global [3 x i16] zeroinitializer, align 1
@words  = global [3 x i32] zeroinitializer, align 1
@dwords = global [3 x i64] zeroinitializer, align 1

define void @fn_i16_dotdotdot_i16(i16 %a, ...) {
entry:
; ALL-LABEL: fn_i16_dotdotdot_i16:

; Set up the stack with an 8-byte local area. N32/N64 must also make room for
; the argument save area (56 bytes).
; O32:           addiu  [[SP:\$sp]], $sp, -8
; N32:           addiu  [[SP:\$sp]], $sp, -64
; N64:           daddiu  [[SP:\$sp]], $sp, -64

; Save variable argument portion on the stack
; O32-DAG:       sw $7, 20([[SP]])
; O32-DAG:       sw $6, 16([[SP]])
; O32-DAG:       sw $5, 12([[SP]])

; NEW-DAG:       sd $11, 56([[SP]])
; NEW-DAG:       sd $10, 48([[SP]])
; NEW-DAG:       sd $9, 40([[SP]])
; NEW-DAG:       sd $8, 32([[SP]])
; NEW-DAG:       sd $7, 24([[SP]])
; NEW-DAG:       sd $6, 16([[SP]])
; NEW-DAG:       sd $5, 8([[SP]])

; Initialize variable argument pointer.
; For O32, the offset is 12 due to the 4 bytes used to store local variables,
; 4 bytes padding to maintain stack alignment, and the 4 byte slot for the first
; fixed argument.
; For N32/N64, it is only 8 since the fixed arguments do not reserve stack
; space.
; O32-DAG:       addiu [[VA:\$[0-9]+]], [[SP]], 12
; O32-DAG:       sw [[VA]], 0([[SP]])

; N32-DAG:       addiu [[VA:\$[0-9]+]], [[SP]], 8
; N32-DAG:       sw [[VA]], 0([[SP]])

; N64-DAG:       daddiu [[VA:\$[0-9]+]], [[SP]], 8
; N64-DAG:       sd [[VA]], 0([[SP]])

; Store [[VA]]
; O32-DAG:       sw [[VA]], 0([[SP]])

; ALL: # ANCHOR1

; Increment [[VA]]
; O32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])

; N32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; N32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 8
; N32-DAG:       sw [[VA2]], 0([[SP]])

; N64-DAG:       ld [[VA:\$[0-9]+]], 0([[SP]])
; N64-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 8
; N64-DAG:       sd [[VA2]], 0([[SP]])

; Load the first argument from the variable portion.
; This has used the stack pointer directly rather than the [[VA]] we just set
; up.
; Big-endian mode for N32/N64 must add an additional 4 to the offset due to byte
; order.
; O32-DAG:       lw [[ARG1:\$[0-9]+]], 0([[VA]])

; NEW-LE-DAG:    lw [[ARG1:\$[0-9]+]], 0([[VA]])
; NEW-BE-DAG:    lw [[ARG1:\$[0-9]+]], 4([[VA]])

; Copy the arg to the global
; O32-DAG:       addiu [[GV:\$[0-9]+]], ${{[0-9]+}}, %lo(hwords)

; N32-DAG:       addiu [[GV:\$[0-9]+]], ${{[0-9]+}}, %lo(hwords)

; N64-DAG:       ld [[GV:\$[0-9]+]], %got_disp(hwords)(

; ALL-DAG:       sh [[ARG1]], 2([[GV]])

; ALL: # ANCHOR2

; Increment [[VA]] again.
; O32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])

; N32-DAG:       lw [[VA2:\$[0-9]+]], 0([[SP]])
; N32-DAG:       addiu [[VA3:\$[0-9]+]], [[VA2]], 8
; N32-DAG:       sw [[VA3]], 0([[SP]])

; N64-DAG:       ld [[VA2:\$[0-9]+]], 0([[SP]])
; N64-DAG:       daddiu [[VA3:\$[0-9]+]], [[VA2]], 8
; N64-DAG:       sd [[VA3]], 0([[SP]])

; Load the second argument from the variable portion.
; O32-DAG:       lw [[ARG2:\$[0-9]+]], 0([[VA]])

; NEW-LE-DAG:    lw [[ARG2:\$[0-9]+]], 0([[VA2]])
; NEW-BE-DAG:    lw [[ARG2:\$[0-9]+]], 4([[VA2]])

; Copy the arg to the global
; ALL-DAG:       sh [[ARG2]], 4([[GV]])

  %ap = alloca i8*, align 8
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap2)

  call void asm sideeffect "# ANCHOR1", ""()
  %arg1 = va_arg i8** %ap, i16
  %e1 = getelementptr [3 x i16]* @hwords, i32 0, i32 1
  store volatile i16 %arg1, i16* %e1, align 2

  call void asm sideeffect "# ANCHOR2", ""()
  %arg2 = va_arg i8** %ap, i16
  %e2 = getelementptr [3 x i16]* @hwords, i32 0, i32 2
  store volatile i16 %arg2, i16* %e2, align 2

  call void @llvm.va_end(i8* %ap2)

  ret void
}

define void @fn_i16_dotdotdot_i32(i16 %a, ...) {
entry:
; ALL-LABEL: fn_i16_dotdotdot_i32:

; Set up the stack with an 8-byte local area. N32/N64 must also make room for
; the argument save area (56 bytes).
; O32:           addiu  [[SP:\$sp]], $sp, -8
; N32:           addiu  [[SP:\$sp]], $sp, -64
; N64:           daddiu  [[SP:\$sp]], $sp, -64

; Save variable argument portion on the stack
; O32-DAG:       sw $7, 20([[SP]])
; O32-DAG:       sw $6, 16([[SP]])
; O32-DAG:       sw $5, 12([[SP]])

; NEW-DAG:       sd $11, 56([[SP]])
; NEW-DAG:       sd $10, 48([[SP]])
; NEW-DAG:       sd $9, 40([[SP]])
; NEW-DAG:       sd $8, 32([[SP]])
; NEW-DAG:       sd $7, 24([[SP]])
; NEW-DAG:       sd $6, 16([[SP]])
; NEW-DAG:       sd $5, 8([[SP]])

; Initialize variable argument pointer.
; For O32, the offset is 12 due to the 4 bytes used to store local variables,
; 4 bytes padding to maintain stack alignment, and the 4 byte slot for the first
; fixed argument.
; For N32/N64, it is only 8 since the fixed arguments do not reserve stack
; space.
; O32-DAG:       addiu [[VA:\$[0-9]+]], [[SP]], 12
; O32-DAG:       sw [[VA]], 0([[SP]])

; N32-DAG:       addiu [[VA:\$[0-9]+]], [[SP]], 8
; N32-DAG:       sw [[VA]], 0([[SP]])

; N64-DAG:       daddiu [[VA:\$[0-9]+]], [[SP]], 8
; N64-DAG:       sd [[VA]], 0([[SP]])

; Store [[VA]]
; O32-DAG:       sw [[VA]], 0([[SP]])

; ALL: # ANCHOR1

; Increment [[VA]]
; O32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])

; N32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; N32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 8
; N32-DAG:       sw [[VA2]], 0([[SP]])

; N64-DAG:       ld [[VA:\$[0-9]+]], 0([[SP]])
; N64-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 8
; N64-DAG:       sd [[VA2]], 0([[SP]])

; Load the first argument from the variable portion.
; This has used the stack pointer directly rather than the [[VA]] we just set
; up.
; Big-endian mode for N32/N64 must add an additional 4 to the offset due to byte
; order.
; O32-DAG:       lw [[ARG1:\$[0-9]+]], 0([[VA]])

; NEW-LE-DAG:    lw [[ARG1:\$[0-9]+]], 0([[VA]])
; NEW-BE-DAG:    lw [[ARG1:\$[0-9]+]], 4([[VA]])

; Copy the arg to the global
; O32-DAG:       addiu [[GV:\$[0-9]+]], ${{[0-9]+}}, %lo(words)

; N32-DAG:       addiu [[GV:\$[0-9]+]], ${{[0-9]+}}, %lo(words)

; N64-DAG:       ld [[GV:\$[0-9]+]], %got_disp(words)(

; ALL-DAG:       sw [[ARG1]], 4([[GV]])

; ALL: # ANCHOR2

; Increment [[VA]] again.
; O32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])

; N32-DAG:       lw [[VA2:\$[0-9]+]], 0([[SP]])
; N32-DAG:       addiu [[VA3:\$[0-9]+]], [[VA2]], 8
; N32-DAG:       sw [[VA3]], 0([[SP]])

; N64-DAG:       ld [[VA2:\$[0-9]+]], 0([[SP]])
; N64-DAG:       daddiu [[VA3:\$[0-9]+]], [[VA2]], 8
; N64-DAG:       sd [[VA3]], 0([[SP]])

; Load the second argument from the variable portion.
; O32-DAG:       lw [[ARG2:\$[0-9]+]], 0([[VA]])

; NEW-LE-DAG:    lw [[ARG2:\$[0-9]+]], 0([[VA2]])
; NEW-BE-DAG:    lw [[ARG2:\$[0-9]+]], 4([[VA2]])

; Copy the arg to the global
; ALL-DAG:       sw [[ARG2]], 8([[GV]])

  %ap = alloca i8*, align 8
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap2)

  call void asm sideeffect "# ANCHOR1", ""()
  %arg1 = va_arg i8** %ap, i32
  %e1 = getelementptr [3 x i32]* @words, i32 0, i32 1
  store volatile i32 %arg1, i32* %e1, align 4

  call void asm sideeffect "# ANCHOR2", ""()
  %arg2 = va_arg i8** %ap, i32
  %e2 = getelementptr [3 x i32]* @words, i32 0, i32 2
  store volatile i32 %arg2, i32* %e2, align 4

  call void @llvm.va_end(i8* %ap2)

  ret void
}

define void @fn_i16_dotdotdot_i64(i16 %a, ...) {
entry:
; ALL-LABEL: fn_i16_dotdotdot_i64:

; Set up the stack with an 8-byte local area. N32/N64 must also make room for
; the argument save area (56 bytes).
; O32:           addiu  [[SP:\$sp]], $sp, -8
; N32:           addiu  [[SP:\$sp]], $sp, -64
; N64:           daddiu  [[SP:\$sp]], $sp, -64

; Save variable argument portion on the stack
; O32-DAG:       sw $7, 20([[SP]])
; O32-DAG:       sw $6, 16([[SP]])
; O32-DAG:       sw $5, 12([[SP]])

; NEW-DAG:       sd $11, 56([[SP]])
; NEW-DAG:       sd $10, 48([[SP]])
; NEW-DAG:       sd $9, 40([[SP]])
; NEW-DAG:       sd $8, 32([[SP]])
; NEW-DAG:       sd $7, 24([[SP]])
; NEW-DAG:       sd $6, 16([[SP]])
; NEW-DAG:       sd $5, 8([[SP]])

; Initialize variable argument pointer.
; For O32, the offset is 12 due to the 4 bytes used to store local variables,
; 4 bytes padding to maintain stack alignment, and the 4 byte slot for the first
; fixed argument.
; For N32/N64, it is only 8 since the fixed arguments do not reserve stack
; space.
; O32-DAG:       addiu [[VA:\$[0-9]+]], [[SP]], 12
; O32-DAG:       sw [[VA]], 0([[SP]])

; N32-DAG:       addiu [[VA:\$[0-9]+]], [[SP]], 8
; N32-DAG:       sw [[VA]], 0([[SP]])

; N64-DAG:       daddiu [[VA:\$[0-9]+]], [[SP]], 8
; N64-DAG:       sd [[VA]], 0([[SP]])

; Store [[VA]]
; O32-DAG:       sw [[VA]], 0([[SP]])

; ALL: # ANCHOR1

; Increment [[VA]] (and realign pointer for O32)
; O32:           lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA_TMP0:\$[0-9]+]], [[VA]], 7
; O32-DAG:       addiu [[VA_TMP1:\$[0-9]+]], $zero, -8
; O32-DAG:       and   [[VA_TMP2:\$[0-9]+]], [[VA_TMP0]], [[VA_TMP1]]
; O32-DAG:       ori   [[VA2:\$[0-9]+]], [[VA_TMP2]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])

; N32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; N32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 8
; N32-DAG:       sw [[VA2]], 0([[SP]])

; N64-DAG:       ld [[VA:\$[0-9]+]], 0([[SP]])
; N64-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 8
; N64-DAG:       sd [[VA2]], 0([[SP]])

; Load the first argument from the variable portion and copy it to the global.
; This has used the stack pointer directly rather than the [[VA]] we just set
; up.
; Big-endian mode for N32/N64 must add an additional 4 to the offset due to byte
; order.
; O32-DAG:       addiu [[GV:\$[0-9]+]], ${{[0-9]+}}, %lo(dwords)
; O32-DAG:       lw [[ARG1:\$[0-9]+]], 0([[VA]])
; O32-DAG:       sw [[ARG1]], 8([[GV]])
; O32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])
; O32-DAG:       lw [[ARG1:\$[0-9]+]], 0([[VA]])
; O32-DAG:       sw [[ARG1]], 12([[GV]])

; N32-DAG:       addiu [[GV:\$[0-9]+]], ${{[0-9]+}}, %lo(dwords)
; N64-DAG:       ld [[GV:\$[0-9]+]], %got_disp(dwords)(
; NEW-DAG:       ld [[ARG1:\$[0-9]+]], 0([[VA]])
; NEW-DAG:       sd [[ARG1]], 8([[GV]])

; ALL: # ANCHOR2

; Increment [[VA]] again.
; FIXME: We're still aligned from the last one but CodeGen doesn't spot that.
; O32:           lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA_TMP0:\$[0-9]+]], [[VA]], 7
; O32-DAG:       and   [[VA_TMP2:\$[0-9]+]], [[VA_TMP0]], [[VA_TMP1]]
; O32-DAG:       ori   [[VA2:\$[0-9]+]], [[VA_TMP2]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])

; N32-DAG:       lw [[VA2:\$[0-9]+]], 0([[SP]])
; N32-DAG:       addiu [[VA3:\$[0-9]+]], [[VA2]], 8
; N32-DAG:       sw [[VA3]], 0([[SP]])

; N64-DAG:       ld [[VA2:\$[0-9]+]], 0([[SP]])
; N64-DAG:       daddiu [[VA3:\$[0-9]+]], [[VA2]], 8
; N64-DAG:       sd [[VA3]], 0([[SP]])

; Load the second argument from the variable portion and copy it to the global.
; O32-DAG:       lw [[ARG2:\$[0-9]+]], 0([[VA]])
; O32-DAG:       sw [[ARG2]], 16([[GV]])
; O32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])
; O32-DAG:       lw [[ARG2:\$[0-9]+]], 0([[VA]])
; O32-DAG:       sw [[ARG2]], 20([[GV]])

; NEW-DAG:       ld [[ARG2:\$[0-9]+]], 0([[VA2]])
; NEW-DAG:       sd [[ARG2]], 16([[GV]])

  %ap = alloca i8*, align 8
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap2)

  call void asm sideeffect "# ANCHOR1", ""()
  %arg1 = va_arg i8** %ap, i64
  %e1 = getelementptr [3 x i64]* @dwords, i32 0, i32 1
  store volatile i64 %arg1, i64* %e1, align 8

  call void asm sideeffect "# ANCHOR2", ""()
  %arg2 = va_arg i8** %ap, i64
  %e2 = getelementptr [3 x i64]* @dwords, i32 0, i32 2
  store volatile i64 %arg2, i64* %e2, align 8

  call void @llvm.va_end(i8* %ap2)

  ret void
}

define void @fn_i32_dotdotdot_i16(i32 %a, ...) {
entry:
; ALL-LABEL: fn_i32_dotdotdot_i16:

; Set up the stack with an 8-byte local area. N32/N64 must also make room for
; the argument save area (56 bytes).
; O32:           addiu  [[SP:\$sp]], $sp, -8
; N32:           addiu  [[SP:\$sp]], $sp, -64
; N64:           daddiu [[SP:\$sp]], $sp, -64

; Save variable argument portion on the stack
; O32-DAG:       sw $7, 20([[SP]])
; O32-DAG:       sw $6, 16([[SP]])
; O32-DAG:       sw $5, 12([[SP]])

; NEW-DAG:       sd $11, 56([[SP]])
; NEW-DAG:       sd $10, 48([[SP]])
; NEW-DAG:       sd $9, 40([[SP]])
; NEW-DAG:       sd $8, 32([[SP]])
; NEW-DAG:       sd $7, 24([[SP]])
; NEW-DAG:       sd $6, 16([[SP]])
; NEW-DAG:       sd $5, 8([[SP]])

; Initialize variable argument pointer.
; For O32, the offset is 12 due to the 4 bytes used to store local variables,
; 4 bytes padding to maintain stack alignment, and the 4 byte slot for the first
; fixed argument.
; For N32/N64, it is only 8 since the fixed arguments do not reserve stack
; space.
; O32-DAG:       addiu [[VA:\$[0-9]+]], [[SP]], 12
; O32-DAG:       sw [[VA]], 0([[SP]])

; N32-DAG:       addiu [[VA:\$[0-9]+]], [[SP]], 8
; N32-DAG:       sw [[VA]], 0([[SP]])

; N64-DAG:       daddiu [[VA:\$[0-9]+]], [[SP]], 8
; N64-DAG:       sd [[VA]], 0([[SP]])

; Store [[VA]]
; O32-DAG:       sw [[VA]], 0([[SP]])

; ALL: # ANCHOR1

; Increment [[VA]]
; O32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])

; N32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; N32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 8
; N32-DAG:       sw [[VA2]], 0([[SP]])

; N64-DAG:       ld [[VA:\$[0-9]+]], 0([[SP]])
; N64-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 8
; N64-DAG:       sd [[VA2]], 0([[SP]])

; Load the first argument from the variable portion.
; This has used the stack pointer directly rather than the [[VA]] we just set
; up.
; Big-endian mode for N32/N64 must add an additional 4 to the offset due to byte
; order.
; O32-DAG:       lw [[ARG1:\$[0-9]+]], 0([[VA]])

; NEW-LE-DAG:    lw [[ARG1:\$[0-9]+]], 0([[VA]])
; NEW-BE-DAG:    lw [[ARG1:\$[0-9]+]], 4([[VA]])

; Copy the arg to the global
; O32-DAG:       addiu [[GV:\$[0-9]+]], ${{[0-9]+}}, %lo(hwords)

; N32-DAG:       addiu [[GV:\$[0-9]+]], ${{[0-9]+}}, %lo(hwords)

; N64-DAG:       ld [[GV:\$[0-9]+]], %got_disp(hwords)(

; ALL-DAG:       sh [[ARG1]], 2([[GV]])

; ALL: # ANCHOR2

; Increment [[VA]] again.
; O32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])

; N32-DAG:       lw [[VA2:\$[0-9]+]], 0([[SP]])
; N32-DAG:       addiu [[VA3:\$[0-9]+]], [[VA2]], 8
; N32-DAG:       sw [[VA3]], 0([[SP]])

; N64-DAG:       ld [[VA2:\$[0-9]+]], 0([[SP]])
; N64-DAG:       daddiu [[VA3:\$[0-9]+]], [[VA2]], 8
; N64-DAG:       sd [[VA3]], 0([[SP]])

; Load the second argument from the variable portion.
; O32-DAG:       lw [[ARG2:\$[0-9]+]], 0([[VA]])

; NEW-LE-DAG:    lw [[ARG2:\$[0-9]+]], 0([[VA2]])
; NEW-BE-DAG:    lw [[ARG2:\$[0-9]+]], 4([[VA2]])

; Copy the arg to the global
; ALL-DAG:       sh [[ARG2]], 4([[GV]])

  %ap = alloca i8*, align 8
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap2)

  call void asm sideeffect "# ANCHOR1", ""()
  %arg1 = va_arg i8** %ap, i16
  %e1 = getelementptr [3 x i16]* @hwords, i32 0, i32 1
  store volatile i16 %arg1, i16* %e1, align 2

  call void asm sideeffect "# ANCHOR2", ""()
  %arg2 = va_arg i8** %ap, i16
  %e2 = getelementptr [3 x i16]* @hwords, i32 0, i32 2
  store volatile i16 %arg2, i16* %e2, align 2

  call void @llvm.va_end(i8* %ap2)

  ret void
}

define void @fn_i32_dotdotdot_i32(i32 %a, ...) {
entry:
; ALL-LABEL: fn_i32_dotdotdot_i32:

; Set up the stack with an 8-byte local area. N32/N64 must also make room for
; the argument save area (56 bytes).
; O32:           addiu  [[SP:\$sp]], $sp, -8
; N32:           addiu  [[SP:\$sp]], $sp, -64
; N64:           daddiu  [[SP:\$sp]], $sp, -64

; Save variable argument portion on the stack
; O32-DAG:       sw $7, 20([[SP]])
; O32-DAG:       sw $6, 16([[SP]])
; O32-DAG:       sw $5, 12([[SP]])

; NEW-DAG:       sd $11, 56([[SP]])
; NEW-DAG:       sd $10, 48([[SP]])
; NEW-DAG:       sd $9, 40([[SP]])
; NEW-DAG:       sd $8, 32([[SP]])
; NEW-DAG:       sd $7, 24([[SP]])
; NEW-DAG:       sd $6, 16([[SP]])
; NEW-DAG:       sd $5, 8([[SP]])

; Initialize variable argument pointer.
; For O32, the offset is 12 due to the 4 bytes used to store local variables,
; 4 bytes padding to maintain stack alignment, and the 4 byte slot for the first
; fixed argument.
; For N32/N64, it is only 8 since the fixed arguments do not reserve stack
; space.
; O32-DAG:       addiu [[VA:\$[0-9]+]], [[SP]], 12
; O32-DAG:       sw [[VA]], 0([[SP]])

; N32-DAG:       addiu [[VA:\$[0-9]+]], [[SP]], 8
; N32-DAG:       sw [[VA]], 0([[SP]])

; N64-DAG:       daddiu [[VA:\$[0-9]+]], [[SP]], 8
; N64-DAG:       sd [[VA]], 0([[SP]])

; Store [[VA]]
; O32-DAG:       sw [[VA]], 0([[SP]])

; ALL: # ANCHOR1

; Increment [[VA]]
; O32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])

; N32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; N32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 8
; N32-DAG:       sw [[VA2]], 0([[SP]])

; N64-DAG:       ld [[VA:\$[0-9]+]], 0([[SP]])
; N64-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 8
; N64-DAG:       sd [[VA2]], 0([[SP]])

; Load the first argument from the variable portion.
; This has used the stack pointer directly rather than the [[VA]] we just set
; up.
; Big-endian mode for N32/N64 must add an additional 4 to the offset due to byte
; order.
; O32-DAG:       lw [[ARG1:\$[0-9]+]], 0([[VA]])

; NEW-LE-DAG:    lw [[ARG1:\$[0-9]+]], 0([[VA]])
; NEW-BE-DAG:    lw [[ARG1:\$[0-9]+]], 4([[VA]])

; Copy the arg to the global
; O32-DAG:       addiu [[GV:\$[0-9]+]], ${{[0-9]+}}, %lo(words)

; N32-DAG:       addiu [[GV:\$[0-9]+]], ${{[0-9]+}}, %lo(words)

; N64-DAG:       ld [[GV:\$[0-9]+]], %got_disp(words)(

; ALL-DAG:       sw [[ARG1]], 4([[GV]])

; ALL: # ANCHOR2

; Increment [[VA]] again.
; O32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])

; N32-DAG:       lw [[VA2:\$[0-9]+]], 0([[SP]])
; N32-DAG:       addiu [[VA3:\$[0-9]+]], [[VA2]], 8
; N32-DAG:       sw [[VA3]], 0([[SP]])

; N64-DAG:       ld [[VA2:\$[0-9]+]], 0([[SP]])
; N64-DAG:       daddiu [[VA3:\$[0-9]+]], [[VA2]], 8
; N64-DAG:       sd [[VA3]], 0([[SP]])

; Load the second argument from the variable portion.
; O32-DAG:       lw [[ARG2:\$[0-9]+]], 0([[VA]])

; NEW-LE-DAG:    lw [[ARG2:\$[0-9]+]], 0([[VA2]])
; NEW-BE-DAG:    lw [[ARG2:\$[0-9]+]], 4([[VA2]])

; Copy the arg to the global
; ALL-DAG:       sw [[ARG2]], 8([[GV]])

  %ap = alloca i8*, align 8
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap2)

  call void asm sideeffect "# ANCHOR1", ""()
  %arg1 = va_arg i8** %ap, i32
  %e1 = getelementptr [3 x i32]* @words, i32 0, i32 1
  store volatile i32 %arg1, i32* %e1, align 4

  call void asm sideeffect "# ANCHOR2", ""()
  %arg2 = va_arg i8** %ap, i32
  %e2 = getelementptr [3 x i32]* @words, i32 0, i32 2
  store volatile i32 %arg2, i32* %e2, align 4

  call void @llvm.va_end(i8* %ap2)

  ret void
}

define void @fn_i32_dotdotdot_i64(i32 %a, ...) {
entry:
; ALL-LABEL: fn_i32_dotdotdot_i64:

; Set up the stack with an 8-byte local area. N32/N64 must also make room for
; the argument save area (56 bytes).
; O32:           addiu  [[SP:\$sp]], $sp, -8
; N32:           addiu  [[SP:\$sp]], $sp, -64
; N64:           daddiu  [[SP:\$sp]], $sp, -64

; Save variable argument portion on the stack
; O32-DAG:       sw $7, 20([[SP]])
; O32-DAG:       sw $6, 16([[SP]])
; O32-DAG:       sw $5, 12([[SP]])

; NEW-DAG:       sd $11, 56([[SP]])
; NEW-DAG:       sd $10, 48([[SP]])
; NEW-DAG:       sd $9, 40([[SP]])
; NEW-DAG:       sd $8, 32([[SP]])
; NEW-DAG:       sd $7, 24([[SP]])
; NEW-DAG:       sd $6, 16([[SP]])
; NEW-DAG:       sd $5, 8([[SP]])

; Initialize variable argument pointer.
; For O32, the offset is 12 due to the 4 bytes used to store local variables,
; 4 bytes padding to maintain stack alignment, and the 4 byte slot for the first
; fixed argument.
; For N32/N64, it is only 8 since the fixed arguments do not reserve stack
; space.
; O32-DAG:       addiu [[VA:\$[0-9]+]], [[SP]], 12
; O32-DAG:       sw [[VA]], 0([[SP]])

; N32-DAG:       addiu [[VA:\$[0-9]+]], [[SP]], 8
; N32-DAG:       sw [[VA]], 0([[SP]])

; N64-DAG:       daddiu [[VA:\$[0-9]+]], [[SP]], 8
; N64-DAG:       sd [[VA]], 0([[SP]])

; Store [[VA]]
; O32-DAG:       sw [[VA]], 0([[SP]])

; ALL: # ANCHOR1

; Increment [[VA]] (and realign pointer for O32)
; O32:           lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA_TMP0:\$[0-9]+]], [[VA]], 7
; O32-DAG:       addiu [[VA_TMP1:\$[0-9]+]], $zero, -8
; O32-DAG:       and   [[VA_TMP2:\$[0-9]+]], [[VA_TMP0]], [[VA_TMP1]]
; O32-DAG:       ori   [[VA2:\$[0-9]+]], [[VA_TMP2]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])

; N32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; N32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 8
; N32-DAG:       sw [[VA2]], 0([[SP]])

; N64-DAG:       ld [[VA:\$[0-9]+]], 0([[SP]])
; N64-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 8
; N64-DAG:       sd [[VA2]], 0([[SP]])

; Load the first argument from the variable portion and copy it to the global.
; This has used the stack pointer directly rather than the [[VA]] we just set
; up.
; Big-endian mode for N32/N64 must add an additional 4 to the offset due to byte
; order.
; O32-DAG:       addiu [[GV:\$[0-9]+]], ${{[0-9]+}}, %lo(dwords)
; O32-DAG:       lw [[ARG1:\$[0-9]+]], 0([[VA]])
; O32-DAG:       sw [[ARG1]], 8([[GV]])
; O32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])
; O32-DAG:       lw [[ARG1:\$[0-9]+]], 0([[VA]])
; O32-DAG:       sw [[ARG1]], 12([[GV]])

; N32-DAG:       addiu [[GV:\$[0-9]+]], ${{[0-9]+}}, %lo(dwords)
; N64-DAG:       ld [[GV:\$[0-9]+]], %got_disp(dwords)(
; NEW-DAG:       ld [[ARG1:\$[0-9]+]], 0([[VA]])
; NEW-DAG:       sd [[ARG1]], 8([[GV]])

; ALL: # ANCHOR2

; Increment [[VA]] again.
; FIXME: We're still aligned from the last one but CodeGen doesn't spot that.
; O32:           lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA_TMP0:\$[0-9]+]], [[VA]], 7
; O32-DAG:       and   [[VA_TMP2:\$[0-9]+]], [[VA_TMP0]], [[VA_TMP1]]
; O32-DAG:       ori   [[VA2:\$[0-9]+]], [[VA_TMP2]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])

; N32-DAG:       lw [[VA2:\$[0-9]+]], 0([[SP]])
; N32-DAG:       addiu [[VA3:\$[0-9]+]], [[VA2]], 8
; N32-DAG:       sw [[VA3]], 0([[SP]])

; N64-DAG:       ld [[VA2:\$[0-9]+]], 0([[SP]])
; N64-DAG:       daddiu [[VA3:\$[0-9]+]], [[VA2]], 8
; N64-DAG:       sd [[VA3]], 0([[SP]])

; Load the second argument from the variable portion and copy it to the global.
; O32-DAG:       lw [[ARG2:\$[0-9]+]], 0([[VA]])
; O32-DAG:       sw [[ARG2]], 16([[GV]])
; O32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])
; O32-DAG:       lw [[ARG2:\$[0-9]+]], 0([[VA]])
; O32-DAG:       sw [[ARG2]], 20([[GV]])

; NEW-DAG:       ld [[ARG2:\$[0-9]+]], 0([[VA2]])
; NEW-DAG:       sd [[ARG2]], 16([[GV]])

  %ap = alloca i8*, align 8
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap2)

  call void asm sideeffect "# ANCHOR1", ""()
  %arg1 = va_arg i8** %ap, i64
  %e1 = getelementptr [3 x i64]* @dwords, i32 0, i32 1
  store volatile i64 %arg1, i64* %e1, align 8

  call void asm sideeffect "# ANCHOR2", ""()
  %arg2 = va_arg i8** %ap, i64
  %e2 = getelementptr [3 x i64]* @dwords, i32 0, i32 2
  store volatile i64 %arg2, i64* %e2, align 8

  call void @llvm.va_end(i8* %ap2)

  ret void
}

define void @fn_i64_dotdotdot_i16(i64 %a, ...) {
entry:
; ALL-LABEL: fn_i64_dotdotdot_i16:

; Set up the stack with an 8-byte local area. N32/N64 must also make room for
; the argument save area (56 bytes).
; O32:           addiu  [[SP:\$sp]], $sp, -8
; N32:           addiu  [[SP:\$sp]], $sp, -64
; N64:           daddiu [[SP:\$sp]], $sp, -64

; Save variable argument portion on the stack
; O32-DAG:       sw $7, 20([[SP]])
; O32-DAG:       sw $6, 16([[SP]])

; NEW-DAG:       sd $11, 56([[SP]])
; NEW-DAG:       sd $10, 48([[SP]])
; NEW-DAG:       sd $9, 40([[SP]])
; NEW-DAG:       sd $8, 32([[SP]])
; NEW-DAG:       sd $7, 24([[SP]])
; NEW-DAG:       sd $6, 16([[SP]])
; NEW-DAG:       sd $5, 8([[SP]])

; Initialize variable argument pointer.
; For O32, the offset is 16 due to the 4 bytes used to store local variables,
; 4 bytes padding to maintain stack alignment, and the two 4 byte slots for the
; first fixed argument.
; For N32/N64, it is only 8 since the fixed arguments do not reserve stack
; space.
; O32-DAG:       addiu [[VA:\$[0-9]+]], [[SP]], 16
; O32-DAG:       sw [[VA]], 0([[SP]])

; N32-DAG:       addiu [[VA:\$[0-9]+]], [[SP]], 8
; N32-DAG:       sw [[VA]], 0([[SP]])

; N64-DAG:       daddiu [[VA:\$[0-9]+]], [[SP]], 8
; N64-DAG:       sd [[VA]], 0([[SP]])

; Store [[VA]]
; O32-DAG:       sw [[VA]], 0([[SP]])

; ALL: # ANCHOR1

; Increment [[VA]]
; O32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])

; N32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; N32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 8
; N32-DAG:       sw [[VA2]], 0([[SP]])

; N64-DAG:       ld [[VA:\$[0-9]+]], 0([[SP]])
; N64-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 8
; N64-DAG:       sd [[VA2]], 0([[SP]])

; Load the first argument from the variable portion.
; This has used the stack pointer directly rather than the [[VA]] we just set
; up.
; Big-endian mode for N32/N64 must add an additional 4 to the offset due to byte
; order.
; O32-DAG:       lw [[ARG1:\$[0-9]+]], 0([[VA]])

; NEW-LE-DAG:    lw [[ARG1:\$[0-9]+]], 0([[VA]])
; NEW-BE-DAG:    lw [[ARG1:\$[0-9]+]], 4([[VA]])

; Copy the arg to the global
; O32-DAG:       addiu [[GV:\$[0-9]+]], ${{[0-9]+}}, %lo(hwords)

; N32-DAG:       addiu [[GV:\$[0-9]+]], ${{[0-9]+}}, %lo(hwords)

; N64-DAG:       ld [[GV:\$[0-9]+]], %got_disp(hwords)(

; ALL-DAG:       sh [[ARG1]], 2([[GV]])

; ALL: # ANCHOR2

; Increment [[VA]] again.
; O32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])

; N32-DAG:       lw [[VA2:\$[0-9]+]], 0([[SP]])
; N32-DAG:       addiu [[VA3:\$[0-9]+]], [[VA2]], 8
; N32-DAG:       sw [[VA3]], 0([[SP]])

; N64-DAG:       ld [[VA2:\$[0-9]+]], 0([[SP]])
; N64-DAG:       daddiu [[VA3:\$[0-9]+]], [[VA2]], 8
; N64-DAG:       sd [[VA3]], 0([[SP]])

; Load the second argument from the variable portion.
; O32-DAG:       lw [[ARG2:\$[0-9]+]], 0([[VA]])

; NEW-LE-DAG:    lw [[ARG2:\$[0-9]+]], 0([[VA2]])
; NEW-BE-DAG:    lw [[ARG2:\$[0-9]+]], 4([[VA2]])

; Copy the arg to the global
; ALL-DAG:       sh [[ARG2]], 4([[GV]])

  %ap = alloca i8*, align 8
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap2)

  call void asm sideeffect "# ANCHOR1", ""()
  %arg1 = va_arg i8** %ap, i16
  %e1 = getelementptr [3 x i16]* @hwords, i32 0, i32 1
  store volatile i16 %arg1, i16* %e1, align 2

  call void asm sideeffect "# ANCHOR2", ""()
  %arg2 = va_arg i8** %ap, i16
  %e2 = getelementptr [3 x i16]* @hwords, i32 0, i32 2
  store volatile i16 %arg2, i16* %e2, align 2

  call void @llvm.va_end(i8* %ap2)

  ret void
}

define void @fn_i64_dotdotdot_i32(i64 %a, ...) {
entry:
; ALL-LABEL: fn_i64_dotdotdot_i32:

; Set up the stack with an 8-byte local area. N32/N64 must also make room for
; the argument save area (56 bytes).
; O32:           addiu  [[SP:\$sp]], $sp, -8
; N32:           addiu  [[SP:\$sp]], $sp, -64
; N64:           daddiu  [[SP:\$sp]], $sp, -64

; Save variable argument portion on the stack
; O32-DAG:       sw $7, 20([[SP]])
; O32-DAG:       sw $6, 16([[SP]])

; NEW-DAG:       sd $11, 56([[SP]])
; NEW-DAG:       sd $10, 48([[SP]])
; NEW-DAG:       sd $9, 40([[SP]])
; NEW-DAG:       sd $8, 32([[SP]])
; NEW-DAG:       sd $7, 24([[SP]])
; NEW-DAG:       sd $6, 16([[SP]])
; NEW-DAG:       sd $5, 8([[SP]])

; Initialize variable argument pointer.
; For O32, the offset is 16 due to the 4 bytes used to store local variables,
; 4 bytes padding to maintain stack alignment, and the two 4 byte slots for the
; first fixed argument.
; For N32/N64, it is only 8 since the fixed arguments do not reserve stack
; space.
; O32-DAG:       addiu [[VA:\$[0-9]+]], [[SP]], 16
; O32-DAG:       sw [[VA]], 0([[SP]])

; N32-DAG:       addiu [[VA:\$[0-9]+]], [[SP]], 8
; N32-DAG:       sw [[VA]], 0([[SP]])

; N64-DAG:       daddiu [[VA:\$[0-9]+]], [[SP]], 8
; N64-DAG:       sd [[VA]], 0([[SP]])

; Store [[VA]]
; O32-DAG:       sw [[VA]], 0([[SP]])

; ALL: # ANCHOR1

; Increment [[VA]]
; O32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])

; N32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; N32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 8
; N32-DAG:       sw [[VA2]], 0([[SP]])

; N64-DAG:       ld [[VA:\$[0-9]+]], 0([[SP]])
; N64-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 8
; N64-DAG:       sd [[VA2]], 0([[SP]])

; Load the first argument from the variable portion.
; This has used the stack pointer directly rather than the [[VA]] we just set
; up.
; Big-endian mode for N32/N64 must add an additional 4 to the offset due to byte
; order.
; O32-DAG:       lw [[ARG1:\$[0-9]+]], 0([[VA]])

; NEW-LE-DAG:    lw [[ARG1:\$[0-9]+]], 0([[VA]])
; NEW-BE-DAG:    lw [[ARG1:\$[0-9]+]], 4([[VA]])

; Copy the arg to the global
; O32-DAG:       addiu [[GV:\$[0-9]+]], ${{[0-9]+}}, %lo(words)

; N32-DAG:       addiu [[GV:\$[0-9]+]], ${{[0-9]+}}, %lo(words)

; N64-DAG:       ld [[GV:\$[0-9]+]], %got_disp(words)(

; ALL-DAG:       sw [[ARG1]], 4([[GV]])

; ALL: # ANCHOR2

; Increment [[VA]] again.
; O32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])

; N32-DAG:       lw [[VA2:\$[0-9]+]], 0([[SP]])
; N32-DAG:       addiu [[VA3:\$[0-9]+]], [[VA2]], 8
; N32-DAG:       sw [[VA3]], 0([[SP]])

; N64-DAG:       ld [[VA2:\$[0-9]+]], 0([[SP]])
; N64-DAG:       daddiu [[VA3:\$[0-9]+]], [[VA2]], 8
; N64-DAG:       sd [[VA3]], 0([[SP]])

; Load the second argument from the variable portion.
; O32-DAG:       lw [[ARG2:\$[0-9]+]], 0([[VA]])

; NEW-LE-DAG:    lw [[ARG2:\$[0-9]+]], 0([[VA2]])
; NEW-BE-DAG:    lw [[ARG2:\$[0-9]+]], 4([[VA2]])

; Copy the arg to the global
; ALL-DAG:       sw [[ARG2]], 8([[GV]])

  %ap = alloca i8*, align 8
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap2)

  call void asm sideeffect "# ANCHOR1", ""()
  %arg1 = va_arg i8** %ap, i32
  %e1 = getelementptr [3 x i32]* @words, i32 0, i32 1
  store volatile i32 %arg1, i32* %e1, align 4

  call void asm sideeffect "# ANCHOR2", ""()
  %arg2 = va_arg i8** %ap, i32
  %e2 = getelementptr [3 x i32]* @words, i32 0, i32 2
  store volatile i32 %arg2, i32* %e2, align 4

  call void @llvm.va_end(i8* %ap2)

  ret void
}

define void @fn_i64_dotdotdot_i64(i64 %a, ...) {
entry:
; ALL-LABEL: fn_i64_dotdotdot_i64:

; Set up the stack with an 8-byte local area. N32/N64 must also make room for
; the argument save area (56 bytes).
; O32:           addiu  [[SP:\$sp]], $sp, -8
; N32:           addiu  [[SP:\$sp]], $sp, -64
; N64:           daddiu  [[SP:\$sp]], $sp, -64

; Save variable argument portion on the stack
; O32-DAG:       sw $7, 20([[SP]])
; O32-DAG:       sw $6, 16([[SP]])

; NEW-DAG:       sd $11, 56([[SP]])
; NEW-DAG:       sd $10, 48([[SP]])
; NEW-DAG:       sd $9, 40([[SP]])
; NEW-DAG:       sd $8, 32([[SP]])
; NEW-DAG:       sd $7, 24([[SP]])
; NEW-DAG:       sd $6, 16([[SP]])
; NEW-DAG:       sd $5, 8([[SP]])

; Initialize variable argument pointer.
; For O32, the offset is 16 due to the 4 bytes used to store local variables,
; 4 bytes padding to maintain stack alignment, and the two 4 byte slots for the
; first fixed argument.
; For N32/N64, it is only 8 since the fixed arguments do not reserve stack
; space.
; O32-DAG:       addiu [[VA:\$[0-9]+]], [[SP]], 16
; O32-DAG:       sw [[VA]], 0([[SP]])

; N32-DAG:       addiu [[VA:\$[0-9]+]], [[SP]], 8
; N32-DAG:       sw [[VA]], 0([[SP]])

; N64-DAG:       daddiu [[VA:\$[0-9]+]], [[SP]], 8
; N64-DAG:       sd [[VA]], 0([[SP]])

; Store [[VA]]
; O32-DAG:       sw [[VA]], 0([[SP]])

; ALL: # ANCHOR1

; Increment [[VA]] (and realign pointer for O32)
; O32:           lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA_TMP0:\$[0-9]+]], [[VA]], 7
; O32-DAG:       addiu [[VA_TMP1:\$[0-9]+]], $zero, -8
; O32-DAG:       and   [[VA_TMP2:\$[0-9]+]], [[VA_TMP0]], [[VA_TMP1]]
; O32-DAG:       ori   [[VA2:\$[0-9]+]], [[VA_TMP2]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])

; N32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; N32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 8
; N32-DAG:       sw [[VA2]], 0([[SP]])

; N64-DAG:       ld [[VA:\$[0-9]+]], 0([[SP]])
; N64-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 8
; N64-DAG:       sd [[VA2]], 0([[SP]])

; Load the first argument from the variable portion and copy it to the global.
; This has used the stack pointer directly rather than the [[VA]] we just set
; up.
; Big-endian mode for N32/N64 must add an additional 4 to the offset due to byte
; order.
; O32-DAG:       addiu [[GV:\$[0-9]+]], ${{[0-9]+}}, %lo(dwords)
; O32-DAG:       lw [[ARG1:\$[0-9]+]], 0([[VA]])
; O32-DAG:       sw [[ARG1]], 8([[GV]])
; O32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])
; O32-DAG:       lw [[ARG1:\$[0-9]+]], 0([[VA]])
; O32-DAG:       sw [[ARG1]], 12([[GV]])

; N32-DAG:       addiu [[GV:\$[0-9]+]], ${{[0-9]+}}, %lo(dwords)
; N64-DAG:       ld [[GV:\$[0-9]+]], %got_disp(dwords)(
; NEW-DAG:       ld [[ARG1:\$[0-9]+]], 0([[VA]])
; NEW-DAG:       sd [[ARG1]], 8([[GV]])

; ALL: # ANCHOR2

; Increment [[VA]] again.
; FIXME: We're still aligned from the last one but CodeGen doesn't spot that.
; O32:           lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA_TMP0:\$[0-9]+]], [[VA]], 7
; O32-DAG:       and   [[VA_TMP2:\$[0-9]+]], [[VA_TMP0]], [[VA_TMP1]]
; O32-DAG:       ori   [[VA2:\$[0-9]+]], [[VA_TMP2]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])

; N32-DAG:       lw [[VA2:\$[0-9]+]], 0([[SP]])
; N32-DAG:       addiu [[VA3:\$[0-9]+]], [[VA2]], 8
; N32-DAG:       sw [[VA3]], 0([[SP]])

; N64-DAG:       ld [[VA2:\$[0-9]+]], 0([[SP]])
; N64-DAG:       daddiu [[VA3:\$[0-9]+]], [[VA2]], 8
; N64-DAG:       sd [[VA3]], 0([[SP]])

; Load the second argument from the variable portion and copy it to the global.
; O32-DAG:       lw [[ARG2:\$[0-9]+]], 0([[VA]])
; O32-DAG:       sw [[ARG2]], 16([[GV]])
; O32-DAG:       lw [[VA:\$[0-9]+]], 0([[SP]])
; O32-DAG:       addiu [[VA2:\$[0-9]+]], [[VA]], 4
; O32-DAG:       sw [[VA2]], 0([[SP]])
; O32-DAG:       lw [[ARG2:\$[0-9]+]], 0([[VA]])
; O32-DAG:       sw [[ARG2]], 20([[GV]])

; NEW-DAG:       ld [[ARG2:\$[0-9]+]], 0([[VA2]])
; NEW-DAG:       sd [[ARG2]], 16([[GV]])

  %ap = alloca i8*, align 8
  %ap2 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap2)

  call void asm sideeffect "# ANCHOR1", ""()
  %arg1 = va_arg i8** %ap, i64
  %e1 = getelementptr [3 x i64]* @dwords, i32 0, i32 1
  store volatile i64 %arg1, i64* %e1, align 8

  call void asm sideeffect "# ANCHOR2", ""()
  %arg2 = va_arg i8** %ap, i64
  %e2 = getelementptr [3 x i64]* @dwords, i32 0, i32 2
  store volatile i64 %arg2, i64* %e2, align 8

  call void @llvm.va_end(i8* %ap2)

  ret void
}

declare void @llvm.va_start(i8*)
declare void @llvm.va_end(i8*)
