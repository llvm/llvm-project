; RUN: llc -relocation-model=pic -march=mipsel -mcpu=mips32r5 \
; RUN:     -mattr=+fp64,+msa -verify-machineinstrs < %s | FileCheck %s \
; RUN:     --check-prefixes=ALL,MIPS32,MIPSR5,MIPS32-O32,MIPS32R5-O32
; RUN: llc -relocation-model=pic -march=mips64el -mcpu=mips64r5 \
; RUN:     -mattr=+fp64,+msa -verify-machineinstrs -target-abi n32 < %s | FileCheck %s \
; RUN:     --check-prefixes=ALL,MIPS64,MIPSR5,MIPS64-N32,MIPS64R5-N32
; RUN: llc -relocation-model=pic -march=mips64el -mcpu=mips64r5 \
; RUN:     -mattr=+fp64,+msa -verify-machineinstrs -target-abi n64 < %s | FileCheck %s \
; RUN:     --check-prefixes=ALL,MIPS64,MIPSR5,MIPS64-N64,MIPS64R5-N64

; RUN: llc -relocation-model=pic -march=mipsel -mcpu=mips32r6 \
; RUN:     -mattr=+fp64,+msa -verify-machineinstrs < %s | FileCheck %s \
; RUN:     --check-prefixes=ALL,MIPS32,MIPSR6,MIPSR6-O32
; RUN: llc -relocation-model=pic -march=mips64el -mcpu=mips64r6 \
; RUN:     -mattr=+fp64,+msa -verify-machineinstrs -target-abi n32 < %s | FileCheck %s \
; RUN:     --check-prefixes=ALL,MIPS64,MIPSR6,MIPS64-N32,MIPSR6-N32
; RUN: llc -relocation-model=pic -march=mips64el -mcpu=mips64r6 \
; RUN:     -mattr=+fp64,+msa -verify-machineinstrs -target-abi n64 < %s | FileCheck %s \
; RUN:     --check-prefixes=ALL,MIPS64,MIPSR6,MIPS64-N64,MIPSR6-N64


; Check the use of frame indexes in the msa pseudo f16 instructions.

@k = external global float

declare float @k2(half *)

define void @f3(i16 %b) {
entry:
; ALL-LABEL: f3:

; ALL: sh $4, [[O0:[0-9]+]]($sp)
; ALL-DAG: jalr $25
; MIPS32-DAG: addiu $4, $sp, [[O0]]
; MIPS64-N32: addiu $4, $sp, [[O0]]
; MIPS64-N64: daddiu $4, $sp, [[O0]]
; ALL: swc1 $f0

  %0 = alloca half
  %1 = bitcast i16 %b to half
  store half %1, half * %0
  %2 = call float @k2(half * %0)
  store float %2, float * @k
  ret void
}

define void  @f(i16 %b) {
; ALL-LABEL: f:

; ALL: sh $4, [[O0:[0-9]+]]($sp)
; ALL: lh $[[R0:[0-9]+]], [[O0]]($sp)
; ALL: fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL: fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL: copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL: mtc1 $[[R1]], $f[[F0:[0-9]+]]
; ALL: swc1 $f[[F0]]

  %1 = bitcast i16 %b to half
  %2 = fpext half %1 to float
  store float %2, float * @k
  ret void
}

@g = external global i16, align 2
@h = external global half, align 2

; Check that fext f16 to double has a fexupr.w, fexupr.d sequence.
; Check that ftrunc double to f16 has fexdo.w, fexdo.h sequence.
; Check that MIPS64R5+ uses 64-bit floating point <-> 64-bit GPR transfers.

; We don't need to check if pre-MIPSR5 expansions occur, the MSA ASE requires
; MIPSR5. Additionally, fp64 mode / FR=1 is required to use MSA.

define void @fadd_f64() {
entry:
; ALL-LABEL: fadd_f64:
  %0 = load half, half * @h, align 2
  %1 = fpext half %0 to double
; ALL:    lh $[[R0:[0-9]+]]
; ALL:    fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:    fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:    fexupr.d $w[[W2:[0-9]+]], $w[[W1]]
; MIPS32: copy_s.w $[[R1:[0-9]+]], $w[[W2]][0]
; MIPS32: mtc1 $[[R1]], $f[[F0:[0-9]+]]
; MIPS32: copy_s.w $[[R2:[0-9]+]], $w[[W2]][1]
; MIPS32: mthc1 $[[R2]], $f[[F0]]
; MIPS64: copy_s.d $[[R2:[0-9]+]], $w[[W2]][0]
; MIPS64: dmtc1 $[[R2]], $f[[F0:[0-9]+]]

  %2 = load half, half * @h, align 2
  %3 = fpext half %2 to double
  %add = fadd double %1, %3

; ALL: add.d $f[[F1:[0-9]+]], $f[[F0]], $f[[F0]]

  %4 = fptrunc double %add to half

; MIPS32: mfc1 $[[R2:[0-9]+]], $f[[F1]]
; MIPS32: fill.w $w[[W2:[0-9]+]], $[[R2]]
; MIPS32: mfhc1 $[[R3:[0-9]+]], $f[[F1]]
; MIPS32: insert.w $w[[W2]][1], $[[R3]]
; MIPS32: insert.w $w[[W2]][3], $[[R3]]

; MIPS64: dmfc1 $[[R2:[0-9]+]], $f[[F1]]
; MIPS64: fill.d $w[[W2:[0-9]+]], $[[R2]]

; ALL:    fexdo.w $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:    fexdo.h $w[[W4:[0-9]+]], $w[[W3]], $w[[W3]]
; ALL:    copy_u.h $[[R3:[0-9]+]], $w[[W4]][0]
; ALL:    sh $[[R3]]
   store half %4, half * @h, align 2
  ret void
}

define i32 @ffptoui() {
entry:
; ALL-LABEL: ffptoui:
  %0 = load half, half * @h, align 2
  %1 = fptoui half %0 to i32

; MIPS32:       lwc1 $f[[FC:[0-9]+]], %lo($CPI{{[0-9]+}}_{{[0-9]+}})
; MIPS64-N32:   lwc1 $f[[FC:[0-9]+]], %got_ofst(.LCPI{{[0-9]+}}_{{[0-9]+}})
; MIPS64-N64:   lwc1 $f[[FC:[0-9]+]], %got_ofst(.LCPI{{[0-9]+}}_{{[0-9]+}})

; ALL:          lh $[[R0:[0-9]+]]
; ALL:          fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:          fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:          copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL:          mtc1 $[[R1]], $f[[F0:[0-9]+]]
; MIPSR6:       cmp.lt.s  $f[[F1:[0-9]+]], $f[[F0]], $f[[FC]]
; ALL:          sub.s $f[[F2:[0-9]+]], $f[[F0]], $f[[FC]]
; ALL:          mfc1 $[[R2:[0-9]]], $f[[F2]]
; ALL:          fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:          fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:          fexupr.w $w[[W4:[0-9]+]], $w[[W3]]
; ALL:          fexupr.d $w[[W5:[0-9]+]], $w[[W4]]

; MIPS32:       copy_s.w $[[R3:[0-9]+]], $w[[W5]][0]
; MIPS32:       mtc1 $[[R3]], $f[[F3:[0-9]+]]
; MIPS32:       copy_s.w $[[R4:[0-9]+]], $w[[W5]][1]
; MIPS32:       mthc1 $[[R3]], $f[[F3]]

; MIPS64:       copy_s.d $[[R2:[0-9]+]], $w[[W2]][0]
; MIPS64:       dmtc1 $[[R2]], $f[[F3:[0-9]+]]

; ALL:          trunc.w.d $f[[F4:[0-9]+]], $f[[F3]]
; ALL:          mfc1 $[[R4:[0-9]+]], $f[[F4]]
; ALL:          fexupr.d $w[[W6:[0-9]+]], $w[[W1]]

; MIPS32:       copy_s.w $[[R5:[0-9]+]], $w[[W6]][0]
; MIPS32:       mtc1 $[[R5]], $f[[F5:[0-9]+]]
; MIPS32:       copy_s.w $[[R6:[0-9]+]], $w[[W6]][1]
; MIPS32:       mthc1 $[[R6]], $f[[F5]]

; MIPS64:       copy_s.d $[[R2:[0-9]+]], $w[[W2]][0]
; MIPS64:       dmtc1 $[[R2]], $f[[F5:[0-9]+]]

; ALL:          trunc.w.d $f[[F6:[0-9]]], $f[[F5]]
; ALL:          mfc1 $[[R7:[0-9]]], $f[[F6]]

; MIPS32R5-O32: lw $[[R13:[0-9]+]], %got($CPI{{[0-9]+}}_{{[0-9]+}})
; MIPS32R5-O32: addiu $[[R14:[0-9]+]], $[[R13]], %lo($CPI{{[0-9]+}}_{{[0-9]+}})

; MIPS64R5-N32: lw $[[R13:[0-9]+]], %got_page(.LCPI{{[0-9]+}}_{{[0-9]+}})
; MIPS64R5-N32: addiu $[[R14:[0-9]+]], $[[R13]], %got_ofst(.LCPI{{[0-9]+}}_{{[0-9]+}})

; MIPS64R5-N64: ld $[[R13:[0-9]+]], %got_page(.LCPI{{[0-9]+}}_{{[0-9]+}})
; MIPS64R5-N64: daddiu $[[R14:[0-9]+]], $[[R13]], %got_ofst(.LCPI{{[0-9]+}}_{{[0-9]+}})

; ALL:          lui $[[R8:[0-9]+]], 32768
; ALL:          xor $[[R9:[0-9]+]], $[[R4]], $[[R8]]

; MIPSR5:       lh $[[R15:[0-9]+]], 0($[[R14]])
; MIPSR5:       fill.h $w[[W7:[0-9]+]], $[[R15]]
; MIPSR5:       fexupr.w $w[[W8:[0-9]+]], $w[[W7]]
; MIPSR5:       copy_s.w $[[R16:[0-9]+]], $w[[W8]][0]
; MIPSR5:       mtc1 $[[R16]], $f[[F7:[0-9]+]]
; MIPSR5:       c.olt.s $f[[F0]], $f[[F7]]
; MIPSR5:       movt $[[R9]], $[[R7]], $fcc0

; MIPSR6:       mfc1 $[[R10:[0-9]+]], $f[[F1]]
; MIPSR6:       seleqz $[[R11:[0-9]]], $[[R9]], $[[R10]]
; MIPSR6:       selnez $[[R12:[0-9]]], $[[R7]], $[[R10]]
; MIPSR6:       or $2, $[[R12]], $[[R11]]

  ret i32 %1
}

define i32 @ffptosi() {
entry:
; ALL-LABEL: ffptosi:
  %0 = load half, half * @h, align 2
  %1 = fptosi half %0 to i32
  ret i32 %1

; ALL:    lh $[[R0:[0-9]+]]
; ALL:    fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:    fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:    fexupr.d $w[[W2:[0-9]+]], $w[[W1]]

; MIPS32: copy_s.w $[[R2:[0-9]+]], $w[[W2]][0]
; MIPS32: mtc1 $[[R2]], $f[[F0:[0-9]+]]
; MIPS32: copy_s.w $[[R3:[0-9]+]], $w[[W2]][1]
; MIPS32: mthc1 $[[R3]], $f[[F0]]

; MIPS64: copy_s.d $[[R2:[0-9]+]], $w[[W2]][0]
; MIPS64: dmtc1 $[[R2]], $f[[F0:[0-9]+]]

; ALL:    trunc.w.d $f[[F1:[0-9]+]], $f[[F0]]
; ALL:    mfc1 $2, $f[[F1]]
}

define void @uitofp(i32 %a) {
entry:
; ALL-LABEL: uitofp:

; MIPS32-O32: ldc1 $f[[F0:[0-9]+]], %lo($CPI{{[0-9]+}}_{{[0-9]+}})
; MIPS32-O32: ldc1 $f[[F1:[0-9]+]], 0($sp)

; MIPS64-N32: ldc1 $f[[F0:[0-9]+]], %got_ofst(.LCPI{{[0-9]+}}_{{[0-9]+}})
; MIPS64-N32: ldc1 $f[[F1:[0-9]+]], 8($sp)

; MIPS64-N64: ldc1 $f[[F0:[0-9]+]], %got_ofst(.LCPI{{[0-9]+}}_{{[0-9]+}})
; MIPS64-N64: ldc1 $f[[F1:[0-9]+]], 8($sp)

; MIPSR5:     sub.d $f[[F2:[0-9]+]], $f[[F1]], $f[[F0]]
; MIPSR6-O32: sub.d $f[[F2:[0-9]+]], $f[[F0]], $f[[F1]]
; MIPSR6-N32: sub.d $f[[F2:[0-9]+]], $f[[F1]], $f[[F0]]
; MIPSR6-N64: sub.d $f[[F2:[0-9]+]], $f[[F1]], $f[[F0]]

; MIPS32:     mfc1 $[[R0:[0-9]+]], $f[[F2]]
; MIPS32:     fill.w $w[[W0:[0-9]+]], $[[R0]]
; MIPS32:     mfhc1 $[[R1:[0-9]+]], $f[[F2]]
; MIPS32:     insert.w $w[[W0]][1], $[[R1]]
; MIPS32:     insert.w $w[[W0]][3], $[[R1]]

; MIPS64-N64-DAG: ld $[[R3:[0-9]+]], %got_disp(h)
; MIPS64-N32-DAG: lw $[[R3:[0-9]+]], %got_disp(h)
; MIPS64-DAG:     dmfc1 $[[R1:[0-9]+]], $f[[F2]]
; MIPS64-DAG:     fill.d $w[[W0:[0-9]+]], $[[R1]]

; ALL-DAG:        fexdo.w $w[[W1:[0-9]+]], $w[[W0]], $w[[W0]]
; ALL-DAG:        fexdo.h $w[[W2:[0-9]+]], $w[[W1]], $w[[W1]]

; MIPS32-DAG:     lw $[[R3:[0-9]+]], %got(h)

; ALL:        copy_u.h $[[R2:[0-9]+]], $w[[W2]]
; ALL:        sh $[[R2]], 0($[[R3]])
  %0 = uitofp i32 %a to half
  store half %0, half * @h, align 2
  ret void
}


; Check that f16 is expanded to f32 and relevant transfer ops occur.
; We don't check f16 -> f64 expansion occurs, as we expand f16 to f32.

define void @fadd() {
entry:
; ALL-LABEL: fadd:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL: lh $[[R0:[0-9]+]]
; ALL: fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL: fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL: copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL: mtc1 $[[R1]], $f[[F0:[0-9]+]]

  %2 = load i16, i16* @g, align 2
  %3 = call float @llvm.convert.from.fp16.f32(i16 %2)
  %add = fadd float %1, %3

; ALL: add.s $f[[F1:[0-9]+]], $f[[F0]], $f[[F0]]

 %4 = call i16 @llvm.convert.to.fp16.f32(float %add)

; ALL: mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL: fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL: fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL: copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]
; ALL: sh $[[R3]]
   store i16 %4, i16* @g, align 2
  ret void
}

; Function Attrs: nounwind readnone
declare float @llvm.convert.from.fp16.f32(i16)

; Function Attrs: nounwind readnone
declare i16 @llvm.convert.to.fp16.f32(float)

; Function Attrs: nounwind
define void @fsub() {
entry:
; ALL-LABEL: fsub:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL: lh $[[R0:[0-9]+]]
; ALL: fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL: fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL: copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL: mtc1 $[[R1]], $f[[F0:[0-9]+]]

  %2 = load i16, i16* @g, align 2
  %3 = call float @llvm.convert.from.fp16.f32(i16 %2)
  %sub = fsub float %1, %3

; ALL: sub.s $f[[F1:[0-9]+]], $f[[F0]], $f[[F0]]

  %4 = call i16 @llvm.convert.to.fp16.f32(float %sub)

; ALL: mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL: fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL: fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL: copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %4, i16* @g, align 2
; ALL: sh $[[R3]]
  ret void
}

define void @fmult() {
entry:
; ALL-LABEL: fmult:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL: lh $[[R0:[0-9]+]]
; ALL: fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL: fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL: copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL: mtc1 $[[R1]], $f[[F0:[0-9]+]]

  %2 = load i16, i16* @g, align 2
  %3 = call float @llvm.convert.from.fp16.f32(i16 %2)
  %mul = fmul float %1, %3

; ALL: mul.s $f[[F1:[0-9]+]], $f[[F0]], $f[[F0]]

  %4 = call i16 @llvm.convert.to.fp16.f32(float %mul)

; ALL: mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL: fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL: fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL: copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %4, i16* @g, align 2

; ALL: sh $[[R3]]
  ret void
}

define void @fdiv() {
entry:
; ALL-LABEL: fdiv:

  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL: lh $[[R0:[0-9]+]]
; ALL: fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL: fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL: copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL: mtc1 $[[R1]], $f[[F0:[0-9]+]]

  %2 = load i16, i16* @g, align 2
  %3 = call float @llvm.convert.from.fp16.f32(i16 %2)
  %div = fdiv float %1, %3

; ALL: div.s $f[[F1:[0-9]+]], $f[[F0]], $f[[F0]]

  %4 = call i16 @llvm.convert.to.fp16.f32(float %div)

; ALL: mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL: fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL: fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL: copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]
  store i16 %4, i16* @g, align 2
; ALL: sh $[[R3]]
  ret void
}

define void @frem() {
entry:
; ALL-LABEL: frem:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL:        lh $[[R0:[0-9]+]]
; ALL:        fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:        fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:        copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL:        mtc1 $[[R1]], $f[[F0:[0-9]+]]

  %2 = load i16, i16* @g, align 2
  %3 = call float @llvm.convert.from.fp16.f32(i16 %2)
  %rem = frem float %1, %3

; MIPS32:     lw $25, %call16(fmodf)($gp)
; MIPS64-N32: lw $25, %call16(fmodf)($gp)
; MIPS64-N64: ld $25, %call16(fmodf)($gp)
; ALL:        jalr $25

  %4 = call i16 @llvm.convert.to.fp16.f32(float %rem)

; ALL:        mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL:        fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:        fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:        copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %4, i16* @g, align 2
; ALL:        sh $[[R3]]

  ret void
}

@i1 = external global i16, align 1

define void @fcmp() {
entry:
; ALL-LABEL: fcmp:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)
; ALL:        lh $[[R0:[0-9]+]]
; ALL:        fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:        fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:        copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL:        mtc1 $[[R1]], $f[[F0:[0-9]+]]

  %2 = load i16, i16* @g, align 2
  %3 = call float @llvm.convert.from.fp16.f32(i16 %2)
  %fcmp = fcmp oeq float %1, %3

; MIPSR5: addiu $[[R2:[0-9]+]], $zero, 1
; MIPSR5: c.un.s $f[[F0]], $f[[F0]]
; MIPSR5: movt $[[R2]], $zero, $fcc0
; MIPSR6: cmp.un.s $f[[F1:[0-9]+]], $f[[F0]], $f[[F0]]
; MIPSR6: mfc1 $[[R3:[0-9]]], $f[[F1]]
; MIPSR6: not $[[R4:[0-9]+]], $[[R3]]
; MIPSR6: andi $[[R2:[0-9]+]], $[[R4]], 1

  %4 = zext i1 %fcmp to i16
  store i16 %4, i16* @i1, align 2
; ALL:        sh $[[R2]]

  ret void
}

declare float @llvm.powi.f32(float, i32)

define void @fpowi() {
entry:
; ALL-LABEL: fpowi:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL: lh $[[R0:[0-9]+]]
; ALL: fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL: fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL: copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL: mtc1 $[[R1]], $f[[F0:[0-9]+]]

  %powi = call float @llvm.powi.f32(float %1, i32 2)

; ALL: mul.s $f[[F1:[0-9]+]], $f[[F0]], $f[[F0]]

  %2 = call i16 @llvm.convert.to.fp16.f32(float %powi)

; ALL: mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL: fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL: fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL: copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2
; ALL: sh $[[R3]]
  ret void
}

define void @fpowi_var(i32 %var) {
entry:
; ALL-LABEL: fpowi_var:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL:            lh $[[R0:[0-9]+]]
; ALL:            fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:            fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:            copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]

  %powi = call float @llvm.powi.f32(float %1, i32 %var)

; ALL-DAG: mtc1 $[[R1]], $f[[F0:[0-9]+]]
; MIPS32-DAG:     lw $25, %call16(__powisf2)($gp)
; MIPS64-N32-DAG: lw $25, %call16(__powisf2)($gp)
; MIPS64-N64-DAG: ld $25, %call16(__powisf2)($gp)
; ALL-DAG:        jalr $25

  %2 = call i16 @llvm.convert.to.fp16.f32(float %powi)

; ALL:            mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL:            fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:            fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:            copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2
; ALL:            sh $[[R3]]
  ret void
}

declare float @llvm.pow.f32(float %Val, float %power)

define void @fpow(float %var) {
entry:
; ALL-LABEL: fpow:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL:            lh $[[R0:[0-9]+]]
; ALL:            fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:            fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:            copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]

  %powi = call float @llvm.pow.f32(float %1, float %var)

; ALL-DAG:        mtc1 $[[R1]], $f[[F0:[0-9]+]]
; MIPS32-DAG:     lw $25, %call16(powf)($gp)
; MIPS64-N32-DAG: lw $25, %call16(powf)($gp)
; MIPS64-N64-DAG: ld $25, %call16(powf)($gp)
; ALL-DAG:        jalr $25

  %2 = call i16 @llvm.convert.to.fp16.f32(float %powi)

; ALL:            mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL:            fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:            fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:            copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2
; ALL:            sh $[[R3]]
  ret void
}

declare float @llvm.log2.f32(float %Val)

define void @flog2() {
entry:
; ALL-LABEL: flog2:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL:            lh $[[R0:[0-9]+]]
; ALL:            fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:            fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:            copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL-DAG:        mtc1 $[[R1]], $f[[F0:[0-9]+]]
; MIPS32-DAG:     lw $25, %call16(log2f)($gp)
; MIPS64-N32-DAG: lw $25, %call16(log2f)($gp)
; MIPS64-N64-DAG: ld $25, %call16(log2f)($gp)
; ALL-DAG:        jalr $25

  %log2 = call float @llvm.log2.f32(float %1)
  %2 = call i16 @llvm.convert.to.fp16.f32(float %log2)

; ALL:            mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL:            fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:            fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:            copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2
; ALL:            sh $[[R3]]

  ret void
}

declare float @llvm.log10.f32(float %Val)

define void @flog10() {
entry:
; ALL-LABEL: flog10:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL:            lh $[[R0:[0-9]+]]
; ALL:            fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:            fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:            copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL-DAG:        mtc1 $[[R1]], $f[[F0:[0-9]+]]
; MIPS32-DAG:     lw $25, %call16(log10f)($gp)
; MIPS64-N32-DAG: lw $25, %call16(log10f)($gp)
; MIPS64-N64-DAG: ld $25, %call16(log10f)($gp)
; ALL-DAG:        jalr $25

  %log10 = call float @llvm.log10.f32(float %1)
  %2 = call i16 @llvm.convert.to.fp16.f32(float %log10)

; ALL:            mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL:            fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:            fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:            copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2
; ALL:            sh $[[R3]]

  ret void
}

declare float @llvm.sqrt.f32(float %Val)

define void @fsqrt() {
entry:
; ALL-LABEL: fsqrt:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL: lh $[[R0:[0-9]+]]
; ALL: fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL: fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL: copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL: mtc1 $[[R1]], $f[[F0:[0-9]+]]
; ALL: sqrt.s $f[[F1:[0-9]+]], $f[[F0]]

  %sqrt = call float @llvm.sqrt.f32(float %1)
  %2 = call i16 @llvm.convert.to.fp16.f32(float %sqrt)

; ALL: mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL: fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL: fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL: copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2
; ALL: sh $[[R3]]

  ret void
}

declare float @llvm.sin.f32(float %Val)

define void @fsin() {
entry:
; ALL-LABEL: fsin:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL:            lh $[[R0:[0-9]+]]
; ALL:            fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:            fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:            copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL-DAG:        mtc1 $[[R1]], $f[[F0:[0-9]+]]
; MIPS32-DAG:     lw $25, %call16(sinf)($gp)
; MIPS64-N32-DAG: lw $25, %call16(sinf)($gp)
; MIPS64-N64-DAG: ld $25, %call16(sinf)($gp)
; ALL-DAG:        jalr $25

  %sin = call float @llvm.sin.f32(float %1)
  %2 = call i16 @llvm.convert.to.fp16.f32(float %sin)

; ALL:            mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL:            fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:            fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:            copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2
; ALL:            sh $[[R3]]

  ret void
}

declare float @llvm.cos.f32(float %Val)

define void @fcos() {
entry:
; ALL-LABEL: fcos:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL:            lh $[[R0:[0-9]+]]
; ALL:            fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:            fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:            copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL-DAG:        mtc1 $[[R1]], $f[[F0:[0-9]+]]
; MIPS32-DAG:     lw $25, %call16(cosf)($gp)
; MIPS64-N32-DAG: lw $25, %call16(cosf)($gp)
; MIPS64-N64-DAG: ld $25, %call16(cosf)($gp)
; ALL-DAG:        jalr $25

  %cos = call float @llvm.cos.f32(float %1)
  %2 = call i16 @llvm.convert.to.fp16.f32(float %cos)

; ALL:            mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL:            fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:            fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:            copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2
; ALL:            sh $[[R3]]

  ret void
}

declare float @llvm.exp.f32(float %Val)

define void @fexp() {
entry:
; ALL-LABEL: fexp:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)
; ALL:            lh $[[R0:[0-9]+]]
; ALL:            fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:            fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:            copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL-DAG:        mtc1 $[[R1]], $f[[F0:[0-9]+]]
; MIPS32-DAG:     lw $25, %call16(expf)($gp)
; MIPS64-N32-DAG: lw $25, %call16(expf)($gp)
; MIPS64-N64-DAG: ld $25, %call16(expf)($gp)
; ALL-DAG:        jalr $25

  %exp = call float @llvm.exp.f32(float %1)
  %2 = call i16 @llvm.convert.to.fp16.f32(float %exp)

; ALL:            mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL:            fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:            fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:            copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2
; ALL:            sh $[[R3]]

  ret void
}

declare float @llvm.exp2.f32(float %Val)

define void @fexp2() {
entry:
; ALL-LABEL: fexp2:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL:            lh $[[R0:[0-9]+]]
; ALL:            fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:            fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:            copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL-DAG:        mtc1 $[[R1]], $f[[F0:[0-9]+]]
; MIPS32-DAG:     lw $25, %call16(exp2f)($gp)
; MIPS64-N32-DAG: lw $25, %call16(exp2f)($gp)
; MIPS64-N64-DAG: ld $25, %call16(exp2f)($gp)
; ALL-DAG:        jalr $25

  %exp2 = call float @llvm.exp2.f32(float %1)
  %2 = call i16 @llvm.convert.to.fp16.f32(float %exp2)

; ALL:            mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL:            fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:            fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:            copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2
; ALL:            sh $[[R3]]

  ret void
}

declare float @llvm.fma.f32(float, float, float)

define void @ffma(float %b, float %c) {
entry:
; ALL-LABEL: ffma:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL:            lh $[[R0:[0-9]+]]
; ALL:            fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:            fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:            copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL-DAG:        mtc1 $[[R1]], $f[[F0:[0-9]+]]
; MIPS32-DAG:     lw $25, %call16(fmaf)($gp)
; MIPS64-N32-DAG: lw $25, %call16(fmaf)($gp)
; MIPS64-N64-DAG: ld $25, %call16(fmaf)($gp)
; ALL-DAG:        jalr $25

  %fma = call float @llvm.fma.f32(float %1, float %b, float %c)
  %2 = call i16 @llvm.convert.to.fp16.f32(float %fma)

; ALL:            mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL:            fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:            fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:            copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2
; ALL:            sh $[[R3]]

  ret void
}

; FIXME: For MIPSR6, this should produced the maddf.s instruction. MIPSR5 cannot
;        fuse the operation such that the intermediate result is not rounded.

declare float @llvm.fmuladd.f32(float, float, float)

define void @ffmuladd(float %b, float %c) {
entry:
; ALL-LABEL: ffmuladd:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL:            lh $[[R0:[0-9]+]]
; ALL:            fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:            fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:            copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL:            mtc1 $[[R1]], $f[[F0:[0-9]+]]
; MIPS32-O32:     madd.s $f[[F1:[0-9]]], $f14, $f[[F0]], $f12
; MIPS32-N32:     madd.s $f[[F1:[0-9]]], $f13, $f[[F0]], $f12
; MIPS32-N64:     madd.s $f[[F1:[0-9]]], $f13, $f[[F0]], $f12
; MIPSR6:         mul.s $f[[F2:[0-9]+]], $f[[F0]], $f12
; MIPSR6-O32:     add.s $f[[F1:[0-9]+]], $f[[F2]], $f14
; MIPSR6-N32:     add.s $f[[F1:[0-9]+]], $f[[F2]], $f13
; MIPSR6-N64:     add.s $f[[F1:[0-9]+]], $f[[F2]], $f13

  %fmuladd = call float @llvm.fmuladd.f32(float %1, float %b, float %c)
  %2 = call i16 @llvm.convert.to.fp16.f32(float %fmuladd)

; ALL:            mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL:            fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:            fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:            copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2
; ALL:            sh $[[R3]]

  ret void
}

declare float @llvm.fabs.f32(float %Val)

define void @ffabs() {
entry:
; ALL-LABEL: ffabs:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL:            lh $[[R0:[0-9]+]]
; ALL:            fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:            fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:            copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL:            mtc1 $[[R1]], $f[[F0:[0-9]+]]
; ALL:            abs.s $f[[F1:[0-9]+]], $f[[F0]]

  %fabs = call float @llvm.fabs.f32(float %1)
  %2 = call i16 @llvm.convert.to.fp16.f32(float %fabs)

; ALL:            mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL:            fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:            fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:            copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2

; ALL:            sh $[[R3]]
  ret void
}

declare float @llvm.minnum.f32(float %Val, float %b)

define void @fminnum(float %b) {
entry:
; ALL-LABEL: fminnum:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL:            lh $[[R0:[0-9]+]]
; ALL:            fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:            fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:            copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL-DAG:        mtc1 $[[R1]], $f[[F0:[0-9]+]]
; MIPS32-DAG:     lw $25, %call16(fminf)($gp)
; MIPS64-N32-DAG: lw $25, %call16(fminf)($gp)
; MIPS64-N64-DAG: ld $25, %call16(fminf)($gp)
; ALL-DAG:        jalr $25

  %minnum = call float @llvm.minnum.f32(float %1, float %b)
  %2 = call i16 @llvm.convert.to.fp16.f32(float %minnum)

; ALL:            mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL:            fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:            fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:            copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2
; ALL:            sh $[[R3]]

  ret void
}

declare float @llvm.maxnum.f32(float %Val, float %b)

define void @fmaxnum(float %b) {
entry:
; ALL-LABEL: fmaxnum:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL:            lh $[[R0:[0-9]+]]
; ALL:            fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:            fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:            copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL-DAG:        mtc1 $[[R1]], $f[[F0:[0-9]+]]
; MIPS32-DAG:     lw $25, %call16(fmaxf)($gp)
; MIPS64-N32-DAG: lw $25, %call16(fmaxf)($gp)
; MIPS64-N64-DAG: ld $25, %call16(fmaxf)($gp)
; ALL-DAG:        jalr $25

  %maxnum = call float @llvm.maxnum.f32(float %1, float %b)
  %2 = call i16 @llvm.convert.to.fp16.f32(float %maxnum)

; ALL:            mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL:            fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:            fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:            copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2
; ALL:             sh $[[R3]]

  ret void
}

; This expansion of fcopysign could be done without converting f16 to float.

declare float @llvm.copysign.f32(float %Val, float %b)

define void @fcopysign(float %b) {
entry:
; ALL-LABEL: fcopysign:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL:            lh $[[R0:[0-9]+]]
; ALL:            fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:            fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:            copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]

  %copysign = call float @llvm.copysign.f32(float %1, float %b)
  %2 = call i16 @llvm.convert.to.fp16.f32(float %copysign)

; ALL:            mfc1 $[[R2:[0-9]+]], $f12
; ALL:            ext $[[R3:[0-9]+]], $3, 31, 1
; ALL:            ins $[[R1]], $[[R3]], 31, 1
; ALL:            fill.w $w[[W2:[0-9]+]], $[[R1]]
; ALL:            fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:            copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2
; ALL:            sh $[[R3]]

  ret void
}

declare float @llvm.floor.f32(float %Val)

define void @ffloor() {
entry:
; ALL-LABEL: ffloor:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL:            lh $[[R0:[0-9]+]]
; ALL:            fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:            fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:            copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL-DAG:        mtc1 $[[R1]], $f[[F0:[0-9]+]]
; MIPS32-DAG:     lw $25, %call16(floorf)($gp)
; MIPS64-N32-DAG: lw $25, %call16(floorf)($gp)
; MIPS64-N64-DAG: ld $25, %call16(floorf)($gp)
; ALL-DAG:        jalr $25

  %floor = call float @llvm.floor.f32(float %1)
  %2 = call i16 @llvm.convert.to.fp16.f32(float %floor)

; ALL:            mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL:            fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:            fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:            copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2
; ALL:            sh $[[R3]]

  ret void
}

declare float @llvm.ceil.f32(float %Val)

define void @fceil() {
entry:
; ALL-LABEL: fceil:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL:            lh $[[R0:[0-9]+]]
; ALL:            fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:            fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:            copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL-DAG:        mtc1 $[[R1]], $f[[F0:[0-9]+]]
; MIPS32-DAG:     lw $25, %call16(ceilf)($gp)
; MIPS64-N32-DAG: lw $25, %call16(ceilf)($gp)
; MIPS64-N64-DAG: ld $25, %call16(ceilf)($gp)
; ALL-DAG:        jalr $25

  %ceil = call float @llvm.ceil.f32(float %1)
  %2 = call i16 @llvm.convert.to.fp16.f32(float %ceil)

; ALL:            mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL:            fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:            fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:            copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2
; ALL:            sh $[[R3]]

  ret void
}

declare float @llvm.trunc.f32(float %Val)

define void @ftrunc() {
entry:
; ALL-LABEL: ftrunc:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL:            lh $[[R0:[0-9]+]]
; ALL:            fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:            fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:            copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL-DAG:        mtc1 $[[R1]], $f[[F0:[0-9]+]]
; MIPS32-DAG:     lw $25, %call16(truncf)($gp)
; MIPS64-N32-DAG: lw $25, %call16(truncf)($gp)
; MIPS64-N64-DAG: ld $25, %call16(truncf)($gp)
; ALL-DAG:        jalr $25

  %trunc = call float @llvm.trunc.f32(float %1)
  %2 = call i16 @llvm.convert.to.fp16.f32(float %trunc)

; ALL:            mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL:            fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:            fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:            copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2
; ALL:            sh $[[R3]]

  ret void
}

declare float @llvm.rint.f32(float %Val)

define void @frint() {
entry:
; ALL-LABEL: frint:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL:            lh $[[R0:[0-9]+]]
; ALL:            fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:            fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:            copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL-DAG:        mtc1 $[[R1]], $f[[F0:[0-9]+]]
; MIPS32-DAG:     lw $25, %call16(rintf)($gp)
; MIPS64-N32-DAG: lw $25, %call16(rintf)($gp)
; MIPS64-N64-DAG: ld $25, %call16(rintf)($gp)
; ALL-DAG:        jalr $25
  %rint = call float @llvm.rint.f32(float %1)
  %2 = call i16 @llvm.convert.to.fp16.f32(float %rint)

; ALL:            mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL:            fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:            fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:            copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]
  store i16 %2, i16* @g, align 2

; ALL:            sh $[[R3]]
  ret void
}

declare float @llvm.nearbyint.f32(float %Val)

define void @fnearbyint() {
entry:
; ALL-LABEL: fnearbyint:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL:            lh $[[R0:[0-9]+]]
; ALL:            fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:            fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:            copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL-DAG:        mtc1 $[[R1]], $f[[F0:[0-9]+]]
; MIPS32-DAG:     lw $25, %call16(nearbyintf)($gp)
; MIPS64-N32-DAG: lw $25, %call16(nearbyintf)($gp)
; MIPS64-N64-DAG: ld $25, %call16(nearbyintf)($gp)
; ALL-DAG:        jalr $25

  %nearbyint = call float @llvm.nearbyint.f32(float %1)
  %2 = call i16 @llvm.convert.to.fp16.f32(float %nearbyint)

; ALL:            mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL:            fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:            fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:            copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2
; ALL:            sh $[[R3]]

  ret void
}

declare float @llvm.round.f32(float %Val)

define void @fround() {
entry:
; ALL-LABEL: fround:
  %0 = load i16, i16* @g, align 2
  %1 = call float @llvm.convert.from.fp16.f32(i16 %0)

; ALL:            lh $[[R0:[0-9]+]]
; ALL:            fill.h $w[[W0:[0-9]+]], $[[R0]]
; ALL:            fexupr.w $w[[W1:[0-9]+]], $w[[W0]]
; ALL:            copy_s.w $[[R1:[0-9]+]], $w[[W1]][0]
; ALL-DAG:        mtc1 $[[R1]], $f[[F0:[0-9]+]]
; MIPS32-DAG:     lw $25, %call16(roundf)($gp)
; MIPS64-N32-DAG: lw $25, %call16(roundf)($gp)
; MIPS64-N64-DAG: ld $25, %call16(roundf)($gp)
; ALL-DAG:        jalr $25

  %round = call float @llvm.round.f32(float %1)
  %2 = call i16 @llvm.convert.to.fp16.f32(float %round)

; ALL:            mfc1 $[[R2:[0-9]+]], $f[[F1]]
; ALL:            fill.w $w[[W2:[0-9]+]], $[[R2]]
; ALL:            fexdo.h $w[[W3:[0-9]+]], $w[[W2]], $w[[W2]]
; ALL:            copy_u.h $[[R3:[0-9]+]], $w[[W3]][0]

  store i16 %2, i16* @g, align 2
; ALL:            sh $[[R3]]

  ret void
}
