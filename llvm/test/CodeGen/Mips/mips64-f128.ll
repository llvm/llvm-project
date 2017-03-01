; RUN: llc -mtriple=mips64el-unknown-unknown -mcpu=mips4 -mattr=+soft-float -O1 \
; RUN:     -disable-mips-delay-filler -relocation-model=pic < %s | FileCheck \
; RUN:     %s -check-prefixes=ALL,C_CC_FMT,PRER6
; RUN: llc -mtriple=mips64el-unknown-unknown -mcpu=mips64 -mattr=+soft-float -O1 \
; RUN:     -disable-mips-delay-filler -relocation-model=pic < %s | FileCheck \
; RUN:     %s -check-prefixes=ALL,C_CC_FMT,PRER6
; RUN: llc -mtriple=mips64el-unknown-unknown -mcpu=mips64r2 -mattr=+soft-float \
; RUN:     -O1 -disable-mips-delay-filler -relocation-model=pic < %s | FileCheck \
; RUN:     %s -check-prefixes=ALL,C_CC_FMT,PRER6
; RUN: llc -mtriple=mips64el-unknown-unknown -mcpu=mips64r6 -mattr=+soft-float \
; RUN:     -O1 -disable-mips-delay-filler -relocation-model=pic < %s | FileCheck \
; RUN:     %s -check-prefixes=ALL,CMP_CC_FMT,R6

@gld0 = external global fp128
@gld1 = external global fp128
@gld2 = external global fp128
@gf1 = external global float
@gd1 = external global double

; ALL-LABEL: addLD:
; ALL: ld $25, %call16(__addtf3)

define fp128 @addLD() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %1 = load fp128, fp128* @gld1, align 16
  %add = fadd fp128 %0, %1
  ret fp128 %add
}

; ALL-LABEL: subLD:
; ALL: ld $25, %call16(__subtf3)

define fp128 @subLD() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %1 = load fp128, fp128* @gld1, align 16
  %sub = fsub fp128 %0, %1
  ret fp128 %sub
}

; ALL-LABEL: mulLD:
; ALL: ld $25, %call16(__multf3)

define fp128 @mulLD() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %1 = load fp128, fp128* @gld1, align 16
  %mul = fmul fp128 %0, %1
  ret fp128 %mul
}

; ALL-LABEL: divLD:
; ALL: ld $25, %call16(__divtf3)

define fp128 @divLD() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %1 = load fp128, fp128* @gld1, align 16
  %div = fdiv fp128 %0, %1
  ret fp128 %div
}

; ALL-LABEL: conv_LD_char:
; ALL: ld $25, %call16(__floatsitf)

define fp128 @conv_LD_char(i8 signext %a) {
entry:
  %conv = sitofp i8 %a to fp128
  ret fp128 %conv
}

; ALL-LABEL: conv_LD_short:
; ALL: ld $25, %call16(__floatsitf)

define fp128 @conv_LD_short(i16 signext %a) {
entry:
  %conv = sitofp i16 %a to fp128
  ret fp128 %conv
}

; ALL-LABEL: conv_LD_int:
; ALL: ld $25, %call16(__floatsitf)

define fp128 @conv_LD_int(i32 %a) {
entry:
  %conv = sitofp i32 %a to fp128
  ret fp128 %conv
}

; ALL-LABEL: conv_LD_LL:
; ALL: ld $25, %call16(__floatditf)

define fp128 @conv_LD_LL(i64 %a) {
entry:
  %conv = sitofp i64 %a to fp128
  ret fp128 %conv
}

; ALL-LABEL: conv_LD_UChar:
; ALL: ld $25, %call16(__floatunsitf)

define fp128 @conv_LD_UChar(i8 zeroext %a) {
entry:
  %conv = uitofp i8 %a to fp128
  ret fp128 %conv
}

; ALL-LABEL: conv_LD_UShort:
; ALL: ld $25, %call16(__floatunsitf)

define fp128 @conv_LD_UShort(i16 zeroext %a) {
entry:
  %conv = uitofp i16 %a to fp128
  ret fp128 %conv
}

; ALL-LABEL: conv_LD_UInt:
; ALL: ld $25, %call16(__floatunsitf)

define fp128 @conv_LD_UInt(i32 signext %a) {
entry:
  %conv = uitofp i32 %a to fp128
  ret fp128 %conv
}

; ALL-LABEL: conv_LD_ULL:
; ALL: ld $25, %call16(__floatunditf)

define fp128 @conv_LD_ULL(i64 %a) {
entry:
  %conv = uitofp i64 %a to fp128
  ret fp128 %conv
}

; ALL-LABEL: conv_char_LD:
; ALL: ld $25, %call16(__fixtfsi)

define signext i8 @conv_char_LD(fp128 %a) {
entry:
  %conv = fptosi fp128 %a to i8
  ret i8 %conv
}

; ALL-LABEL: conv_short_LD:
; ALL: ld $25, %call16(__fixtfsi)

define signext i16 @conv_short_LD(fp128 %a) {
entry:
  %conv = fptosi fp128 %a to i16
  ret i16 %conv
}

; ALL-LABEL: conv_int_LD:
; ALL: ld $25, %call16(__fixtfsi)

define i32 @conv_int_LD(fp128 %a) {
entry:
  %conv = fptosi fp128 %a to i32
  ret i32 %conv
}

; ALL-LABEL: conv_LL_LD:
; ALL: ld $25, %call16(__fixtfdi)

define i64 @conv_LL_LD(fp128 %a) {
entry:
  %conv = fptosi fp128 %a to i64
  ret i64 %conv
}

; ALL-LABEL: conv_UChar_LD:
; ALL: ld $25, %call16(__fixtfsi)

define zeroext i8 @conv_UChar_LD(fp128 %a) {
entry:
  %conv = fptoui fp128 %a to i8
  ret i8 %conv
}

; ALL-LABEL: conv_UShort_LD:
; ALL: ld $25, %call16(__fixtfsi)

define zeroext i16 @conv_UShort_LD(fp128 %a) {
entry:
  %conv = fptoui fp128 %a to i16
  ret i16 %conv
}

; ALL-LABEL: conv_UInt_LD:
; ALL: ld $25, %call16(__fixunstfsi)

define i32 @conv_UInt_LD(fp128 %a) {
entry:
  %conv = fptoui fp128 %a to i32
  ret i32 %conv
}

; ALL-LABEL: conv_ULL_LD:
; ALL: ld $25, %call16(__fixunstfdi)

define i64 @conv_ULL_LD(fp128 %a) {
entry:
  %conv = fptoui fp128 %a to i64
  ret i64 %conv
}

; ALL-LABEL: conv_LD_float:
; ALL: ld $25, %call16(__extendsftf2)

define fp128 @conv_LD_float(float %a) {
entry:
  %conv = fpext float %a to fp128
  ret fp128 %conv
}

; ALL-LABEL: conv_LD_double:
; ALL: ld $25, %call16(__extenddftf2)

define fp128 @conv_LD_double(double %a) {
entry:
  %conv = fpext double %a to fp128
  ret fp128 %conv
}

; ALL-LABEL: conv_float_LD:
; ALL: ld $25, %call16(__trunctfsf2)

define float @conv_float_LD(fp128 %a) {
entry:
  %conv = fptrunc fp128 %a to float
  ret float %conv
}

; ALL-LABEL: conv_double_LD:
; ALL: ld $25, %call16(__trunctfdf2)

define double @conv_double_LD(fp128 %a) {
entry:
  %conv = fptrunc fp128 %a to double
  ret double %conv
}

; ALL-LABEL:             libcall1_fabsl:
; ALL-DAG: ld      $[[R0:[0-9]+]], 8($[[R4:[0-9]+]])
; ALL-DAG: daddiu  $[[R1:[0-9]+]], $zero, 1
; ALL-DAG: dsll    $[[R2:[0-9]+]], $[[R1]], 63
; ALL-DAG: daddiu  $[[R3:[0-9]+]], $[[R2]], -1
; ALL-DAG: and     $4, $[[R0]], $[[R3]]
; ALL-DAG: ld      $2, 0($[[R4]])

define fp128 @libcall1_fabsl() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %call = tail call fp128 @fabsl(fp128 %0) nounwind readnone
  ret fp128 %call
}

declare fp128 @fabsl(fp128) #1

; ALL-LABEL: libcall1_ceill:
; ALL: ld $25, %call16(ceill)

define fp128 @libcall1_ceill() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %call = tail call fp128 @ceill(fp128 %0) nounwind readnone
  ret fp128 %call
}

declare fp128 @ceill(fp128) #1

; ALL-LABEL: libcall1_sinl:
; ALL: ld $25, %call16(sinl)

define fp128 @libcall1_sinl() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %call = tail call fp128 @sinl(fp128 %0) nounwind
  ret fp128 %call
}

declare fp128 @sinl(fp128) #2

; ALL-LABEL: libcall1_cosl:
; ALL: ld $25, %call16(cosl)

define fp128 @libcall1_cosl() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %call = tail call fp128 @cosl(fp128 %0) nounwind
  ret fp128 %call
}

declare fp128 @cosl(fp128) #2

; ALL-LABEL: libcall1_expl:
; ALL: ld $25, %call16(expl)

define fp128 @libcall1_expl() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %call = tail call fp128 @expl(fp128 %0) nounwind
  ret fp128 %call
}

declare fp128 @expl(fp128) #2

; ALL-LABEL: libcall1_exp2l:
; ALL: ld $25, %call16(exp2l)

define fp128 @libcall1_exp2l() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %call = tail call fp128 @exp2l(fp128 %0) nounwind
  ret fp128 %call
}

declare fp128 @exp2l(fp128) #2

; ALL-LABEL: libcall1_logl:
; ALL: ld $25, %call16(logl)

define fp128 @libcall1_logl() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %call = tail call fp128 @logl(fp128 %0) nounwind
  ret fp128 %call
}

declare fp128 @logl(fp128) #2

; ALL-LABEL: libcall1_log2l:
; ALL: ld $25, %call16(log2l)

define fp128 @libcall1_log2l() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %call = tail call fp128 @log2l(fp128 %0) nounwind
  ret fp128 %call
}

declare fp128 @log2l(fp128) #2

; ALL-LABEL: libcall1_log10l:
; ALL: ld $25, %call16(log10l)

define fp128 @libcall1_log10l() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %call = tail call fp128 @log10l(fp128 %0) nounwind
  ret fp128 %call
}

declare fp128 @log10l(fp128) #2

; ALL-LABEL: libcall1_nearbyintl:
; ALL: ld $25, %call16(nearbyintl)

define fp128 @libcall1_nearbyintl() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %call = tail call fp128 @nearbyintl(fp128 %0) nounwind readnone
  ret fp128 %call
}

declare fp128 @nearbyintl(fp128) #1

; ALL-LABEL: libcall1_floorl:
; ALL: ld $25, %call16(floorl)

define fp128 @libcall1_floorl() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %call = tail call fp128 @floorl(fp128 %0) nounwind readnone
  ret fp128 %call
}

declare fp128 @floorl(fp128) #1

; ALL-LABEL: libcall1_sqrtl:
; ALL: ld $25, %call16(sqrtl)

define fp128 @libcall1_sqrtl() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %call = tail call fp128 @sqrtl(fp128 %0) nounwind
  ret fp128 %call
}

declare fp128 @sqrtl(fp128) #2

; ALL-LABEL: libcall1_rintl:
; ALL: ld $25, %call16(rintl)

define fp128 @libcall1_rintl() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %call = tail call fp128 @rintl(fp128 %0) nounwind readnone
  ret fp128 %call
}

declare fp128 @rintl(fp128) #1

; ALL-LABEL: libcall_powil:
; ALL: ld $25, %call16(__powitf2)

define fp128 @libcall_powil(fp128 %a, i32 %b) {
entry:
  %0 = tail call fp128 @llvm.powi.f128(fp128 %a, i32 %b)
  ret fp128 %0
}

declare fp128 @llvm.powi.f128(fp128, i32) #3

; ALL-LABEL:     libcall2_copysignl:
; ALL-DAG: daddiu $[[R2:[0-9]+]], $zero, 1
; ALL-DAG: dsll   $[[R3:[0-9]+]], $[[R2]], 63
; ALL-DAG: ld     $[[R0:[0-9]+]], %got_disp(gld1)
; ALL-DAG: ld     $[[R1:[0-9]+]], 8($[[R0]])
; ALL-DAG: and    $[[R4:[0-9]+]], $[[R1]], $[[R3]]
; ALL-DAG: ld     $[[R5:[0-9]+]], %got_disp(gld0)
; ALL-DAG: ld     $[[R6:[0-9]+]], 8($[[R5]])
; ALL-DAG: daddiu $[[R7:[0-9]+]], $[[R3]], -1
; ALL-DAG: and    $[[R8:[0-9]+]], $[[R6]], $[[R7]]
; ALL-DAG: or     $4, $[[R8]], $[[R4]]
; ALL-DAG: ld     $2, 0($[[R5]])

define fp128 @libcall2_copysignl() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %1 = load fp128, fp128* @gld1, align 16
  %call = tail call fp128 @copysignl(fp128 %0, fp128 %1) nounwind readnone
  ret fp128 %call
}

declare fp128 @copysignl(fp128, fp128) #1

; ALL-LABEL: libcall2_powl:
; ALL: ld $25, %call16(powl)

define fp128 @libcall2_powl() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %1 = load fp128, fp128* @gld1, align 16
  %call = tail call fp128 @powl(fp128 %0, fp128 %1) nounwind
  ret fp128 %call
}

declare fp128 @powl(fp128, fp128) #2

; ALL-LABEL: libcall2_fmodl:
; ALL: ld $25, %call16(fmodl)

define fp128 @libcall2_fmodl() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %1 = load fp128, fp128* @gld1, align 16
  %call = tail call fp128 @fmodl(fp128 %0, fp128 %1) nounwind
  ret fp128 %call
}

declare fp128 @fmodl(fp128, fp128) #2

; ALL-LABEL: libcall3_fmal:
; ALL: ld $25, %call16(fmal)

define fp128 @libcall3_fmal() {
entry:
  %0 = load fp128, fp128* @gld0, align 16
  %1 = load fp128, fp128* @gld2, align 16
  %2 = load fp128, fp128* @gld1, align 16
  %3 = tail call fp128 @llvm.fma.f128(fp128 %0, fp128 %2, fp128 %1)
  ret fp128 %3
}

declare fp128 @llvm.fma.f128(fp128, fp128, fp128) #4

; ALL-LABEL: cmp_lt:
; ALL: ld $25, %call16(__lttf2)

define i32 @cmp_lt(fp128 %a, fp128 %b) {
entry:
  %cmp = fcmp olt fp128 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; ALL-LABEL: cmp_le:
; ALL: ld $25, %call16(__letf2)

define i32 @cmp_le(fp128 %a, fp128 %b) {
entry:
  %cmp = fcmp ole fp128 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; ALL-LABEL: cmp_gt:
; ALL: ld $25, %call16(__gttf2)

define i32 @cmp_gt(fp128 %a, fp128 %b) {
entry:
  %cmp = fcmp ogt fp128 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; ALL-LABEL: cmp_ge:
; ALL: ld $25, %call16(__getf2)

define i32 @cmp_ge(fp128 %a, fp128 %b) {
entry:
  %cmp = fcmp oge fp128 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; ALL-LABEL: cmp_eq:
; ALL: ld $25, %call16(__eqtf2)

define i32 @cmp_eq(fp128 %a, fp128 %b) {
entry:
  %cmp = fcmp oeq fp128 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; ALL-LABEL: cmp_ne:
; ALL: ld $25, %call16(__netf2)

define i32 @cmp_ne(fp128 %a, fp128 %b) {
entry:
  %cmp = fcmp une fp128 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; ALL-LABEL: load_LD_LD:
; ALL: ld $[[R0:[0-9]+]], %got_disp(gld1)
; ALL: ld $2, 0($[[R0]])
; ALL: ld $4, 8($[[R0]])

define fp128 @load_LD_LD() {
entry:
  %0 = load fp128, fp128* @gld1, align 16
  ret fp128 %0
}

; ALL-LABEL: load_LD_float:
; ALL:   ld   $[[R0:[0-9]+]], %got_disp(gf1)
; ALL:   lw   $4, 0($[[R0]])
; ALL:   ld   $25, %call16(__extendsftf2)
; PRER6: jalr $25
; R6:    jalrc $25

define fp128 @load_LD_float() {
entry:
  %0 = load float, float* @gf1, align 4
  %conv = fpext float %0 to fp128
  ret fp128 %conv
}

; ALL-LABEL: load_LD_double:
; ALL:   ld   $[[R0:[0-9]+]], %got_disp(gd1)
; ALL:   ld   $4, 0($[[R0]])
; ALL:   ld   $25, %call16(__extenddftf2)
; PRER6: jalr $25
; R6:    jalrc $25

define fp128 @load_LD_double() {
entry:
  %0 = load double, double* @gd1, align 8
  %conv = fpext double %0 to fp128
  ret fp128 %conv
}

; ALL-LABEL: store_LD_LD:
; ALL: ld $[[R0:[0-9]+]], %got_disp(gld1)
; ALL: ld $[[R2:[0-9]+]], 8($[[R0]])
; ALL: ld $[[R3:[0-9]+]], %got_disp(gld0)
; ALL: sd $[[R2]], 8($[[R3]])
; ALL: ld $[[R1:[0-9]+]], 0($[[R0]])
; ALL: sd $[[R1]], 0($[[R3]])

define void @store_LD_LD() {
entry:
  %0 = load fp128, fp128* @gld1, align 16
  store fp128 %0, fp128* @gld0, align 16
  ret void
}

; ALL-LABEL: store_LD_float:
; ALL:   ld   $[[R0:[0-9]+]], %got_disp(gld1)
; ALL:   ld   $4, 0($[[R0]])
; ALL:   ld   $5, 8($[[R0]])
; ALL:   ld   $25, %call16(__trunctfsf2)
; PRER6: jalr $25
; R6:    jalrc $25
; ALL:   ld   $[[R1:[0-9]+]], %got_disp(gf1)
; ALL:   sw   $2, 0($[[R1]])

define void @store_LD_float() {
entry:
  %0 = load fp128, fp128* @gld1, align 16
  %conv = fptrunc fp128 %0 to float
  store float %conv, float* @gf1, align 4
  ret void
}

; ALL-LABEL: store_LD_double:
; ALL:   ld   $[[R0:[0-9]+]], %got_disp(gld1)
; ALL:   ld   $4, 0($[[R0]])
; ALL:   ld   $5, 8($[[R0]])
; ALL:   ld   $25, %call16(__trunctfdf2)
; PRER6: jalr $25
; R6:    jalrc $25
; ALL:   ld   $[[R1:[0-9]+]], %got_disp(gd1)
; ALL:   sd   $2, 0($[[R1]])

define void @store_LD_double() {
entry:
  %0 = load fp128, fp128* @gld1, align 16
  %conv = fptrunc fp128 %0 to double
  store double %conv, double* @gd1, align 8
  ret void
}

; ALL-LABEL: select_LD:
; C_CC_FMT:      movn $8, $6, $4
; C_CC_FMT:      movn $9, $7, $4
; C_CC_FMT:      move $2, $8
; C_CC_FMT:      move $4, $9

; FIXME: This sll works around an implementation detail in the code generator
;        (setcc's result is i32 so bits 32-63 are undefined). It's not really
;        needed.
; CMP_CC_FMT-DAG: sll $[[CC:[0-9]+]], $4, 0
; CMP_CC_FMT-DAG: seleqz $[[EQ1:[0-9]+]], $8, $[[CC]]
; CMP_CC_FMT-DAG: selnez $[[NE1:[0-9]+]], $6, $[[CC]]
; CMP_CC_FMT-DAG: or $2, $[[NE1]], $[[EQ1]]
; CMP_CC_FMT-DAG: seleqz $[[EQ2:[0-9]+]], $9, $[[CC]]
; CMP_CC_FMT-DAG: selnez $[[NE2:[0-9]+]], $7, $[[CC]]
; CMP_CC_FMT-DAG: or $4, $[[NE2]], $[[EQ2]]

define fp128 @select_LD(i32 signext %a, i64, fp128 %b, fp128 %c) {
entry:
  %tobool = icmp ne i32 %a, 0
  %cond = select i1 %tobool, fp128 %b, fp128 %c
  ret fp128 %cond
}

; ALL-LABEL: selectCC_LD:
; ALL:           move $[[R0:[0-9]+]], $11
; ALL:           move $[[R1:[0-9]+]], $10
; ALL:           move $[[R2:[0-9]+]], $9
; ALL:           move $[[R3:[0-9]+]], $8
; ALL:           ld   $25, %call16(__gttf2)($gp)
; PRER6:         jalr $25
; R6:            jalrc $25

; C_CC_FMT:      slti $[[CC:[0-9]+]], $2, 1
; C_CC_FMT:      movz $[[R1]], $[[R3]], $[[CC]]
; C_CC_FMT:      movz $[[R0]], $[[R2]], $[[CC]]
; C_CC_FMT:      move $2, $[[R1]]
; C_CC_FMT:      move $4, $[[R0]]

; CMP_CC_FMT:    slt $[[CC:[0-9]+]], $zero, $2
; CMP_CC_FMT:    seleqz $[[EQ1:[0-9]+]], $[[R1]], $[[CC]]
; CMP_CC_FMT:    selnez $[[NE1:[0-9]+]], $[[R3]], $[[CC]]
; CMP_CC_FMT:    or $2, $[[NE1]], $[[EQ1]]
; CMP_CC_FMT:    seleqz $[[EQ2:[0-9]+]], $[[R0]], $[[CC]]
; CMP_CC_FMT:    selnez $[[NE2:[0-9]+]], $[[R2]], $[[CC]]
; CMP_CC_FMT:    or $4, $[[NE2]], $[[EQ2]]

define fp128 @selectCC_LD(fp128 %a, fp128 %b, fp128 %c, fp128 %d) {
entry:
  %cmp = fcmp ogt fp128 %a, %b
  %cond = select i1 %cmp, fp128 %c, fp128 %d
  ret fp128 %cond
}
