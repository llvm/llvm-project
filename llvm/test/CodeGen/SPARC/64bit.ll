; RUN: llc < %s -march=sparcv9 -mattr=+popc -disable-sparc-delay-filler -disable-sparc-leaf-proc | FileCheck %s
; RUN: llc < %s -march=sparcv9 -mattr=+popc | FileCheck %s -check-prefix=OPT
; RUN: llc %s -march=sparcv9 -mattr=+popc -filetype=null

; CHECK-LABEL: ret2:
; CHECK: mov %i1, %i0

; OPT-LABEL: ret2:
; OPT: retl
; OPT: mov %o1, %o0
define i64 @ret2(i64 %a, i64 %b) {
  ret i64 %b
}

; CHECK: shl_imm
; CHECK: sllx %i0, 7, %i0

; OPT-LABEL: shl_imm:
; OPT: retl
; OPT: sllx %o0, 7, %o0
define i64 @shl_imm(i64 %a) {
  %x = shl i64 %a, 7
  ret i64 %x
}

; CHECK: sra_reg
; CHECK: srax %i0, %i1, %i0

; OPT-LABEL: sra_reg:
; OPT: retl
; OPT: srax %o0, %o1, %o0
define i64 @sra_reg(i64 %a, i64 %b) {
  %x = ashr i64 %a, %b
  ret i64 %x
}

; Immediate materialization. Many of these patterns could actually be merged
; into the restore instruction:
;
;     restore %g0, %g0, %o0
;
; CHECK: ret_imm0
; CHECK: mov 0, %i0

; OPT: ret_imm0
; OPT: retl
; OPT: mov 0, %o0
define i64 @ret_imm0() {
  ret i64 0
}

; CHECK: ret_simm13
; CHECK: mov -4096, %i0

; OPT:   ret_simm13
; OPT:   retl
; OPT:   mov -4096, %o0
define i64 @ret_simm13() {
  ret i64 -4096
}

; CHECK: ret_sethi
; CHECK: sethi 4, %i0
; CHECK-NOT: or
; CHECK: restore

; OPT:  ret_sethi
; OPT:  retl
; OPT:  sethi 4, %o0
define i64 @ret_sethi() {
  ret i64 4096
}

; CHECK: ret_sethi_or
; CHECK: sethi 4, [[R:%[goli][0-7]]]
; CHECK: or [[R]], 1, %i0

; OPT: ret_sethi_or
; OPT: sethi 4, [[R:%[go][0-7]]]
; OPT: retl
; OPT: or [[R]], 1, %o0

define i64 @ret_sethi_or() {
  ret i64 4097
}

; CHECK: ret_nimm33
; CHECK: sethi 4, [[R:%[goli][0-7]]]
; CHECK: xor [[R]], -4, %i0

; OPT: ret_nimm33
; OPT: sethi 4, [[R:%[go][0-7]]]
; OPT: retl
; OPT: xor [[R]], -4, %o0

define i64 @ret_nimm33() {
  ret i64 -4100
}

; CHECK: ret_bigimm
; CHECK: sethi
; CHECK: sethi
define i64 @ret_bigimm() {
  ret i64 6800754272627607872
}

; CHECK: ret_bigimm2
; CHECK: sethi 1048576
define i64 @ret_bigimm2() {
  ret i64 4611686018427387904 ; 0x4000000000000000
}

; CHECK: reg_reg_alu
; CHECK: add %i0, %i1, [[R0:%[goli][0-7]]]
; CHECK: sub [[R0]], %i2, [[R1:%[goli][0-7]]]
; CHECK: andn [[R1]], %i0, %i0
define i64 @reg_reg_alu(i64 %x, i64 %y, i64 %z) {
  %a = add i64 %x, %y
  %b = sub i64 %a, %z
  %c = xor i64 %x, -1
  %d = and i64 %b, %c
  ret i64 %d
}

; CHECK: reg_imm_alu
; CHECK: add %i0, -5, [[R0:%[goli][0-7]]]
; CHECK: xor [[R0]], 2, %i0
define i64 @reg_imm_alu(i64 %x, i64 %y, i64 %z) {
  %a = add i64 %x, -5
  %b = xor i64 %a, 2
  ret i64 %b
}

; CHECK: loads
; CHECK: ldx [%i0]
; CHECK: stx %
; CHECK: ld [%i1]
; CHECK: st %
; CHECK: ldsw [%i2]
; CHECK: stx %
; CHECK: ldsh [%i3]
; CHECK: sth %
define i64 @loads(i64* %p, i32* %q, i32* %r, i16* %s) {
  %a = load i64, i64* %p
  %ai = add i64 1, %a
  store i64 %ai, i64* %p
  %b = load i32, i32* %q
  %b2 = zext i32 %b to i64
  %bi = trunc i64 %ai to i32
  store i32 %bi, i32* %q
  %c = load i32, i32* %r
  %c2 = sext i32 %c to i64
  store i64 %ai, i64* %p
  %d = load i16, i16* %s
  %d2 = sext i16 %d to i64
  %di = trunc i64 %ai to i16
  store i16 %di, i16* %s

  %x1 = add i64 %a, %b2
  %x2 = add i64 %c2, %d2
  %x3 = add i64 %x1, %x2
  ret i64 %x3
}

; CHECK: load_bool
; CHECK: ldub [%i0], %i0
define i64 @load_bool(i1* %p) {
  %a = load i1, i1* %p
  %b = zext i1 %a to i64
  ret i64 %b
}

; CHECK: stores
; CHECK: ldx [%i0+8], [[R:%[goli][0-7]]]
; CHECK: stx [[R]], [%i0+16]
; CHECK: st [[R]], [%i1+-8]
; CHECK: sth [[R]], [%i2+40]
; CHECK: stb [[R]], [%i3+-20]
define void @stores(i64* %p, i32* %q, i16* %r, i8* %s) {
  %p1 = getelementptr i64, i64* %p, i64 1
  %p2 = getelementptr i64, i64* %p, i64 2
  %pv = load i64, i64* %p1
  store i64 %pv, i64* %p2

  %q2 = getelementptr i32, i32* %q, i32 -2
  %qv = trunc i64 %pv to i32
  store i32 %qv, i32* %q2

  %r2 = getelementptr i16, i16* %r, i16 20
  %rv = trunc i64 %pv to i16
  store i16 %rv, i16* %r2

  %s2 = getelementptr i8, i8* %s, i8 -20
  %sv = trunc i64 %pv to i8
  store i8 %sv, i8* %s2

  ret void
}

; CHECK: promote_shifts
; CHECK: ldub [%i0], [[R:%[goli][0-7]]]
; CHECK: sll [[R]], [[R]], %i0
define i8 @promote_shifts(i8* %p) {
  %L24 = load i8, i8* %p
  %L32 = load i8, i8* %p
  %B36 = shl i8 %L24, %L32
  ret i8 %B36
}

; CHECK: multiply
; CHECK: mulx %i0, %i1, %i0
define i64 @multiply(i64 %a, i64 %b) {
  %r = mul i64 %a, %b
  ret i64 %r
}

; CHECK: signed_divide
; CHECK: sdivx %i0, %i1, %i0
define i64 @signed_divide(i64 %a, i64 %b) {
  %r = sdiv i64 %a, %b
  ret i64 %r
}

; CHECK: unsigned_divide
; CHECK: udivx %i0, %i1, %i0
define i64 @unsigned_divide(i64 %a, i64 %b) {
  %r = udiv i64 %a, %b
  ret i64 %r
}

define void @access_fi() {
entry:
  %b = alloca [32 x i8], align 1
  %arraydecay = getelementptr inbounds [32 x i8], [32 x i8]* %b, i64 0, i64 0
  call void @g(i8* %arraydecay) #2
  ret void
}

declare void @g(i8*)

; CHECK: expand_setcc
; CHECK: movrgz %i0, 1,
define i32 @expand_setcc(i64 %a) {
  %cond = icmp sle i64 %a, 0
  %cast2 = zext i1 %cond to i32
  %RV = sub i32 1, %cast2
  ret i32 %RV
}

; CHECK: spill_i64
; CHECK: stx
; CHECK: ldx
define i64 @spill_i64(i64 %x) {
  call void asm sideeffect "", "~{i0},~{i1},~{i2},~{i3},~{i4},~{i5},~{o0},~{o1},~{o2},~{o3},~{o4},~{o5},~{o7},~{l0},~{l1},~{l2},~{l3},~{l4},~{l5},~{l6},~{l7},~{g1},~{g2},~{g3},~{g4},~{g5},~{g6},~{g7}"()
  ret i64 %x
}

; CHECK: bitcast_i64_f64
; CHECK: std
; CHECK: ldx
define i64 @bitcast_i64_f64(double %x) {
  %y = bitcast double %x to i64
  ret i64 %y
}

; CHECK: bitcast_f64_i64
; CHECK: stx
; CHECK: ldd
define double @bitcast_f64_i64(i64 %x) {
  %y = bitcast i64 %x to double
  ret double %y
}

; CHECK-LABEL: store_zero:
; CHECK: stx %g0, [%i0]
; CHECK: stx %g0, [%i1+8]

; OPT-LABEL:  store_zero:
; OPT:  stx %g0, [%o0]
; OPT:  stx %g0, [%o1+8]
define i64 @store_zero(i64* nocapture %a, i64* nocapture %b) {
entry:
  store i64 0, i64* %a, align 8
  %0 = getelementptr inbounds i64, i64* %b, i32 1
  store i64 0, i64* %0, align 8
  ret i64 0
}

; CHECK-LABEL: bit_ops
; CHECK:       popc

; OPT-LABEL: bit_ops
; OPT:       popc

define i64 @bit_ops(i64 %arg) {
entry:
  %0 = tail call i64 @llvm.ctpop.i64(i64 %arg)
  %1 = tail call i64 @llvm.ctlz.i64(i64 %arg, i1 true)
  %2 = tail call i64 @llvm.cttz.i64(i64 %arg, i1 true)
  %3 = tail call i64 @llvm.bswap.i64(i64 %arg)
  %4 = add i64 %0, %1
  %5 = add i64 %2, %3
  %6 = add i64 %4, %5
  ret i64 %6
}

declare i64 @llvm.ctpop.i64(i64) nounwind readnone
declare i64 @llvm.ctlz.i64(i64, i1) nounwind readnone
declare i64 @llvm.cttz.i64(i64, i1) nounwind readnone
declare i64 @llvm.bswap.i64(i64) nounwind readnone
