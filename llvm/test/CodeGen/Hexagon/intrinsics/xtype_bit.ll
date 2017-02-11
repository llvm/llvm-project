; RUN: llc -march=hexagon -O0 < %s | FileCheck %s
; RUN: llc -march=hexagon -O0 < %s | FileCheck -check-prefix=CHECK-CALL %s
; Hexagon Programmer's Reference Manual 11.10.2 XTYPE/BIT

; CHECK-CALL-NOT: call

; Count leading
declare i32 @llvm.hexagon.S2.clbp(i64)
define i32 @S2_clbp(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.clbp(i64 %a)
  ret i32 %z
}
; CHECK: = clb({{.*}})

declare i32 @llvm.hexagon.S2.cl0p(i64)
define i32 @S2_cl0p(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.cl0p(i64 %a)
  ret i32 %z
}
; CHECK: = cl0({{.*}})

declare i32 @llvm.hexagon.S2.cl1p(i64)
define i32 @S2_cl1p(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.cl1p(i64 %a)
  ret i32 %z
}
; CHECK: = cl1({{.*}})

declare i32 @llvm.hexagon.S4.clbpnorm(i64)
define i32 @S4_clbpnorm(i64 %a) {
  %z = call i32 @llvm.hexagon.S4.clbpnorm(i64 %a)
  ret i32 %z
}
; CHECK: = normamt({{.*}})

declare i32 @llvm.hexagon.S4.clbpaddi(i64, i32)
define i32 @S4_clbpaddi(i64 %a) {
  %z = call i32 @llvm.hexagon.S4.clbpaddi(i64 %a, i32 0)
  ret i32 %z
}
; CHECK: = add(clb({{.*}}),#0)

declare i32 @llvm.hexagon.S4.clbaddi(i32, i32)
define i32 @S4_clbaddi(i32 %a) {
  %z = call i32 @llvm.hexagon.S4.clbaddi(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: = add(clb({{.*}}),#0)

declare i32 @llvm.hexagon.S2.cl0(i32)
define i32 @S2_cl0(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.cl0(i32 %a)
  ret i32 %z
}
; CHECK: = cl0({{.*}})

declare i32 @llvm.hexagon.S2.cl1(i32)
define i32 @S2_cl1(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.cl1(i32 %a)
  ret i32 %z
}
; CHECK: = cl1({{.*}})

declare i32 @llvm.hexagon.S2.clbnorm(i32)
define i32 @S4_clbnorm(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.clbnorm(i32 %a)
  ret i32 %z
}
; CHECK: = normamt({{.*}})

; Count population
declare i32 @llvm.hexagon.S5.popcountp(i64)
define i32 @S5_popcountp(i64 %a) {
  %z = call i32 @llvm.hexagon.S5.popcountp(i64 %a)
  ret i32 %z
}
; CHECK: = popcount({{.*}})

; Count trailing
declare i32 @llvm.hexagon.S2.ct0p(i64)
define i32 @S2_ct0p(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.ct0p(i64 %a)
  ret i32 %z
}
; CHECK: = ct0({{.*}})

declare i32 @llvm.hexagon.S2.ct1p(i64)
define i32 @S2_ct1p(i64 %a) {
  %z = call i32 @llvm.hexagon.S2.ct1p(i64 %a)
  ret i32 %z
}
; CHECK: = ct1({{.*}})

declare i32 @llvm.hexagon.S2.ct0(i32)
define i32 @S2_ct0(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.ct0(i32 %a)
  ret i32 %z
}
; CHECK: = ct0({{.*}})

declare i32 @llvm.hexagon.S2.ct1(i32)
define i32 @S2_ct1(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.ct1(i32 %a)
  ret i32 %z
}
; CHECK: = ct1({{.*}})

; Extract bitfield
declare i64 @llvm.hexagon.S2.extractup(i64, i32, i32)
define i64 @S2_extractup(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.extractup(i64 %a, i32 0, i32 0)
  ret i64 %z
}
; CHECK: = extractu({{.*}},#0,#0)

declare i64 @llvm.hexagon.S4.extractp(i64, i32, i32)
define i64 @S2_extractp(i64 %a) {
  %z = call i64 @llvm.hexagon.S4.extractp(i64 %a, i32 0, i32 0)
  ret i64 %z
}
; CHECK: = extract({{.*}},#0,#0)

declare i32 @llvm.hexagon.S2.extractu(i32, i32, i32)
define i32 @S2_extractu(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.extractu(i32 %a, i32 0, i32 0)
  ret i32 %z
}
; CHECK: = extractu({{.*}},#0,#0)

declare i32 @llvm.hexagon.S4.extract(i32, i32, i32)
define i32 @S2_extract(i32 %a) {
  %z = call i32 @llvm.hexagon.S4.extract(i32 %a, i32 0, i32 0)
  ret i32 %z
}
; CHECK: = extract({{.*}},#0,#0)

declare i64 @llvm.hexagon.S2.extractup.rp(i64, i64)
define i64 @S2_extractup_rp(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.extractup.rp(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = extractu({{.*}},{{.*}})

declare i64 @llvm.hexagon.S4.extractp.rp(i64, i64)
define i64 @S4_extractp_rp(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S4.extractp.rp(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = extract({{.*}},{{.*}})

declare i32 @llvm.hexagon.S2.extractu.rp(i32, i64)
define i32 @S2_extractu_rp(i32 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.S2.extractu.rp(i32 %a, i64 %b)
  ret i32 %z
}
; CHECK: = extractu({{.*}},{{.*}})

declare i32 @llvm.hexagon.S4.extract.rp(i32, i64)
define i32 @S4_extract_rp(i32 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.S4.extract.rp(i32 %a, i64 %b)
  ret i32 %z
}
; CHECK: = extract({{.*}},{{.*}})

; Insert bitfield
declare i64 @llvm.hexagon.S2.insertp(i64, i64, i32, i32)
define i64 @S2_insertp(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.insertp(i64 %a, i64 %b, i32 0, i32 0)
  ret i64 %z
}
; CHECK: = insert({{.*}},#0,#0)

declare i32 @llvm.hexagon.S2.insert(i32, i32, i32, i32)
define i32 @S2_insert(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.insert(i32 %a, i32 %b, i32 0, i32 0)
  ret i32 %z
}
; CHECK: = insert({{.*}},#0,#0)

declare i32 @llvm.hexagon.S2.insert.rp(i32, i32, i64)
define i32 @S2_insert_rp(i32 %a, i32 %b, i64 %c) {
  %z = call i32 @llvm.hexagon.S2.insert.rp(i32 %a, i32 %b, i64 %c)
  ret i32 %z
}
; CHECK: = insert({{.*}},{{.*}})

declare i64 @llvm.hexagon.S2.insertp.rp(i64, i64, i64)
define i64 @S2_insertp_rp(i64 %a, i64 %b, i64 %c) {
  %z = call i64 @llvm.hexagon.S2.insertp.rp(i64 %a, i64 %b, i64 %c)
  ret i64 %z
}
; CHECK: = insert({{.*}},r5:4)

; Interleave/deinterleave
declare i64 @llvm.hexagon.S2.deinterleave(i64)
define i64 @S2_deinterleave(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.deinterleave(i64 %a)
  ret i64 %z
}
; CHECK: = deinterleave({{.*}})

declare i64 @llvm.hexagon.S2.interleave(i64)
define i64 @S2_interleave(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.interleave(i64 %a)
  ret i64 %z
}
; CHECK: = interleave({{.*}})

; Linear feedback-shift operation
declare i64 @llvm.hexagon.S2.lfsp(i64, i64)
define i64 @S2_lfsp(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S2.lfsp(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = lfs({{.*}},{{.*}})

; Masked parity
declare i32 @llvm.hexagon.S2.parityp(i64, i64)
define i32 @S2_parityp(i64 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.S2.parityp(i64 %a, i64 %b)
  ret i32 %z
}
; CHECK: = parity({{.*}},{{.*}})

declare i32 @llvm.hexagon.S4.parity(i32, i32)
define i32 @S4_parity(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S4.parity(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = parity({{.*}},{{.*}})

; Bit reverse
declare i64 @llvm.hexagon.S2.brevp(i64)
define i64 @S2_brevp(i64 %a) {
  %z = call i64 @llvm.hexagon.S2.brevp(i64 %a)
  ret i64 %z
}
; CHECK: = brev({{.*}})

declare i32 @llvm.hexagon.S2.brev(i32)
define i32 @S2_brev(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.brev(i32 %a)
  ret i32 %z
}
; CHECK: = brev({{.*}})

; Set/clear/toggle bit
declare i32 @llvm.hexagon.S2.setbit.i(i32, i32)
define i32 @S2_setbit_i(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.setbit.i(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: = setbit({{.*}},#0)

declare i32 @llvm.hexagon.S2.clrbit.i(i32, i32)
define i32 @S2_clrbit_i(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.clrbit.i(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: = clrbit({{.*}},#0)

declare i32 @llvm.hexagon.S2.togglebit.i(i32, i32)
define i32 @S2_togglebit_i(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.togglebit.i(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: = togglebit({{.*}},#0)

declare i32 @llvm.hexagon.S2.setbit.r(i32, i32)
define i32 @S2_setbit_r(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.setbit.r(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = setbit({{.*}},{{.*}})

declare i32 @llvm.hexagon.S2.clrbit.r(i32, i32)
define i32 @S2_clrbit_r(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.clrbit.r(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = clrbit({{.*}},{{.*}})

declare i32 @llvm.hexagon.S2.togglebit.r(i32, i32)
define i32 @S2_togglebit_r(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.togglebit.r(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = togglebit({{.*}},{{.*}})

; Split bitfield
declare i64 @llvm.hexagon.A4.bitspliti(i32, i32)
define i64 @A4_bitspliti(i32 %a) {
  %z = call i64 @llvm.hexagon.A4.bitspliti(i32 %a, i32 0)
  ret i64 %z
}
; CHECK: = bitsplit({{.*}},#0)

declare i64 @llvm.hexagon.A4.bitsplit(i32, i32)
define i64 @A4_bitsplit(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.A4.bitsplit(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: = bitsplit({{.*}},{{.*}})

; Table index
declare i32 @llvm.hexagon.S2.tableidxb.goodsyntax(i32, i32, i32, i32)
define i32 @S2_tableidxb_goodsyntax(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.tableidxb.goodsyntax(i32 %a, i32 %b, i32 0, i32 0)
  ret i32 %z
}
; CHECK: = tableidxb({{.*}},#0,#0)

declare i32 @llvm.hexagon.S2.tableidxh.goodsyntax(i32, i32, i32, i32)
define i32 @S2_tableidxh_goodsyntax(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.tableidxh.goodsyntax(i32 %a, i32 %b, i32 0, i32 0)
  ret i32 %z
}
; CHECK: = tableidxh({{.*}},#0,#-1)

declare i32 @llvm.hexagon.S2.tableidxw.goodsyntax(i32, i32, i32, i32)
define i32 @S2_tableidxw_goodsyntax(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.tableidxw.goodsyntax(i32 %a, i32 %b, i32 0, i32 0)
  ret i32 %z
}
; CHECK: = tableidxw({{.*}},#0,#-2)

declare i32 @llvm.hexagon.S2.tableidxd.goodsyntax(i32, i32, i32, i32)
define i32 @S2_tableidxd_goodsyntax(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.tableidxd.goodsyntax(i32 %a, i32 %b, i32 0, i32 0)
  ret i32 %z
}
; CHECK: = tableidxd({{.*}},#0,#-3)
