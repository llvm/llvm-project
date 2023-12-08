; RUN: llc -march=hexagon -O0 < %s | FileCheck %s
; RUN: llc -march=hexagon -O0 < %s | FileCheck -check-prefix=CHECK-CALL %s
; Hexagon Programmer's Reference Manual 11.10.7 XTYPE/PRED

; CHECK-CALL-NOT: call

; Compare byte
declare i32 @llvm.hexagon.A4.cmpbgt(i32, i32)
define i32 @A4_cmpbgt(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A4.cmpbgt(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = cmpb.gt({{.*}},{{.*}})

declare i32 @llvm.hexagon.A4.cmpbeq(i32, i32)
define i32 @A4_cmpbeq(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A4.cmpbeq(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = cmpb.eq({{.*}},{{.*}})

declare i32 @llvm.hexagon.A4.cmpbgtu(i32, i32)
define i32 @A4_cmpbgtu(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A4.cmpbgtu(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = cmpb.gtu({{.*}},{{.*}})

declare i32 @llvm.hexagon.A4.cmpbgti(i32, i32)
define i32 @A4_cmpbgti(i32 %a) {
  %z = call i32 @llvm.hexagon.A4.cmpbgti(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: = cmpb.gt({{.*}},#0)

declare i32 @llvm.hexagon.A4.cmpbeqi(i32, i32)
define i32 @A4_cmpbeqi(i32 %a) {
  %z = call i32 @llvm.hexagon.A4.cmpbeqi(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: = cmpb.eq({{.*}},#0)

declare i32 @llvm.hexagon.A4.cmpbgtui(i32, i32)
define i32 @A4_cmpbgtui(i32 %a) {
  %z = call i32 @llvm.hexagon.A4.cmpbgtui(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: = cmpb.gtu({{.*}},#0)

; Compare half
declare i32 @llvm.hexagon.A4.cmphgt(i32, i32)
define i32 @A4_cmphgt(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A4.cmphgt(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = cmph.gt({{.*}},{{.*}})

declare i32 @llvm.hexagon.A4.cmpheq(i32, i32)
define i32 @A4_cmpheq(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A4.cmpheq(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = cmph.eq({{.*}},{{.*}})

declare i32 @llvm.hexagon.A4.cmphgtu(i32, i32)
define i32 @A4_cmphgtu(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A4.cmphgtu(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = cmph.gtu({{.*}},{{.*}})

declare i32 @llvm.hexagon.A4.cmphgti(i32, i32)
define i32 @A4_cmphgti(i32 %a) {
  %z = call i32 @llvm.hexagon.A4.cmphgti(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: = cmph.gt({{.*}},#0)

declare i32 @llvm.hexagon.A4.cmpheqi(i32, i32)
define i32 @A4_cmpheqi(i32 %a) {
  %z = call i32 @llvm.hexagon.A4.cmpheqi(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: = cmph.eq({{.*}},#0)

declare i32 @llvm.hexagon.A4.cmphgtui(i32, i32)
define i32 @A4_cmphgtui(i32 %a) {
  %z = call i32 @llvm.hexagon.A4.cmphgtui(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: = cmph.gtu({{.*}},#0)

; Compare doublewords
declare i32 @llvm.hexagon.C2.cmpgtp(i64, i64)
define i32 @C2_cmpgtp(i64 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.C2.cmpgtp(i64 %a, i64 %b)
  ret i32 %z
}
; CHECK: = cmp.gt({{.*}},{{.*}})

declare i32 @llvm.hexagon.C2.cmpeqp(i64, i64)
define i32 @C2_cmpeqp(i64 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.C2.cmpeqp(i64 %a, i64 %b)
  ret i32 %z
}
; CHECK: = cmp.eq({{.*}},{{.*}})

declare i32 @llvm.hexagon.C2.cmpgtup(i64, i64)
define i32 @C2_cmpgtup(i64 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.C2.cmpgtup(i64 %a, i64 %b)
  ret i32 %z
}
; CHECK: = cmp.gtu({{.*}},{{.*}})

; Compare bitmask
declare i32 @llvm.hexagon.C2.bitsclri(i32, i32)
define i32 @C2_bitsclri(i32 %a) {
  %z = call i32 @llvm.hexagon.C2.bitsclri(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: = bitsclr({{.*}},#0)

declare i32 @llvm.hexagon.C4.nbitsclri(i32, i32)
define i32 @C4_nbitsclri(i32 %a) {
  %z = call i32 @llvm.hexagon.C4.nbitsclri(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: = !bitsclr({{.*}},#0)

declare i32 @llvm.hexagon.C2.bitsset(i32, i32)
define i32 @C2_bitsset(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.C2.bitsset(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = bitsset({{.*}},{{.*}})

declare i32 @llvm.hexagon.C4.nbitsset(i32, i32)
define i32 @C4_nbitsset(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.C4.nbitsset(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = !bitsset({{.*}},{{.*}})

declare i32 @llvm.hexagon.C2.bitsclr(i32, i32)
define i32 @C2_bitsclr(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.C2.bitsclr(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = bitsclr({{.*}},{{.*}})

declare i32 @llvm.hexagon.C4.nbitsclr(i32, i32)
define i32 @C4_nbitsclr(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.C4.nbitsclr(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = !bitsclr({{.*}},{{.*}})

; Mask generate from predicate
declare i64 @llvm.hexagon.C2.mask(i32)
define i64 @C2_mask(i32 %a) {
  %z = call i64 @llvm.hexagon.C2.mask(i32 %a)
  ret i64 %z
}
; CHECK: = mask({{.*}})

; Check for TLB match
declare i32 @llvm.hexagon.A4.tlbmatch(i64, i32)
define i32 @A4_tlbmatch(i64 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.A4.tlbmatch(i64 %a, i32 %b)
  ret i32 %z
}
; CHECK: = tlbmatch({{.*}},{{.*}})

; Test bit
declare i32 @llvm.hexagon.S2.tstbit.i(i32, i32)
define i32 @S2_tstbit_i(i32 %a) {
  %z = call i32 @llvm.hexagon.S2.tstbit.i(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: = tstbit({{.*}},#0)

declare i32 @llvm.hexagon.S4.ntstbit.i(i32, i32)
define i32 @S4_ntstbit_i(i32 %a) {
  %z = call i32 @llvm.hexagon.S4.ntstbit.i(i32 %a, i32 0)
  ret i32 %z
}
; CHECK: = !tstbit({{.*}},#0)

declare i32 @llvm.hexagon.S2.tstbit.r(i32, i32)
define i32 @S2_tstbit_r(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S2.tstbit.r(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = tstbit({{.*}},{{.*}})

declare i32 @llvm.hexagon.S4.ntstbit.r(i32, i32)
define i32 @S4_ntstbit_r(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.S4.ntstbit.r(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = !tstbit({{.*}},{{.*}})

; Vector compare halfwords
declare i32 @llvm.hexagon.A2.vcmpheq(i64, i64)
define i32 @A2_vcmpheq(i64 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.A2.vcmpheq(i64 %a, i64 %b)
  ret i32 %z
}
; CHECK: = vcmph.eq({{.*}},{{.*}})

declare i32 @llvm.hexagon.A2.vcmphgt(i64, i64)
define i32 @A2_vcmphgt(i64 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.A2.vcmphgt(i64 %a, i64 %b)
  ret i32 %z
}
; CHECK: = vcmph.gt({{.*}},{{.*}})

declare i32 @llvm.hexagon.A2.vcmphgtu(i64, i64)
define i32 @A2_vcmphgtu(i64 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.A2.vcmphgtu(i64 %a, i64 %b)
  ret i32 %z
}
; CHECK: = vcmph.gtu({{.*}},{{.*}})

declare i32 @llvm.hexagon.A4.vcmpheqi(i64, i32)
define i32 @A4_vcmpheqi(i64 %a) {
  %z = call i32 @llvm.hexagon.A4.vcmpheqi(i64 %a, i32 0)
  ret i32 %z
}
; CHECK: = vcmph.eq({{.*}},#0)

declare i32 @llvm.hexagon.A4.vcmphgti(i64, i32)
define i32 @A4_vcmphgti(i64 %a) {
  %z = call i32 @llvm.hexagon.A4.vcmphgti(i64 %a, i32 0)
  ret i32 %z
}
; CHECK: = vcmph.gt({{.*}},#0)

declare i32 @llvm.hexagon.A4.vcmphgtui(i64, i32)
define i32 @A4_vcmphgtui(i64 %a) {
  %z = call i32 @llvm.hexagon.A4.vcmphgtui(i64 %a, i32 0)
  ret i32 %z
}
; CHECK: = vcmph.gtu({{.*}},#0)

; Vector compare bytes for any match
declare i32 @llvm.hexagon.A4.vcmpbeq.any(i64, i64)
define i32 @A4_vcmpbeq_any(i64 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.A4.vcmpbeq.any(i64 %a, i64 %b)
  ret i32 %z
}
; CHECK: = any8(vcmpb.eq({{.*}},{{.*}}))

; Vector compare bytes
declare i32 @llvm.hexagon.A2.vcmpbeq(i64, i64)
define i32 @A2_vcmpbeq(i64 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.A2.vcmpbeq(i64 %a, i64 %b)
  ret i32 %z
}
; CHECK: = vcmpb.eq({{.*}},{{.*}})

declare i32 @llvm.hexagon.A2.vcmpbgtu(i64, i64)
define i32 @A2_vcmpbgtu(i64 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.A2.vcmpbgtu(i64 %a, i64 %b)
  ret i32 %z
}
; CHECK: = vcmpb.gtu({{.*}},{{.*}})

declare i32 @llvm.hexagon.A4.vcmpbgt(i64, i64)
define i32 @A4_vcmpbgt(i64 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.A4.vcmpbgt(i64 %a, i64 %b)
  ret i32 %z
}
; CHECK: = vcmpb.gt({{.*}},{{.*}})

declare i32 @llvm.hexagon.A4.vcmpbeqi(i64, i32)
define i32 @A4_vcmpbeqi(i64 %a) {
  %z = call i32 @llvm.hexagon.A4.vcmpbeqi(i64 %a, i32 0)
  ret i32 %z
}
; CHECK: = vcmpb.eq({{.*}},#0)

declare i32 @llvm.hexagon.A4.vcmpbgti(i64, i32)
define i32 @A4_vcmpbgti(i64 %a) {
  %z = call i32 @llvm.hexagon.A4.vcmpbgti(i64 %a, i32 0)
  ret i32 %z
}
; CHECK: = vcmpb.gt({{.*}},#0)

declare i32 @llvm.hexagon.A4.vcmpbgtui(i64, i32)
define i32 @A4_vcmpbgtui(i64 %a) {
  %z = call i32 @llvm.hexagon.A4.vcmpbgtui(i64 %a, i32 0)
  ret i32 %z
}
; CHECK: = vcmpb.gtu({{.*}},#0)

; Vector compare words
declare i32 @llvm.hexagon.A2.vcmpweq(i64, i64)
define i32 @A2_vcmpweq(i64 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.A2.vcmpweq(i64 %a, i64 %b)
  ret i32 %z
}
; CHECK: = vcmpw.eq({{.*}},{{.*}})

declare i32 @llvm.hexagon.A2.vcmpwgt(i64, i64)
define i32 @A2_vcmpwgt(i64 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.A2.vcmpwgt(i64 %a, i64 %b)
  ret i32 %z
}
; CHECK: = vcmpw.gt({{.*}},{{.*}})

declare i32 @llvm.hexagon.A2.vcmpwgtu(i64, i64)
define i32 @A2_vcmpwgtu(i64 %a, i64 %b) {
  %z = call i32 @llvm.hexagon.A2.vcmpwgtu(i64 %a, i64 %b)
  ret i32 %z
}
; CHECK: = vcmpw.gtu({{.*}},{{.*}})

declare i32 @llvm.hexagon.A4.vcmpweqi(i64, i32)
define i32 @A4_vcmpweqi(i64 %a) {
  %z = call i32 @llvm.hexagon.A4.vcmpweqi(i64 %a, i32 0)
  ret i32 %z
}
; CHECK: = vcmpw.eq({{.*}},#0)

declare i32 @llvm.hexagon.A4.vcmpwgti(i64, i32)
define i32 @A4_vcmpwgti(i64 %a) {
  %z = call i32 @llvm.hexagon.A4.vcmpwgti(i64 %a, i32 0)
  ret i32 %z
}
; CHECK: = vcmpw.gt({{.*}},#0)

declare i32 @llvm.hexagon.A4.vcmpwgtui(i64, i32)
define i32 @A4_vcmpwgtui(i64 %a) {
  %z = call i32 @llvm.hexagon.A4.vcmpwgtui(i64 %a, i32 0)
  ret i32 %z
}
; CHECK: = vcmpw.gtu({{.*}},#0)

; Viterbi pack even and odd predicate bitsclr
declare i32 @llvm.hexagon.C2.vitpack(i32, i32)
define i32 @C2_vitpack(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.C2.vitpack(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = vitpack({{.*}},{{.*}})

; Vector mux
declare i64 @llvm.hexagon.C2.vmux(i32, i64, i64)
define i64 @C2_vmux(i32 %a, i64 %b, i64 %c) {
  %z = call i64 @llvm.hexagon.C2.vmux(i32 %a, i64 %b, i64 %c)
  ret i64 %z
}
; CHECK: = vmux({{.*}},{{.*}},{{.*}})
