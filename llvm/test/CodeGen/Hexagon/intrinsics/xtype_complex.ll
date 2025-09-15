; RUN: llc -mtriple=hexagon -O0 < %s | FileCheck %s
; RUN: llc -mtriple=hexagon -O0 < %s | FileCheck -check-prefix=CHECK-CALL %s
; Hexagon Programmer's Reference Manual 11.10.3 XTYPE/COMPLEX

; CHECK-CALL-NOT: call

; Complex add/sub halfwords
declare i64 @llvm.hexagon.S4.vxaddsubh(i64, i64)
define i64 @S4_vxaddsubh(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S4.vxaddsubh(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = vxaddsubh({{.*}},{{.*}}):sat

declare i64 @llvm.hexagon.S4.vxsubaddh(i64, i64)
define i64 @S4_vxsubaddh(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S4.vxsubaddh(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = vxsubaddh({{.*}},{{.*}}):sat

declare i64 @llvm.hexagon.S4.vxaddsubhr(i64, i64)
define i64 @S4_vxaddsubhr(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S4.vxaddsubhr(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = vxaddsubh({{.*}},{{.*}}):rnd:>>1:sat

declare i64 @llvm.hexagon.S4.vxsubaddhr(i64, i64)
define i64 @S4_vxsubaddhr(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S4.vxsubaddhr(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = vxsubaddh({{.*}},{{.*}}):rnd:>>1:sat

; Complex add/sub words
declare i64 @llvm.hexagon.S4.vxaddsubw(i64, i64)
define i64 @S4_vxaddsubw(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S4.vxaddsubw(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = vxaddsubw({{.*}},{{.*}}):sat

declare i64 @llvm.hexagon.S4.vxsubaddw(i64, i64)
define i64 @S4_vxsubaddw(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.S4.vxsubaddw(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = vxsubaddw({{.*}},{{.*}}):sat

; Complex multiply
declare i64 @llvm.hexagon.M2.cmpys.s0(i32, i32)
define i64 @M2_cmpys_s0(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.cmpys.s0(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: = cmpy({{.*}},{{.*}}):sat

declare i64 @llvm.hexagon.M2.cmpys.s1(i32, i32)
define i64 @M2_cmpys_s1(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.cmpys.s1(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: = cmpy({{.*}},{{.*}}):<<1:sat

declare i64 @llvm.hexagon.M2.cmpysc.s0(i32, i32)
define i64 @M2_cmpysc_s0(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.cmpysc.s0(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: = cmpy({{.*}},{{.*}}*):sat

declare i64 @llvm.hexagon.M2.cmpysc.s1(i32, i32)
define i64 @M2_cmpysc_s1(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.cmpysc.s1(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: = cmpy({{.*}},{{.*}}*):<<1:sat

declare i64 @llvm.hexagon.M2.cmacs.s0(i64, i32, i32)
define i64 @M2_cmacs_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.cmacs.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: += cmpy({{.*}},{{.*}}):sat

declare i64 @llvm.hexagon.M2.cmacs.s1(i64, i32, i32)
define i64 @M2_cmacs_s1(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.cmacs.s1(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: += cmpy({{.*}},{{.*}}):<<1:sat

declare i64 @llvm.hexagon.M2.cnacs.s0(i64, i32, i32)
define i64 @M2_cnacs_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.cnacs.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: -= cmpy({{.*}},{{.*}}):sat

declare i64 @llvm.hexagon.M2.cnacs.s1(i64, i32, i32)
define i64 @M2_cnacs_s1(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.cnacs.s1(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: -= cmpy({{.*}},{{.*}}):<<1:sat

declare i64 @llvm.hexagon.M2.cmacsc.s0(i64, i32, i32)
define i64 @M2_cmacsc_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.cmacsc.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: += cmpy({{.*}},{{.*}}*):sat

declare i64 @llvm.hexagon.M2.cmacsc.s1(i64, i32, i32)
define i64 @M2_cmacsc_s1(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.cmacsc.s1(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: += cmpy({{.*}},{{.*}}*):<<1:sat

declare i64 @llvm.hexagon.M2.cnacsc.s0(i64, i32, i32)
define i64 @M2_cnacsc_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.cnacsc.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: -= cmpy({{.*}},{{.*}}*):sat

declare i64 @llvm.hexagon.M2.cnacsc.s1(i64, i32, i32)
define i64 @M2_cnacsc_s1(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.cnacsc.s1(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: -= cmpy({{.*}},{{.*}}*):<<1:sat

; Complex multiply real or imaginary
declare i64 @llvm.hexagon.M2.cmpyi.s0(i32, i32)
define i64 @M2_cmpyi_s0(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.cmpyi.s0(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: = cmpyi({{.*}},{{.*}})

declare i64 @llvm.hexagon.M2.cmpyr.s0(i32, i32)
define i64 @M2_cmpyr_s0(i32 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.M2.cmpyr.s0(i32 %a, i32 %b)
  ret i64 %z
}
; CHECK: = cmpyr({{.*}},{{.*}})

declare i64 @llvm.hexagon.M2.cmaci.s0(i64, i32, i32)
define i64 @M2_cmaci_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.cmaci.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: += cmpyi({{.*}},{{.*}})

declare i64 @llvm.hexagon.M2.cmacr.s0(i64, i32, i32)
define i64 @M2_cmacr_s0(i64 %a, i32 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.M2.cmacr.s0(i64 %a, i32 %b, i32 %c)
  ret i64 %z
}
; CHECK: += cmpyr({{.*}},{{.*}})

; Complex multiply with round and pack
declare i32 @llvm.hexagon.M2.cmpyrs.s0(i32, i32)
define i32 @M2_cmpyrs_s0(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.cmpyrs.s0(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = cmpy({{.*}},{{.*}}):rnd:sat

declare i32 @llvm.hexagon.M2.cmpyrs.s1(i32, i32)
define i32 @M2_cmpyrs_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.cmpyrs.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = cmpy({{.*}},{{.*}}):<<1:rnd:sat

declare i32 @llvm.hexagon.M2.cmpyrsc.s0(i32, i32)
define i32 @M2_cmpyrsc_s0(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.cmpyrsc.s0(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = cmpy({{.*}},{{.*}}*):rnd:sat

declare i32 @llvm.hexagon.M2.cmpyrsc.s1(i32, i32)
define i32 @M2_cmpyrsc_s1(i32 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M2.cmpyrsc.s1(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = cmpy({{.*}},{{.*}}*):<<1:rnd:sat

; Complex multiply 32x16
declare i32 @llvm.hexagon.M4.cmpyi.wh(i64, i32)
define i32 @M4_cmpyi_wh(i64 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M4.cmpyi.wh(i64 %a, i32 %b)
  ret i32 %z
}
; CHECK: = cmpyiwh({{.*}},{{.*}}):<<1:rnd:sat

declare i32 @llvm.hexagon.M4.cmpyi.whc(i64, i32)
define i32 @M4_cmpyi_whc(i64 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M4.cmpyi.whc(i64 %a, i32 %b)
  ret i32 %z
}
; CHECK: = cmpyiwh({{.*}},{{.*}}*):<<1:rnd:sat

declare i32 @llvm.hexagon.M4.cmpyr.wh(i64, i32)
define i32 @M4_cmpyr_wh(i64 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M4.cmpyr.wh(i64 %a, i32 %b)
  ret i32 %z
}
; CHECK: = cmpyrwh({{.*}},{{.*}}):<<1:rnd:sat

declare i32 @llvm.hexagon.M4.cmpyr.whc(i64, i32)
define i32 @M4_cmpyr_whc(i64 %a, i32 %b) {
  %z = call i32 @llvm.hexagon.M4.cmpyr.whc(i64 %a, i32 %b)
  ret i32 %z
}
; CHECK: = cmpyrwh({{.*}},{{.*}}*):<<1:rnd:sat

; Vector complex multiply real or imaginary
declare i64 @llvm.hexagon.M2.vcmpy.s0.sat.r(i64, i64)
define i64 @M2_vcmpy_s0_sat_r(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.M2.vcmpy.s0.sat.r(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = vcmpyr({{.*}},{{.*}}):sat

declare i64 @llvm.hexagon.M2.vcmpy.s1.sat.r(i64, i64)
define i64 @M2_vcmpy_s1_sat_r(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.M2.vcmpy.s1.sat.r(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = vcmpyr({{.*}},{{.*}}):<<1:sat

declare i64 @llvm.hexagon.M2.vcmpy.s0.sat.i(i64, i64)
define i64 @M2_vcmpy_s0_sat_i(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.M2.vcmpy.s0.sat.i(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = vcmpyi({{.*}},{{.*}}):sat

declare i64 @llvm.hexagon.M2.vcmpy.s1.sat.i(i64, i64)
define i64 @M2_vcmpy_s1_sat_i(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.M2.vcmpy.s1.sat.i(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = vcmpyi({{.*}},{{.*}}):<<1:sat

declare i64 @llvm.hexagon.M2.vcmac.s0.sat.r(i64, i64, i64)
define i64 @M2_vcmac_s0_sat_r(i64 %a, i64 %b, i64 %c) {
  %z = call i64 @llvm.hexagon.M2.vcmac.s0.sat.r(i64 %a, i64 %b, i64 %c)
  ret i64 %z
}
; CHECK: += vcmpyr({{.*}},r5:4):sat

declare i64 @llvm.hexagon.M2.vcmac.s0.sat.i(i64, i64, i64)
define i64 @M2_vcmac_s0_sat_i(i64 %a, i64 %b, i64 %c) {
  %z = call i64 @llvm.hexagon.M2.vcmac.s0.sat.i(i64 %a, i64 %b, i64 %c)
  ret i64 %z
}
; CHECK: += vcmpyi({{.*}},r5:4):sat

; Vector complex conjugate
declare i64 @llvm.hexagon.A2.vconj(i64)
define i64 @A2_vconj(i64 %a) {
  %z = call i64 @llvm.hexagon.A2.vconj(i64 %a)
  ret i64 %z
}
; CHECK: = vconj({{.*}}):sat

; Vector complex rotate
declare i64 @llvm.hexagon.S2.vcrotate(i64, i32)
define i64 @S2_vcrotate(i64 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.S2.vcrotate(i64 %a, i32 %b)
  ret i64 %z
}
; CHECK: = vcrotate({{.*}},{{.*}})

; Vector reduce complex multiply real or imaginary
declare i64 @llvm.hexagon.M2.vrcmpyi.s0(i64, i64)
define i64 @M2_vrcmpyi_s0(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.M2.vrcmpyi.s0(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = vrcmpyi({{.*}},{{.*}})

declare i64 @llvm.hexagon.M2.vrcmpyr.s0(i64, i64)
define i64 @M2_vrcmpyr_s0(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.M2.vrcmpyr.s0(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = vrcmpyr({{.*}},{{.*}})

declare i64 @llvm.hexagon.M2.vrcmpyi.s0c(i64, i64)
define i64 @M2_vrcmpyi_s0c(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.M2.vrcmpyi.s0c(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = vrcmpyi({{.*}},{{.*}}*)

declare i64 @llvm.hexagon.M2.vrcmpyr.s0c(i64, i64)
define i64 @M2_vrcmpyr_s0c(i64 %a, i64 %b) {
  %z = call i64 @llvm.hexagon.M2.vrcmpyr.s0c(i64 %a, i64 %b)
  ret i64 %z
}
; CHECK: = vrcmpyr({{.*}},{{.*}}*)

declare i64 @llvm.hexagon.M2.vrcmaci.s0(i64, i64, i64)
define i64 @M2_vrcmaci_s0(i64 %a, i64 %b, i64 %c) {
  %z = call i64 @llvm.hexagon.M2.vrcmaci.s0(i64 %a, i64 %b, i64 %c)
  ret i64 %z
}
; CHECK: += vrcmpyi({{.*}},r5:4)

declare i64 @llvm.hexagon.M2.vrcmacr.s0(i64, i64, i64)
define i64 @M2_vrcmacr_s0(i64 %a, i64 %b, i64 %c) {
  %z = call i64 @llvm.hexagon.M2.vrcmacr.s0(i64 %a, i64 %b, i64 %c)
  ret i64 %z
}
; CHECK: += vrcmpyr({{.*}},r5:4)

declare i64 @llvm.hexagon.M2.vrcmaci.s0c(i64, i64, i64)
define i64 @M2_vrcmaci_s0c(i64 %a, i64 %b, i64 %c) {
  %z = call i64 @llvm.hexagon.M2.vrcmaci.s0c(i64 %a, i64 %b, i64 %c)
  ret i64 %z
}
; CHECK: += vrcmpyi({{.*}},r5:4*)

declare i64 @llvm.hexagon.M2.vrcmacr.s0c(i64, i64, i64)
define i64 @M2_vrcmacr_s0c(i64 %a, i64 %b, i64 %c) {
  %z = call i64 @llvm.hexagon.M2.vrcmacr.s0c(i64 %a, i64 %b, i64 %c)
  ret i64 %z
}
; CHECK: += vrcmpyr({{.*}},r5:4*)

; Vector reduce complex rotate
declare i64 @llvm.hexagon.S4.vrcrotate(i64, i32, i32)
define i64 @S4_vrcrotate(i64 %a, i32 %b) {
  %z = call i64 @llvm.hexagon.S4.vrcrotate(i64 %a, i32 %b, i32 0)
  ret i64 %z
}
; CHECK: = vrcrotate({{.*}},{{.*}},#0)

declare i64 @llvm.hexagon.S4.vrcrotate.acc(i64, i64, i32, i32)
define i64 @S4_vrcrotate_acc(i64 %a, i64 %b, i32 %c) {
  %z = call i64 @llvm.hexagon.S4.vrcrotate.acc(i64 %a, i64 %b, i32 %c, i32 0)
  ret i64 %z
}
; CHECK: += vrcrotate({{.*}},{{.*}},#0)
