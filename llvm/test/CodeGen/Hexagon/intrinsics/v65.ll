; RUN: llc -mv65 -mattr=+hvxv65,hvx-length64b -mtriple=hexagon -O0 < %s | FileCheck %s
; RUN: llc -mv65 -mattr=+hvxv65,hvx-length64b -mtriple=hexagon -O0 < %s | FileCheck -check-prefix=CHECK-CALL %s

; CHECK-CALL-NOT: call

declare i32 @llvm.hexagon.A6.vcmpbeq.notany(i64, i64)
define i32 @A6_vcmpbeq_notany(i64 %a, i64 %b) {
  %c = call i32 @llvm.hexagon.A6.vcmpbeq.notany(i64 %a, i64 %b)
  ret i32 %c
}
; CHECK: = !any8(vcmpb.eq(r1:0,r3:2))

declare <16 x i32> @llvm.hexagon.V6.vabsb(<16 x i32>)
define <16 x i32> @V6_vabsb(<16 x i32> %a) {
  %b = call <16 x i32> @llvm.hexagon.V6.vabsb(<16 x i32> %a)
  ret <16 x i32> %b
}
; CHECK: = vabs(v0.b)

declare <16 x i32> @llvm.hexagon.V6.vabsb.sat(<16 x i32>)
define <16 x i32> @V6_vabsb_sat(<16 x i32> %a) {
  %b = call <16 x i32> @llvm.hexagon.V6.vabsb.sat(<16 x i32> %a)
  ret <16 x i32> %b
}
; CHECK: = vabs(v0.b):sat

declare <16 x i32> @llvm.hexagon.V6.vaslh.acc(<16 x i32>, <16 x i32>, i32)
define <16 x i32> @V6_vaslh_acc(<16 x i32> %a, <16 x i32> %b, i32 %c) {
  %d = call <16 x i32> @llvm.hexagon.V6.vaslh.acc(<16 x i32> %a, <16 x i32> %b, i32 %c)
  ret <16 x i32> %d
}
; CHECK: += vasl(v1.h,r0)

declare <16 x i32> @llvm.hexagon.V6.vasrh.acc(<16 x i32>, <16 x i32>, i32)
define <16 x i32> @V6_vasrh_acc(<16 x i32> %a, <16 x i32> %b, i32 %c) {
  %d = call <16 x i32> @llvm.hexagon.V6.vasrh.acc(<16 x i32> %a, <16 x i32> %b, i32 %c)
  ret <16 x i32> %d
}
; CHECK: += vasr(v1.h,r0)

declare <16 x i32> @llvm.hexagon.V6.vasruwuhsat(<16 x i32>, <16 x i32>, i32)
define <16 x i32> @V6_vasruwuhsat(<16 x i32> %a, <16 x i32> %b, i32 %c) {
  %d = call <16 x i32> @llvm.hexagon.V6.vasruwuhsat(<16 x i32> %a, <16 x i32> %b, i32 %c)
  ret <16 x i32> %d
}
; CHECK: = vasr(v0.uw,v1.uw,r0):sat

declare <16 x i32> @llvm.hexagon.V6.vasruhubsat(<16 x i32>, <16 x i32>, i32)
define <16 x i32> @V6_vasruhubsat(<16 x i32> %a, <16 x i32> %b, i32 %c) {
  %d = call <16 x i32> @llvm.hexagon.V6.vasruhubsat(<16 x i32> %a, <16 x i32> %b, i32 %c)
  ret <16 x i32> %d
}
; CHECK: = vasr(v0.uh,v1.uh,r0):sat

declare <16 x i32> @llvm.hexagon.V6.vasruhubrndsat(<16 x i32>, <16 x i32>, i32)
define <16 x i32> @V6_vasruhubrndsat(<16 x i32> %a, <16 x i32> %b, i32 %c) {
  %d = call <16 x i32> @llvm.hexagon.V6.vasruhubrndsat(<16 x i32> %a, <16 x i32> %b, i32 %c)
  ret <16 x i32> %d
}
; CHECK: = vasr(v0.uh,v1.uh,r0):rnd:sat

declare <16 x i32> @llvm.hexagon.V6.vavguw(<16 x i32>, <16 x i32>)
define <16 x i32> @V6_vavguw(<16 x i32> %a, <16 x i32> %b) {
  %c = call <16 x i32> @llvm.hexagon.V6.vavguw(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %c
}
; CHECK: = vavg(v0.uw,v1.uw)

declare <16 x i32> @llvm.hexagon.V6.vavguwrnd(<16 x i32>, <16 x i32>)
define <16 x i32> @V6_vavguwrnd(<16 x i32> %a, <16 x i32> %b) {
  %c = call <16 x i32> @llvm.hexagon.V6.vavguwrnd(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %c
}
; CHECK: = vavg(v0.uw,v1.uw):rnd

declare <16 x i32> @llvm.hexagon.V6.vavgb(<16 x i32>, <16 x i32>)
define <16 x i32> @V6_vavgb(<16 x i32> %a, <16 x i32> %b) {
  %c = call <16 x i32> @llvm.hexagon.V6.vavgb(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %c
}
; CHECK: = vavg(v0.b,v1.b)

declare <16 x i32> @llvm.hexagon.V6.vavgbrnd(<16 x i32>, <16 x i32>)
define <16 x i32> @V6_vavgbrnd(<16 x i32> %a, <16 x i32> %b) {
  %c = call <16 x i32> @llvm.hexagon.V6.vavgbrnd(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %c
}
; CHECK: = vavg(v0.b,v1.b):rnd

declare <16 x i32> @llvm.hexagon.V6.vnavgb(<16 x i32>, <16 x i32>)
define <16 x i32> @V6_vnavgb(<16 x i32> %a, <16 x i32> %b) {
  %c = call <16 x i32> @llvm.hexagon.V6.vnavgb(<16 x i32> %a, <16 x i32> %b)
  ret <16 x i32> %c
}
; CHECK: = vnavg(v0.b,v1.b)

declare <32 x i32> @llvm.hexagon.V6.vmpabuu(<32 x i32>, i32)
define <32 x i32> @V6_vmpabuu(<32 x i32> %a, i32 %b) {
  %c = call <32 x i32> @llvm.hexagon.V6.vmpabuu(<32 x i32> %a, i32 %b)
  ret <32 x i32> %c
}
; CHECK: = vmpa(v1:0.ub,r0.ub)

declare <32 x i32> @llvm.hexagon.V6.vmpabuu.acc(<32 x i32>, <32 x i32>, i32)
define <32 x i32> @V6_vmpabuu_acc(<32 x i32> %a, <32 x i32> %b, i32 %c) {
  %d = call <32 x i32> @llvm.hexagon.V6.vmpabuu.acc(<32 x i32> %a, <32 x i32> %b, i32 %c)
  ret <32 x i32> %d
}
; CHECK: += vmpa(v3:2.ub,r0.ub)

declare <16 x i32> @llvm.hexagon.V6.vmpauhuhsat(<16 x i32>, <16 x i32>, i64)
define <16 x i32> @V6_vmpauhuhsat(<16 x i32> %a, <16 x i32> %b, i64 %c) {
  %d = call <16 x i32> @llvm.hexagon.V6.vmpauhuhsat(<16 x i32> %a, <16 x i32> %b, i64 %c)
  ret <16 x i32> %d
}
; CHECK: = vmpa(v0.h,v1.uh,r1:0.uh):sat

declare <16 x i32> @llvm.hexagon.V6.vmpsuhuhsat(<16 x i32>, <16 x i32>, i64)
define <16 x i32> @V6_vmpsuhuhsat(<16 x i32> %a, <16 x i32> %b, i64 %c) {
  %d = call <16 x i32> @llvm.hexagon.V6.vmpsuhuhsat(<16 x i32> %a, <16 x i32> %b, i64 %c)
  ret <16 x i32> %d
}
; CHECK: = vmps(v0.h,v1.uh,r1:0.uh):sat

declare <32 x i32> @llvm.hexagon.V6.vmpyh.acc(<32 x i32>, <16 x i32>, i32)
define <32 x i32> @V6_vmpyh_acc(<32 x i32> %a, <16 x i32> %b, i32 %c) {
  %d = call <32 x i32> @llvm.hexagon.V6.vmpyh.acc(<32 x i32> %a, <16 x i32> %b, i32 %c)
  ret <32 x i32> %d
}
; CHECK: += vmpy(v2.h,r0.h)

declare <16 x i32> @llvm.hexagon.V6.vmpyuhe(<16 x i32>, i32)
define <16 x i32> @V6_vmpyuhe(<16 x i32> %a, i32 %b) {
  %c = call <16 x i32> @llvm.hexagon.V6.vmpyuhe(<16 x i32> %a, i32 %b)
  ret <16 x i32> %c
}
; CHECK: = vmpye(v0.uh,r0.uh)

;declare <16 x i32> @llvm.hexagon.V6.vprefixqb(<64 x i1>)
;define <16 x i32> @V6_vprefixqb(<64 x i1> %a) {
;  %b = call <16 x i32> @llvm.hexagon.V6.vprefixqb(<64 x i1> %a)
;  ret <16 x i32> %b
;}

;declare <16 x i32> @llvm.hexagon.V6.vprefixqh(<64 x i1>)
;define <16 x i32> @V6_vprefixqh(<64 x i1> %a) {
;  %b = call <16 x i32> @llvm.hexagon.V6.vprefixqh(<64 x i1> %a)
;  ret <16 x i32> %b
;}

;declare <16 x i32> @llvm.hexagon.V6.vprefixqw(<64 x i1>)
;define <16 x i32> @V6_vprefixqw(<64 x i1> %a) {
;  %b = call <16 x i32> @llvm.hexagon.V6.vprefixqw(<64 x i1> %a)
;  ret <16 x i32> %b
;}

