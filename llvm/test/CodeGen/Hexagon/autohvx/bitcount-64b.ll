; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: f0
; CHECK: v[[V00:[0-9]+]]:[[V01:[0-9]+]].uh = vzxt(v0.ub)
; CHECK-DAG: v[[V02:[0-9]+]].h = vpopcount(v[[V00]].h)
; CHECK-DAG: v[[V03:[0-9]+]].h = vpopcount(v[[V01]].h)
; CHECK: v0.b = vshuffe(v[[V02]].b,v[[V03]].b)
define <64 x i8> @f0(<64 x i8> %a0) #0 {
  %t0 = call <64 x i8> @llvm.ctpop.v64i8(<64 x i8> %a0)
  ret <64 x i8> %t0
}

; CHECK-LABEL: f1
; CHECK: v0.h = vpopcount(v0.h)
define <32 x i16> @f1(<32 x i16> %a0) #0 {
  %t0 = call <32 x i16> @llvm.ctpop.v32i16(<32 x i16> %a0)
  ret <32 x i16> %t0
}

; CHECK-LABEL: f2
; CHECK: v[[V20:[0-9]+]].h = vpopcount(v0.h)
; CHECK: v[[V21:[0-9]+]]:[[V22:[0-9]+]].uw = vzxt(v[[V20]].uh)
; CHECK: v0.w = vadd(v[[V22]].w,v[[V21]].w)
define <16 x i32> @f2(<16 x i32> %a0) #0 {
  %t0 = call <16 x i32> @llvm.ctpop.v16i32(<16 x i32> %a0)
  ret <16 x i32> %t0
}

; CHECK-LABEL: f3
; CHECK-DAG: r[[R30:[0-9]+]] = ##134744072
; CHECK-DAG: v[[V31:[0-9]+]]:[[V32:[0-9]+]].uh = vzxt(v0.ub)
; CHECK: v[[V33:[0-9]+]] = vsplat(r[[R30]])
; CHECK-DAG: v[[V34:[0-9]+]].uh = vcl0(v[[V31]].uh)
; CHECK-DAG: v[[V35:[0-9]+]].uh = vcl0(v[[V32]].uh)
; CHECK: v[[V36:[0-9]+]].b = vshuffe(v[[V34]].b,v[[V35]].b)
; CHECK: v0.b = vsub(v[[V36]].b,v[[V33]].b)
define <64 x i8> @f3(<64 x i8> %a0) #0 {
  %t0 = call <64 x i8> @llvm.ctlz.v64i8(<64 x i8> %a0)
  ret <64 x i8> %t0
}

; CHECK-LABEL: f4
; CHECK: v0.uh = vcl0(v0.uh)
define <32 x i16> @f4(<32 x i16> %a0) #0 {
  %t0 = call <32 x i16> @llvm.ctlz.v32i16(<32 x i16> %a0)
  ret <32 x i16> %t0
}

; CHECK-LABEL: f5
; CHECK: v0.uw = vcl0(v0.uw)
define <16 x i32> @f5(<16 x i32> %a0) #0 {
  %t0 = call <16 x i32> @llvm.ctlz.v16i32(<16 x i32> %a0)
  ret <16 x i32> %t0
}

; CHECK-LABEL: f6
; r = 0x01010101
; CHECK-DAG: r[[R60:[0-9]+]] = ##16843009
; CHECK-DAG: v[[V61:[0-9]+]] = vnot(v0)
; r = 0x08080808
; CHECK-DAG: r[[R62:[0-9]+]] = ##134744072
; CHECK: v[[V63:[0-9]+]] = vsplat(r[[R60]])
; CHECK-DAG: v[[V64:[0-9]+]] = vsplat(r[[R62]])
; CHECK: v[[V65:[0-9]+]].b = vsub(v0.b,v[[V63]].b)
; CHECK: v[[V66:[0-9]+]] = vand(v[[V61]],v[[V65]])
; Ctlz:
; CHECK: v[[V67:[0-9]+]]:[[V68:[0-9]+]].uh = vzxt(v[[V66]].ub)
; CHECK: v[[V69:[0-9]+]].uh = vcl0(v[[V68]].uh)
; CHECK: v[[V6A:[0-9]+]].uh = vcl0(v[[V67]].uh)
; CHECK: v[[V6B:[0-9]+]].b = vshuffe(v[[V6A]].b,v[[V69]].b)
; CHECK: v[[V6C:[0-9]+]].b = vsub(v[[V6B]].b,v[[V64]].b)
; CHECK: v0.b = vsub(v[[V64]].b,v[[V6C]].b)
define <64 x i8> @f6(<64 x i8> %a0) #0 {
  %t0 = call <64 x i8> @llvm.cttz.v64i8(<64 x i8> %a0)
  ret <64 x i8> %t0
}

; CHECK-LABEL: f7
; r = 0x00010001
; CHECK-DAG: r[[R70:[0-9]+]] = ##65537
; CHECK-DAG: v[[V71:[0-9]+]] = vnot(v0)
; r = 0x00100010  // halfword bitwidths
; CHECK-DAG: r[[R72:[0-9]+]] = ##1048592
; CHECK: v[[V73:[0-9]+]] = vsplat(r[[R70]])
; CHECK: v[[V74:[0-9]+]] = vsplat(r[[R72]])
; CHECK: v[[V75:[0-9]+]].h = vsub(v0.h,v[[V73]].h)
; CHECK: v[[V76:[0-9]+]] = vand(v[[V71]],v[[V75]])
; Ctlz:
; CHECK: v[[V77:[0-9]+]].uh = vcl0(v[[V76]].uh)
; CHECK: v0.h = vsub(v[[V74]].h,v[[V77]].h)
define <32 x i16> @f7(<32 x i16> %a0) #0 {
  %t0 = call <32 x i16> @llvm.cttz.v32i16(<32 x i16> %a0)
  ret <32 x i16> %t0
}

; CHECK-LABEL: f8
; CHECK-DAG: r[[R80:[0-9]+]] = #1
; CHECK-DAG: v[[V81:[0-9]+]] = vnot(v0)
; CHECK-DAG: r[[R82:[0-9]+]] = #32
; CHECK: v[[V83:[0-9]+]] = vsplat(r[[R80]])
; CHECK: v[[V84:[0-9]+]] = vsplat(r[[R82]])
; CHECK: v[[V85:[0-9]+]].w = vsub(v0.w,v[[V83]].w)
; CHECK: v[[V86:[0-9]+]] = vand(v[[V81]],v[[V85]])
; Ctlz:
; CHECK: v[[V87:[0-9]+]].uw = vcl0(v[[V86]].uw)
; CHECK: v0.w = vsub(v[[V84]].w,v[[V87]].w)
define <16 x i32> @f8(<16 x i32> %a0) #0 {
  %t0 = call <16 x i32> @llvm.cttz.v16i32(<16 x i32> %a0)
  ret <16 x i32> %t0
}


declare <64 x i8>  @llvm.ctpop.v64i8(<64 x i8>) #0
declare <32 x i16> @llvm.ctpop.v32i16(<32 x i16>) #0
declare <16 x i32> @llvm.ctpop.v16i32(<16 x i32>) #0

declare <64 x i8>  @llvm.ctlz.v64i8(<64 x i8>) #0
declare <32 x i16> @llvm.ctlz.v32i16(<32 x i16>) #0
declare <16 x i32> @llvm.ctlz.v16i32(<16 x i32>) #0

declare <64 x i8>  @llvm.cttz.v64i8(<64 x i8>) #0
declare <32 x i16> @llvm.cttz.v32i16(<32 x i16>) #0
declare <16 x i32> @llvm.cttz.v16i32(<16 x i32>) #0

attributes #0 = { readnone nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b,-packets" }
