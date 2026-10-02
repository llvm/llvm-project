; Checks if conversion is only inserted for the spilled register
; instead of 2 conversions for the W register
; XFAIL: *
; NOTE: XFAIL until Hexagon HVX IEEE→QFloat isel translation is upstreamed; remove XFAIL when that lands.
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv81 -enable-xqf-gen=true \
; RUN: -hexagon-qfloat-mode=ieee -mattr=+hvxv81,+hvx-length128B \
; RUN: -enable-postra-xqf-check -debug-only=handle-qfp 2>&1 < %s -o /dev/null | FileCheck %s

; CHECK: Finding uses of:   renamable $v1 = PS_vloadrv_ai %stack.0
; CHECK: Collecting convert instruction with type Hi Op :  renamable $v{{[0-9]+}} = V6_vconv_hf_qf32 killed renamable $w0
; CHECK: Inserting new instruction:   $v1 = V6_vconv_qf32_sf killed renamable $v1

define void @foo(ptr %0) {
entry:
  br label %.preheader78.i.i

.preheader78.i.i:
  %1 = load ptr, ptr %0, align 16
  tail call void (i32, i32, ptr, ...) %1(i32 0, i32 0, ptr null, ptr null, i32 0, ptr null, ptr null)
  %2 = load <32 x i32>, ptr %0, align 1
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vcvt.hf.sf.128B(<32 x i32> %2, <32 x i32> zeroinitializer)
  %bc.i8.i.i = bitcast <32 x i32> %3 to <64 x i16>
  %4 = extractelement <64 x i16> %bc.i8.i.i, i64 0
  store i16 %4, ptr %0, align 2
  br label %.preheader78.i.i
}

declare <32 x i32> @llvm.hexagon.V6.vcvt.hf.sf.128B(<32 x i32>, <32 x i32>)
