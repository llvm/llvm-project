; RUN: llc -mcpu=pwr6 -mattr=+altivec < %s | FileCheck %s

; Check vector comparisons using altivec. For non-native types, just basic
; comparison instruction check is done. For altivec supported type (16i8,
; 8i16, 4i32, and 4f32) all the comparisons operators (==, !=, >, >=, <, <=)
; are checked.


target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define <2 x i8> @v2si8_cmp(<2 x i8> %x, <2 x i8> %y) nounwind readnone {
  %cmp = icmp eq <2 x i8> %x, %y
  %sext = sext <2 x i1> %cmp to <2 x i8>
  ret <2 x i8> %sext
}
; CHECK-LABEL: v2si8_cmp:
; CHECK: vcmpequb {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <4 x i8> @v4si8_cmp(<4 x i8> %x, <4 x i8> %y) nounwind readnone {
  %cmp = icmp eq <4 x i8> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i8>
  ret <4 x i8> %sext
}
; CHECK-LABEL: v4si8_cmp:
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <8 x i8> @v8si8_cmp(<8 x i8> %x, <8 x i8> %y) nounwind readnone {
  %cmp = icmp eq <8 x i8> %x, %y
  %sext = sext <8 x i1> %cmp to <8 x i8>
  ret <8 x i8> %sext
}
; CHECK-LABEL: v8si8_cmp:
; CHECK: vcmpequh {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


; Additional tests for v16i8 since it is a altivec native type

define <16 x i8> @v16si8_cmp_eq(<16 x i8> %x, <16 x i8> %y) nounwind readnone {
  %cmp = icmp eq <16 x i8> %x, %y
  %sext = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %sext
}
; CHECK-LABEL: v16si8_cmp_eq:
; CHECK: vcmpequb 2, 2, 3

define <16 x i8> @v16si8_cmp_ne(<16 x i8> %x, <16 x i8> %y) nounwind readnone {
entry:
  %cmp = icmp ne <16 x i8> %x, %y
  %sext = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %sext
}
; CHECK-LABEL:     v16si8_cmp_ne:
; CHECK:     vcmpequb [[RET:[0-9]+]], 2, 3
; CHECK-NEXT: vnor     2, [[RET]], [[RET]]

define <16 x i8> @v16si8_cmp_le(<16 x i8> %x, <16 x i8> %y) nounwind readnone {
entry:
  %cmp = icmp sle <16 x i8> %x, %y
  %sext = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %sext
}
; CHECK-LABEL:      v16si8_cmp_le:
; CHECK:      vcmpgtsb [[RET:[0-9]+]], 2, 3
; CHECK-NEXT: vnor     2, [[RET]], [[RET]]

define <16 x i8> @v16ui8_cmp_le(<16 x i8> %x, <16 x i8> %y) nounwind readnone {
entry:
  %cmp = icmp ule <16 x i8> %x, %y
  %sext = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %sext
}
; CHECK-LABEL:      v16ui8_cmp_le:
; CHECK:      vcmpgtub [[RET:[0-9]+]], 2, 3
; CHECK-NEXT: vnor     2, [[RET]], [[RET]]

define <16 x i8> @v16si8_cmp_lt(<16 x i8> %x, <16 x i8> %y) nounwind readnone {
entry:
  %cmp = icmp slt <16 x i8> %x, %y
  %sext = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %sext
}
; CHECK-LABEL: v16si8_cmp_lt:
; CHECK: vcmpgtsb 2, 3, 2

define <16 x i8> @v16ui8_cmp_lt(<16 x i8> %x, <16 x i8> %y) nounwind readnone {
entry:
  %cmp = icmp ult <16 x i8> %x, %y
  %sext = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %sext
}
; CHECK-LABEL: v16ui8_cmp_lt:
; CHECK: vcmpgtub 2, 3, 2

define <16 x i8> @v16si8_cmp_gt(<16 x i8> %x, <16 x i8> %y) nounwind readnone {
entry:
  %cmp = icmp sgt <16 x i8> %x, %y
  %sext = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %sext
}
; CHECK-LABEL: v16si8_cmp_gt:
; CHECK: vcmpgtsb 2, 2, 3

define <16 x i8> @v16ui8_cmp_gt(<16 x i8> %x, <16 x i8> %y) nounwind readnone {
entry:
  %cmp = icmp ugt <16 x i8> %x, %y
  %sext = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %sext
}
; CHECK-LABEL: v16ui8_cmp_gt:
; CHECK: vcmpgtub 2, 2, 3

define <16 x i8> @v16si8_cmp_ge(<16 x i8> %x, <16 x i8> %y) nounwind readnone {
entry:
  %cmp = icmp sge <16 x i8> %x, %y
  %sext = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %sext
}
; CHECK-LABEL:      v16si8_cmp_ge:
; CHECK:      vcmpgtsb [[RET:[0-9]+]], 3, 2
; CHECK-NEXT: vnor     2, [[RET]], [[RET]]

define <16 x i8> @v16ui8_cmp_ge(<16 x i8> %x, <16 x i8> %y) nounwind readnone {
entry:
  %cmp = icmp uge <16 x i8> %x, %y
  %sext = sext <16 x i1> %cmp to <16 x i8>
  ret <16 x i8> %sext
}
; CHECK-LABEL:      v16ui8_cmp_ge:
; CHECK:      vcmpgtub [[RET:[0-9]+]], 3, 2
; CHECK-NEXT: vnor     2, [[RET]], [[RET]]


define <32 x i8> @v32si8_cmp(<32 x i8> %x, <32 x i8> %y) nounwind readnone {
  %cmp = icmp eq <32 x i8> %x, %y
  %sext = sext <32 x i1> %cmp to <32 x i8>
  ret <32 x i8> %sext
}
; CHECK-LABEL: v32si8_cmp:
; CHECK: vcmpequb {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequb {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <2 x i16> @v2si16_cmp(<2 x i16> %x, <2 x i16> %y) nounwind readnone {
  %cmp = icmp eq <2 x i16> %x, %y
  %sext = sext <2 x i1> %cmp to <2 x i16>
  ret <2 x i16> %sext
}
; CHECK-LABEL: v2si16_cmp:
; CHECK: vcmpequh {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <4 x i16> @v4si16_cmp(<4 x i16> %x, <4 x i16> %y) nounwind readnone {
  %cmp = icmp eq <4 x i16> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i16>
  ret <4 x i16> %sext
}
; CHECK-LABEL: v4si16_cmp:
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


; Additional tests for v8i16 since it is an altivec native type

define <8 x i16> @v8si16_cmp_eq(<8 x i16> %x, <8 x i16> %y) nounwind readnone {
entry:
  %cmp = icmp eq <8 x i16> %x, %y
  %sext = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %sext
}
; CHECK-LABEL: v8si16_cmp_eq:
; CHECK: vcmpequh 2, 2, 3

define <8 x i16> @v8si16_cmp_ne(<8 x i16> %x, <8 x i16> %y) nounwind readnone {
entry:
  %cmp = icmp ne <8 x i16> %x, %y
  %sext = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %sext
}
; CHECK-LABEL:      v8si16_cmp_ne:
; CHECK:      vcmpequh [[RET:[0-9]+]], 2, 3
; CHECK-NEXT: vnor     2, [[RET]], [[RET]]

define <8 x i16> @v8si16_cmp_le(<8 x i16> %x, <8 x i16> %y) nounwind readnone {
entry:
  %cmp = icmp sle <8 x i16> %x, %y
  %sext = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %sext
}
; CHECK-LABEL:      v8si16_cmp_le:
; CHECK:      vcmpgtsh [[RET:[0-9]+]], 2, 3
; CHECK-NEXT: vnor     2, [[RET]], [[RET]]

define <8 x i16> @v8ui16_cmp_le(<8 x i16> %x, <8 x i16> %y) nounwind readnone {
entry:
  %cmp = icmp ule <8 x i16> %x, %y
  %sext = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %sext
}
; CHECK-LABEL:      v8ui16_cmp_le:
; CHECK:      vcmpgtuh [[RET:[0-9]+]], 2, 3
; CHECK-NEXT: vnor     2, [[RET]], [[RET]]

define <8 x i16> @v8si16_cmp_lt(<8 x i16> %x, <8 x i16> %y) nounwind readnone {
entry:
  %cmp = icmp slt <8 x i16> %x, %y
  %sext = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %sext
}
; CHECK-LABEL: v8si16_cmp_lt:
; CHECK: vcmpgtsh 2, 3, 2

define <8 x i16> @v8ui16_cmp_lt(<8 x i16> %x, <8 x i16> %y) nounwind readnone {
entry:
  %cmp = icmp ult <8 x i16> %x, %y
  %sext = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %sext
}
; CHECK-LABEL: v8ui16_cmp_lt:
; CHECK: vcmpgtuh 2, 3, 2

define <8 x i16> @v8si16_cmp_gt(<8 x i16> %x, <8 x i16> %y) nounwind readnone {
entry:
  %cmp = icmp sgt <8 x i16> %x, %y
  %sext = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %sext
}
; CHECK-LABEL: v8si16_cmp_gt:
; CHECK: vcmpgtsh 2, 2, 3

define <8 x i16> @v8ui16_cmp_gt(<8 x i16> %x, <8 x i16> %y) nounwind readnone {
entry:
  %cmp = icmp ugt <8 x i16> %x, %y
  %sext = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %sext
}
; CHECK-LABEL: v8ui16_cmp_gt:
; CHECK: vcmpgtuh 2, 2, 3

define <8 x i16> @v8si16_cmp_ge(<8 x i16> %x, <8 x i16> %y) nounwind readnone {
entry:
  %cmp = icmp sge <8 x i16> %x, %y
  %sext = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %sext
}
; CHECK-LABEL:      v8si16_cmp_ge:
; CHECK:      vcmpgtsh [[RET:[0-9]+]], 3, 2
; CHECK-NEXT: vnor     2, [[RET]], [[RET]]

define <8 x i16> @v8ui16_cmp_ge(<8 x i16> %x, <8 x i16> %y) nounwind readnone {
entry:
  %cmp = icmp uge <8 x i16> %x, %y
  %sext = sext <8 x i1> %cmp to <8 x i16>
  ret <8 x i16> %sext
}
; CHECK-LABEL:      v8ui16_cmp_ge:
; CHECK:      vcmpgtuh [[RET:[0-9]+]], 3, 2
; CHECK-NEXT: vnor     2, [[RET]], [[RET]]


define <16 x i16> @v16si16_cmp(<16 x i16> %x, <16 x i16> %y) nounwind readnone {
  %cmp = icmp eq <16 x i16> %x, %y
  %sext = sext <16 x i1> %cmp to <16 x i16>
  ret <16 x i16> %sext
}
; CHECK-LABEL: v16si16_cmp:
; CHECK: vcmpequh {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequh {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <32 x i16> @v32si16_cmp(<32 x i16> %x, <32 x i16> %y) nounwind readnone {
  %cmp = icmp eq <32 x i16> %x, %y
  %sext = sext <32 x i1> %cmp to <32 x i16>
  ret <32 x i16> %sext
}
; CHECK-LABEL: v32si16_cmp:
; CHECK: vcmpequh {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequh {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequh {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequh {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <2 x i32> @v2si32_cmp(<2 x i32> %x, <2 x i32> %y) nounwind readnone {
  %cmp = icmp eq <2 x i32> %x, %y
  %sext = sext <2 x i1> %cmp to <2 x i32>
  ret <2 x i32> %sext
}
; CHECK-LABEL: v2si32_cmp:
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


; Additional tests for v4si32 since it is an altivec native type

define <4 x i32> @v4si32_cmp_eq(<4 x i32> %x, <4 x i32> %y) nounwind readnone {
entry:
  %cmp = icmp eq <4 x i32> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %sext
}
; CHECK-LABEL: v4si32_cmp_eq:
; CHECK: vcmpequw 2, 2, 3

define <4 x i32> @v4si32_cmp_ne(<4 x i32> %x, <4 x i32> %y) nounwind readnone {
entry:
  %cmp = icmp ne <4 x i32> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %sext
}
; CHECK-LABEL:      v4si32_cmp_ne:
; CHECK:      vcmpequw [[RCMP:[0-9]+]], 2, 3
; CHECK-NEXT: vnor     2, [[RCMP]], [[RCMP]]

define <4 x i32> @v4si32_cmp_le(<4 x i32> %x, <4 x i32> %y) nounwind readnone {
entry:
  %cmp = icmp sle <4 x i32> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %sext
}
; CHECK-LABEL:      v4si32_cmp_le:
; CHECK:      vcmpgtsw [[RET:[0-9]+]], 2, 3
; CHECK-NEXT: vnor     2, [[RET]], [[RET]]

define <4 x i32> @v4ui32_cmp_le(<4 x i32> %x, <4 x i32> %y) nounwind readnone {
entry:
  %cmp = icmp ule <4 x i32> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %sext
}
; CHECK-LABEL:      v4ui32_cmp_le:
; CHECK:      vcmpgtuw [[RET:[0-9]+]], 2, 3
; CHECK-NEXT: vnor     2, [[RET]], [[RET]]

define <4 x i32> @v4si32_cmp_lt(<4 x i32> %x, <4 x i32> %y) nounwind readnone {
entry:
  %cmp = icmp slt <4 x i32> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %sext
}
; CHECK-LABEL: v4si32_cmp_lt:
; CHECK: vcmpgtsw 2, 3, 2

define <4 x i32> @v4ui32_cmp_lt(<4 x i32> %x, <4 x i32> %y) nounwind readnone {
entry:
  %cmp = icmp ult <4 x i32> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %sext
}
; CHECK-LABEL: v4ui32_cmp_lt:
; CHECK: vcmpgtuw 2, 3, 2

define <4 x i32> @v4si32_cmp_gt(<4 x i32> %x, <4 x i32> %y) nounwind readnone {
entry:
  %cmp = icmp sgt <4 x i32> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %sext
}
; CHECK-LABEL: v4si32_cmp_gt:
; CHECK: vcmpgtsw 2, 2, 3

define <4 x i32> @v4ui32_cmp_gt(<4 x i32> %x, <4 x i32> %y) nounwind readnone {
entry:
  %cmp = icmp ugt <4 x i32> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %sext
}
; CHECK-LABEL: v4ui32_cmp_gt:
; CHECK: vcmpgtuw 2, 2, 3

define <4 x i32> @v4si32_cmp_ge(<4 x i32> %x, <4 x i32> %y) nounwind readnone {
entry:
  %cmp = icmp sge <4 x i32> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %sext
}
; CHECK-LABEL:      v4si32_cmp_ge:
; CHECK:      vcmpgtsw [[RET:[0-9]+]], 3, 2
; CHECK-NEXT: vnor     2, [[RET]], [[RET]]

define <4 x i32> @v4ui32_cmp_ge(<4 x i32> %x, <4 x i32> %y) nounwind readnone {
entry:
  %cmp = icmp uge <4 x i32> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %sext
}
; CHECK-LABEL:      v4ui32_cmp_ge:
; CHECK:      vcmpgtuw [[RET:[0-9]+]], 3, 2
; CHECK-NEXT: vnor     2, [[RET]], [[RET]]


define <8 x i32> @v8si32_cmp(<8 x i32> %x, <8 x i32> %y) nounwind readnone {
  %cmp = icmp eq <8 x i32> %x, %y
  %sext = sext <8 x i1> %cmp to <8 x i32>
  ret <8 x i32> %sext
}
; CHECK-LABEL: v8si32_cmp:
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <16 x i32> @v16si32_cmp(<16 x i32> %x, <16 x i32> %y) nounwind readnone {
  %cmp = icmp eq <16 x i32> %x, %y
  %sext = sext <16 x i1> %cmp to <16 x i32>
  ret <16 x i32> %sext
}
; CHECK-LABEL: v16si32_cmp:
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <32 x i32> @v32si32_cmp(<32 x i32> %x, <32 x i32> %y) nounwind readnone {
  %cmp = icmp eq <32 x i32> %x, %y
  %sext = sext <32 x i1> %cmp to <32 x i32>
  ret <32 x i32> %sext
}
; CHECK-LABEL: v32si32_cmp:
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpequw {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


define <2 x float> @v2f32_cmp(<2 x float> %x, <2 x float> %y) nounwind readnone {
entry:
  %cmp = fcmp oeq <2 x float> %x, %y
  %sext = sext <2 x i1> %cmp to <2 x i32>
  %0 = bitcast <2 x i32> %sext to <2 x float>
  ret <2 x float> %0
}
; CHECK-LABEL: v2f32_cmp:
; CHECK: vcmpeqfp {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}


; Additional tests for v4f32 since it is a altivec native type

define <4 x float> @v4f32_cmp_eq(<4 x float> %x, <4 x float> %y) nounwind readnone {
entry:
  %cmp = fcmp oeq <4 x float> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  %0 = bitcast <4 x i32> %sext to <4 x float>
  ret <4 x float> %0
}
; CHECK-LABEL: v4f32_cmp_eq:
; CHECK: vcmpeqfp 2, 2, 3

define <4 x float> @v4f32_cmp_ne(<4 x float> %x, <4 x float> %y) nounwind readnone {
entry:
  %cmp = fcmp une <4 x float> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  %0 = bitcast <4 x i32> %sext to <4 x float>
  ret <4 x float> %0
}
; CHECK-LABEL:      v4f32_cmp_ne:
; CHECK:      vcmpeqfp [[RET:[0-9]+]], 2, 3
; CHECK-NEXT: vnor     2, [[RET]], [[RET]]

define <4 x float> @v4f32_cmp_le(<4 x float> %x, <4 x float> %y) nounwind readnone {
entry:
  %cmp = fcmp ole <4 x float> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  %0 = bitcast <4 x i32> %sext to <4 x float>
  ret <4 x float> %0
}
; CHECK-LABEL:      v4f32_cmp_le:
; CHECK: vcmpgefp 2, 3, 2

define <4 x float> @v4f32_cmp_lt(<4 x float> %x, <4 x float> %y) nounwind readnone {
entry:
  %cmp = fcmp olt <4 x float> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  %0 = bitcast <4 x i32> %sext to <4 x float>
  ret <4 x float> %0
}
; CHECK-LABEL: v4f32_cmp_lt:
; CHECK: vcmpgtfp 2, 3, 2

define <4 x float> @v4f32_cmp_ge(<4 x float> %x, <4 x float> %y) nounwind readnone {
entry:
  %cmp = fcmp oge <4 x float> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  %0 = bitcast <4 x i32> %sext to <4 x float>
  ret <4 x float> %0
}
; CHECK-LABEL: v4f32_cmp_ge:
; CHECK: vcmpgefp 2, 2, 3

define <4 x float> @v4f32_cmp_gt(<4 x float> %x, <4 x float> %y) nounwind readnone {
entry:
  %cmp = fcmp ogt <4 x float> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  %0 = bitcast <4 x i32> %sext to <4 x float>
  ret <4 x float> %0
}
; CHECK-LABEL: v4f32_cmp_gt:
; CHECK: vcmpgtfp 2, 2, 3

define <4 x float> @v4f32_cmp_ule(<4 x float> %x, <4 x float> %y) nounwind readnone {
entry:
  %cmp = fcmp ule <4 x float> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  %0 = bitcast <4 x i32> %sext to <4 x float>
  ret <4 x float> %0
}
; CHECK-LABEL: v4f32_cmp_ule:
; CHECK:      vcmpgtfp [[RET:[0-9]+]], 2, 3
; CHECK-NEXT: vnor     2, [[RET]], [[RET]]

define <4 x float> @v4f32_cmp_ult(<4 x float> %x, <4 x float> %y) nounwind readnone {
entry:
  %cmp = fcmp ult <4 x float> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  %0 = bitcast <4 x i32> %sext to <4 x float>
  ret <4 x float> %0
}
; CHECK-LABEL: v4f32_cmp_ult:
; CHECK:      vcmpgefp [[RET:[0-9]+]], 2, 3
; CHECK-NEXT: vnor     2, [[RET]], [[RET]]

define <4 x float> @v4f32_cmp_uge(<4 x float> %x, <4 x float> %y) nounwind readnone {
entry:
  %cmp = fcmp uge <4 x float> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  %0 = bitcast <4 x i32> %sext to <4 x float>
  ret <4 x float> %0
}
; CHECK-LABEL: v4f32_cmp_uge:
; CHECK:      vcmpgtfp [[RET:[0-9]+]], 3, 2
; CHECK-NEXT: vnor     2, [[RET]], [[RET]]

define <4 x float> @v4f32_cmp_ugt(<4 x float> %x, <4 x float> %y) nounwind readnone {
entry:
  %cmp = fcmp ugt <4 x float> %x, %y
  %sext = sext <4 x i1> %cmp to <4 x i32>
  %0 = bitcast <4 x i32> %sext to <4 x float>
  ret <4 x float> %0
}
; CHECK-LABEL: v4f32_cmp_ugt:
; CHECK:      vcmpgefp [[RET:[0-9]+]], 3, 2
; CHECK-NEXT: vnor     2, [[RET]], [[RET]]


define <8 x float> @v8f32_cmp(<8 x float> %x, <8 x float> %y) nounwind readnone {
entry:
  %cmp = fcmp oeq <8 x float> %x, %y
  %sext = sext <8 x i1> %cmp to <8 x i32>
  %0 = bitcast <8 x i32> %sext to <8 x float>
  ret <8 x float> %0
}
; CHECK-LABEL: v8f32_cmp:
; CHECK: vcmpeqfp {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
; CHECK: vcmpeqfp {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
