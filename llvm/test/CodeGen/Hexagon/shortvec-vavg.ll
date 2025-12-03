
; RUN: llc -march=hexagon -hexagon-hvx-widen=32 -hexagon-widen-short-vector -mattr=+hvxv73,+hvx-length128b < %s | FileCheck %s

; CHECK: = vavg(v{{[0-9:]+}}.h,v{{[0-9]+}}.h)

define dllexport void @tvm_vavg(i8 %val0, i8 %val1, i8* %dst) local_unnamed_addr {
entry:
  %1 = insertelement <64 x i8> poison, i8 %val0, i32 62
  %2 = insertelement <64 x i8> %1, i8 %val1, i32 63
  %3 = zext <64 x i8> %2 to <64 x i16>
  %t.7 = insertelement <64 x i8> poison, i8 %val1, i32 62
  %t.8 = insertelement <64 x i8> %t.7, i8 %val0, i32 63
  %t.9 = zext <64 x i8> %t.8 to <64 x i16>
  %t.17 = add nuw nsw <64 x i16> %t.9, %3
  %t.18 = lshr <64 x i16> %t.17, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %t.19 = trunc <64 x i16> %t.18 to <64 x i8>
  %t.29 = extractelement <64 x i8> %t.19, i32 6
  store i8 %t.29, i8* %dst, align 2
  ret void
}
