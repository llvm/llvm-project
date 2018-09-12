; Test spills of zero extensions when high GR32s are available.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Test a case where we spill the source of at least one LLCRMux.  We want
; to use LLC(H) if possible.
define void @f1(i32 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: llc{{h?}} {{%r[0-9]+}}, 1{{[67]}}{{[379]}}(%r15)
; CHECK: br %r14
  %val0 = load volatile i32, i32 *%ptr
  %val1 = load volatile i32, i32 *%ptr
  %val2 = load volatile i32, i32 *%ptr
  %val3 = load volatile i32, i32 *%ptr
  %val4 = load volatile i32, i32 *%ptr
  %val5 = load volatile i32, i32 *%ptr
  %val6 = load volatile i32, i32 *%ptr
  %val7 = load volatile i32, i32 *%ptr
  %val8 = load volatile i32, i32 *%ptr
  %val9 = load volatile i32, i32 *%ptr
  %val10 = load volatile i32, i32 *%ptr
  %val11 = load volatile i32, i32 *%ptr
  %val12 = load volatile i32, i32 *%ptr
  %val13 = load volatile i32, i32 *%ptr
  %val14 = load volatile i32, i32 *%ptr
  %val15 = load volatile i32, i32 *%ptr
  %val16 = load volatile i32, i32 *%ptr
  %val17 = load volatile i32, i32 *%ptr
  %val18 = load volatile i32, i32 *%ptr
  %val19 = load volatile i32, i32 *%ptr
  %val20 = load volatile i32, i32 *%ptr
  %val21 = load volatile i32, i32 *%ptr
  %val22 = load volatile i32, i32 *%ptr
  %val23 = load volatile i32, i32 *%ptr
  %val24 = load volatile i32, i32 *%ptr
  %val25 = load volatile i32, i32 *%ptr
  %val26 = load volatile i32, i32 *%ptr
  %val27 = load volatile i32, i32 *%ptr
  %val28 = load volatile i32, i32 *%ptr
  %val29 = load volatile i32, i32 *%ptr
  %val30 = load volatile i32, i32 *%ptr
  %val31 = load volatile i32, i32 *%ptr

  %trunc0 = trunc i32 %val0 to i8
  %trunc1 = trunc i32 %val1 to i8
  %trunc2 = trunc i32 %val2 to i8
  %trunc3 = trunc i32 %val3 to i8
  %trunc4 = trunc i32 %val4 to i8
  %trunc5 = trunc i32 %val5 to i8
  %trunc6 = trunc i32 %val6 to i8
  %trunc7 = trunc i32 %val7 to i8
  %trunc8 = trunc i32 %val8 to i8
  %trunc9 = trunc i32 %val9 to i8
  %trunc10 = trunc i32 %val10 to i8
  %trunc11 = trunc i32 %val11 to i8
  %trunc12 = trunc i32 %val12 to i8
  %trunc13 = trunc i32 %val13 to i8
  %trunc14 = trunc i32 %val14 to i8
  %trunc15 = trunc i32 %val15 to i8
  %trunc16 = trunc i32 %val16 to i8
  %trunc17 = trunc i32 %val17 to i8
  %trunc18 = trunc i32 %val18 to i8
  %trunc19 = trunc i32 %val19 to i8
  %trunc20 = trunc i32 %val20 to i8
  %trunc21 = trunc i32 %val21 to i8
  %trunc22 = trunc i32 %val22 to i8
  %trunc23 = trunc i32 %val23 to i8
  %trunc24 = trunc i32 %val24 to i8
  %trunc25 = trunc i32 %val25 to i8
  %trunc26 = trunc i32 %val26 to i8
  %trunc27 = trunc i32 %val27 to i8
  %trunc28 = trunc i32 %val28 to i8
  %trunc29 = trunc i32 %val29 to i8
  %trunc30 = trunc i32 %val30 to i8
  %trunc31 = trunc i32 %val31 to i8

  %ext0 = zext i8 %trunc0 to i32
  %ext1 = zext i8 %trunc1 to i32
  %ext2 = zext i8 %trunc2 to i32
  %ext3 = zext i8 %trunc3 to i32
  %ext4 = zext i8 %trunc4 to i32
  %ext5 = zext i8 %trunc5 to i32
  %ext6 = zext i8 %trunc6 to i32
  %ext7 = zext i8 %trunc7 to i32
  %ext8 = zext i8 %trunc8 to i32
  %ext9 = zext i8 %trunc9 to i32
  %ext10 = zext i8 %trunc10 to i32
  %ext11 = zext i8 %trunc11 to i32
  %ext12 = zext i8 %trunc12 to i32
  %ext13 = zext i8 %trunc13 to i32
  %ext14 = zext i8 %trunc14 to i32
  %ext15 = zext i8 %trunc15 to i32
  %ext16 = zext i8 %trunc16 to i32
  %ext17 = zext i8 %trunc17 to i32
  %ext18 = zext i8 %trunc18 to i32
  %ext19 = zext i8 %trunc19 to i32
  %ext20 = zext i8 %trunc20 to i32
  %ext21 = zext i8 %trunc21 to i32
  %ext22 = zext i8 %trunc22 to i32
  %ext23 = zext i8 %trunc23 to i32
  %ext24 = zext i8 %trunc24 to i32
  %ext25 = zext i8 %trunc25 to i32
  %ext26 = zext i8 %trunc26 to i32
  %ext27 = zext i8 %trunc27 to i32
  %ext28 = zext i8 %trunc28 to i32
  %ext29 = zext i8 %trunc29 to i32
  %ext30 = zext i8 %trunc30 to i32
  %ext31 = zext i8 %trunc31 to i32

  store volatile i32 %val0, i32 *%ptr
  store volatile i32 %val1, i32 *%ptr
  store volatile i32 %val2, i32 *%ptr
  store volatile i32 %val3, i32 *%ptr
  store volatile i32 %val4, i32 *%ptr
  store volatile i32 %val5, i32 *%ptr
  store volatile i32 %val6, i32 *%ptr
  store volatile i32 %val7, i32 *%ptr
  store volatile i32 %val8, i32 *%ptr
  store volatile i32 %val9, i32 *%ptr
  store volatile i32 %val10, i32 *%ptr
  store volatile i32 %val11, i32 *%ptr
  store volatile i32 %val12, i32 *%ptr
  store volatile i32 %val13, i32 *%ptr
  store volatile i32 %val14, i32 *%ptr
  store volatile i32 %val15, i32 *%ptr
  store volatile i32 %val16, i32 *%ptr
  store volatile i32 %val17, i32 *%ptr
  store volatile i32 %val18, i32 *%ptr
  store volatile i32 %val19, i32 *%ptr
  store volatile i32 %val20, i32 *%ptr
  store volatile i32 %val21, i32 *%ptr
  store volatile i32 %val22, i32 *%ptr
  store volatile i32 %val23, i32 *%ptr
  store volatile i32 %val24, i32 *%ptr
  store volatile i32 %val25, i32 *%ptr
  store volatile i32 %val26, i32 *%ptr
  store volatile i32 %val27, i32 *%ptr
  store volatile i32 %val28, i32 *%ptr
  store volatile i32 %val29, i32 *%ptr
  store volatile i32 %val30, i32 *%ptr
  store volatile i32 %val31, i32 *%ptr

  store volatile i32 %ext0, i32 *%ptr
  store volatile i32 %ext1, i32 *%ptr
  store volatile i32 %ext2, i32 *%ptr
  store volatile i32 %ext3, i32 *%ptr
  store volatile i32 %ext4, i32 *%ptr
  store volatile i32 %ext5, i32 *%ptr
  store volatile i32 %ext6, i32 *%ptr
  store volatile i32 %ext7, i32 *%ptr
  store volatile i32 %ext8, i32 *%ptr
  store volatile i32 %ext9, i32 *%ptr
  store volatile i32 %ext10, i32 *%ptr
  store volatile i32 %ext11, i32 *%ptr
  store volatile i32 %ext12, i32 *%ptr
  store volatile i32 %ext13, i32 *%ptr
  store volatile i32 %ext14, i32 *%ptr
  store volatile i32 %ext15, i32 *%ptr
  store volatile i32 %ext16, i32 *%ptr
  store volatile i32 %ext17, i32 *%ptr
  store volatile i32 %ext18, i32 *%ptr
  store volatile i32 %ext19, i32 *%ptr
  store volatile i32 %ext20, i32 *%ptr
  store volatile i32 %ext21, i32 *%ptr
  store volatile i32 %ext22, i32 *%ptr
  store volatile i32 %ext23, i32 *%ptr
  store volatile i32 %ext24, i32 *%ptr
  store volatile i32 %ext25, i32 *%ptr
  store volatile i32 %ext26, i32 *%ptr
  store volatile i32 %ext27, i32 *%ptr
  store volatile i32 %ext28, i32 *%ptr
  store volatile i32 %ext29, i32 *%ptr
  store volatile i32 %ext30, i32 *%ptr
  store volatile i32 %ext31, i32 *%ptr

  ret void
}

; Same again with i16, which should use LLH(H).
define void @f2(i32 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: llh{{h?}} {{%r[0-9]+}}, 1{{[67]}}{{[268]}}(%r15)
; CHECK: br %r14
  %val0 = load volatile i32, i32 *%ptr
  %val1 = load volatile i32, i32 *%ptr
  %val2 = load volatile i32, i32 *%ptr
  %val3 = load volatile i32, i32 *%ptr
  %val4 = load volatile i32, i32 *%ptr
  %val5 = load volatile i32, i32 *%ptr
  %val6 = load volatile i32, i32 *%ptr
  %val7 = load volatile i32, i32 *%ptr
  %val8 = load volatile i32, i32 *%ptr
  %val9 = load volatile i32, i32 *%ptr
  %val10 = load volatile i32, i32 *%ptr
  %val11 = load volatile i32, i32 *%ptr
  %val12 = load volatile i32, i32 *%ptr
  %val13 = load volatile i32, i32 *%ptr
  %val14 = load volatile i32, i32 *%ptr
  %val15 = load volatile i32, i32 *%ptr
  %val16 = load volatile i32, i32 *%ptr
  %val17 = load volatile i32, i32 *%ptr
  %val18 = load volatile i32, i32 *%ptr
  %val19 = load volatile i32, i32 *%ptr
  %val20 = load volatile i32, i32 *%ptr
  %val21 = load volatile i32, i32 *%ptr
  %val22 = load volatile i32, i32 *%ptr
  %val23 = load volatile i32, i32 *%ptr
  %val24 = load volatile i32, i32 *%ptr
  %val25 = load volatile i32, i32 *%ptr
  %val26 = load volatile i32, i32 *%ptr
  %val27 = load volatile i32, i32 *%ptr
  %val28 = load volatile i32, i32 *%ptr
  %val29 = load volatile i32, i32 *%ptr
  %val30 = load volatile i32, i32 *%ptr
  %val31 = load volatile i32, i32 *%ptr

  %trunc0 = trunc i32 %val0 to i16
  %trunc1 = trunc i32 %val1 to i16
  %trunc2 = trunc i32 %val2 to i16
  %trunc3 = trunc i32 %val3 to i16
  %trunc4 = trunc i32 %val4 to i16
  %trunc5 = trunc i32 %val5 to i16
  %trunc6 = trunc i32 %val6 to i16
  %trunc7 = trunc i32 %val7 to i16
  %trunc8 = trunc i32 %val8 to i16
  %trunc9 = trunc i32 %val9 to i16
  %trunc10 = trunc i32 %val10 to i16
  %trunc11 = trunc i32 %val11 to i16
  %trunc12 = trunc i32 %val12 to i16
  %trunc13 = trunc i32 %val13 to i16
  %trunc14 = trunc i32 %val14 to i16
  %trunc15 = trunc i32 %val15 to i16
  %trunc16 = trunc i32 %val16 to i16
  %trunc17 = trunc i32 %val17 to i16
  %trunc18 = trunc i32 %val18 to i16
  %trunc19 = trunc i32 %val19 to i16
  %trunc20 = trunc i32 %val20 to i16
  %trunc21 = trunc i32 %val21 to i16
  %trunc22 = trunc i32 %val22 to i16
  %trunc23 = trunc i32 %val23 to i16
  %trunc24 = trunc i32 %val24 to i16
  %trunc25 = trunc i32 %val25 to i16
  %trunc26 = trunc i32 %val26 to i16
  %trunc27 = trunc i32 %val27 to i16
  %trunc28 = trunc i32 %val28 to i16
  %trunc29 = trunc i32 %val29 to i16
  %trunc30 = trunc i32 %val30 to i16
  %trunc31 = trunc i32 %val31 to i16

  %ext0 = zext i16 %trunc0 to i32
  %ext1 = zext i16 %trunc1 to i32
  %ext2 = zext i16 %trunc2 to i32
  %ext3 = zext i16 %trunc3 to i32
  %ext4 = zext i16 %trunc4 to i32
  %ext5 = zext i16 %trunc5 to i32
  %ext6 = zext i16 %trunc6 to i32
  %ext7 = zext i16 %trunc7 to i32
  %ext8 = zext i16 %trunc8 to i32
  %ext9 = zext i16 %trunc9 to i32
  %ext10 = zext i16 %trunc10 to i32
  %ext11 = zext i16 %trunc11 to i32
  %ext12 = zext i16 %trunc12 to i32
  %ext13 = zext i16 %trunc13 to i32
  %ext14 = zext i16 %trunc14 to i32
  %ext15 = zext i16 %trunc15 to i32
  %ext16 = zext i16 %trunc16 to i32
  %ext17 = zext i16 %trunc17 to i32
  %ext18 = zext i16 %trunc18 to i32
  %ext19 = zext i16 %trunc19 to i32
  %ext20 = zext i16 %trunc20 to i32
  %ext21 = zext i16 %trunc21 to i32
  %ext22 = zext i16 %trunc22 to i32
  %ext23 = zext i16 %trunc23 to i32
  %ext24 = zext i16 %trunc24 to i32
  %ext25 = zext i16 %trunc25 to i32
  %ext26 = zext i16 %trunc26 to i32
  %ext27 = zext i16 %trunc27 to i32
  %ext28 = zext i16 %trunc28 to i32
  %ext29 = zext i16 %trunc29 to i32
  %ext30 = zext i16 %trunc30 to i32
  %ext31 = zext i16 %trunc31 to i32

  store volatile i32 %val0, i32 *%ptr
  store volatile i32 %val1, i32 *%ptr
  store volatile i32 %val2, i32 *%ptr
  store volatile i32 %val3, i32 *%ptr
  store volatile i32 %val4, i32 *%ptr
  store volatile i32 %val5, i32 *%ptr
  store volatile i32 %val6, i32 *%ptr
  store volatile i32 %val7, i32 *%ptr
  store volatile i32 %val8, i32 *%ptr
  store volatile i32 %val9, i32 *%ptr
  store volatile i32 %val10, i32 *%ptr
  store volatile i32 %val11, i32 *%ptr
  store volatile i32 %val12, i32 *%ptr
  store volatile i32 %val13, i32 *%ptr
  store volatile i32 %val14, i32 *%ptr
  store volatile i32 %val15, i32 *%ptr
  store volatile i32 %val16, i32 *%ptr
  store volatile i32 %val17, i32 *%ptr
  store volatile i32 %val18, i32 *%ptr
  store volatile i32 %val19, i32 *%ptr
  store volatile i32 %val20, i32 *%ptr
  store volatile i32 %val21, i32 *%ptr
  store volatile i32 %val22, i32 *%ptr
  store volatile i32 %val23, i32 *%ptr
  store volatile i32 %val24, i32 *%ptr
  store volatile i32 %val25, i32 *%ptr
  store volatile i32 %val26, i32 *%ptr
  store volatile i32 %val27, i32 *%ptr
  store volatile i32 %val28, i32 *%ptr
  store volatile i32 %val29, i32 *%ptr
  store volatile i32 %val30, i32 *%ptr
  store volatile i32 %val31, i32 *%ptr

  store volatile i32 %ext0, i32 *%ptr
  store volatile i32 %ext1, i32 *%ptr
  store volatile i32 %ext2, i32 *%ptr
  store volatile i32 %ext3, i32 *%ptr
  store volatile i32 %ext4, i32 *%ptr
  store volatile i32 %ext5, i32 *%ptr
  store volatile i32 %ext6, i32 *%ptr
  store volatile i32 %ext7, i32 *%ptr
  store volatile i32 %ext8, i32 *%ptr
  store volatile i32 %ext9, i32 *%ptr
  store volatile i32 %ext10, i32 *%ptr
  store volatile i32 %ext11, i32 *%ptr
  store volatile i32 %ext12, i32 *%ptr
  store volatile i32 %ext13, i32 *%ptr
  store volatile i32 %ext14, i32 *%ptr
  store volatile i32 %ext15, i32 *%ptr
  store volatile i32 %ext16, i32 *%ptr
  store volatile i32 %ext17, i32 *%ptr
  store volatile i32 %ext18, i32 *%ptr
  store volatile i32 %ext19, i32 *%ptr
  store volatile i32 %ext20, i32 *%ptr
  store volatile i32 %ext21, i32 *%ptr
  store volatile i32 %ext22, i32 *%ptr
  store volatile i32 %ext23, i32 *%ptr
  store volatile i32 %ext24, i32 *%ptr
  store volatile i32 %ext25, i32 *%ptr
  store volatile i32 %ext26, i32 *%ptr
  store volatile i32 %ext27, i32 *%ptr
  store volatile i32 %ext28, i32 *%ptr
  store volatile i32 %ext29, i32 *%ptr
  store volatile i32 %ext30, i32 *%ptr
  store volatile i32 %ext31, i32 *%ptr

  ret void
}
