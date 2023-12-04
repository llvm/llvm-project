; RUN: llc -o - %s -mtriple=aarch64-none-linux-gnu | FileCheck %s
; RUN: llc -o - %s -mtriple=aarch64-none-linux-gnu -aarch64-optimize-rounding=false -aarch64-optimize-rounding-saturation=false -aarch64-extract-vector-element-trunc-combine=false | FileCheck %s --check-prefix=NOROUNDING

target triple = "aarch64-unknown-linux-gnu"

define void @test_srshr(i8* nocapture %dst, i8* nocapture readonly %pix1) {
; CHECK-LABEL: test_srshr:
; CHECK:    ldr
; NOROUNDING: ldr
; CHECK:    srshr
; NOROUNDING-NOT:: srshr
; CHECK-NOT:   sshr
; CHECK:    str
; NOROUNDING: ret
; CHECK:    ret
entry:
  %0 = bitcast i8* %pix1 to <4 x i32>*
  %1 = load <4 x i32>, <4 x i32>* %0, align 1
  %2 = add nuw nsw <4 x i32> %1, <i32 32, i32 32, i32 32, i32 32>
  %3 = ashr <4 x i32> %2, <i32 6, i32 6, i32 6, i32 6>
  %4 = bitcast i8* %dst to <4 x i32>*
  store <4 x i32> %3, <4 x i32>* %4, align 1
  ret void
}

define void @test_srshr2(i8* nocapture %dst, i8* nocapture readonly %pix1) {
; CHECK-LABEL: test_srshr2:
; CHECK:    ldr
; NOROUNDING: ldr
; CHECK:    sqrshrun
; CHECK:    uqxtn
; NOROUNDING-NOT: sqrshrun
; NOROUNDING-NOT: uqxtn
; CHECK-NOT:   sshr
; CHECK:    str
; NOROUNDING: ret
; CHECK:    ret
entry:
  %0 = bitcast i8* %pix1 to <4 x i32>*
  %1 = load <4 x i32>, <4 x i32>* %0, align 1
  %2 = add nuw nsw <4 x i32> %1, <i32 32, i32 32, i32 32, i32 32>
  %3 = ashr <4 x i32> %2, <i32 6, i32 6, i32 6, i32 6>
  %4 = icmp sgt <4 x i32> %3, zeroinitializer
  %5 = select <4 x i1> %4, <4 x i32> %3, <4 x i32> zeroinitializer
  %6 = icmp ult <4 x i32> %5, <i32 255, i32 255, i32 255, i32 255>
  %7 = select <4 x i1> %6, <4 x i32> %5, <4 x i32> <i32 255, i32 255, i32 255, i32 255>
  %8 = trunc <4 x i32> %7 to <4 x i8>
  %9 = bitcast i8* %dst to <4 x i8>*
  store <4 x i8> %8, <4 x i8>* %9, align 1
  ret void
}

define void @test_srshr3(i8* nocapture %dst, i8* nocapture readonly %pix1, i8* nocapture readonly %pix2) {
; CHECK-LABEL: test_srshr3:
; CHECK:    ldr
; NOROUNDING: ldr
; CHECK:    add
; CHECK:    srshr
; NOROUNDING-NOT: srshr
; CHECK-NOT:   sshr
; CHECK:    str
; NOROUNDING: ret
; CHECK:    ret
entry:
  %0 = bitcast i8* %pix1 to <4 x i32>*
  %1 = load <4 x i32>, <4 x i32>* %0, align 1
  %2 = bitcast i8* %pix2 to <4 x i32>*
  %3 = load <4 x i32>, <4 x i32>* %2, align 1
  %4 = add nuw nsw <4 x i32> %1, <i32 32, i32 32, i32 32, i32 32>
  %5 = add nuw nsw <4 x i32> %4, %3
  %6 = ashr <4 x i32> %5, <i32 6, i32 6, i32 6, i32 6>
  %7 = bitcast i8* %dst to <4 x i32>*
  store <4 x i32> %6, <4 x i32>* %7, align 1
  ret void
}


define void @test_sqrshrun(i8* nocapture %dst, i8* nocapture readonly %pix1) {
; CHECK-LABEL: test_sqrshrun:
; CHECK:    ldr
; NOROUNDING: ldr
; CHECK:    sqrshrun
; NOROUNDING-NOT: sqrshrun
; CHECK-NOT:   sshr
; CHECK-NOT:   smax
; CHECK:    str
; NOROUNDING: ret
; CHECK:    ret
entry:
  %0 = bitcast i8* %pix1 to <4 x i32>*
  %1 = load <4 x i32>, <4 x i32>* %0, align 1
  %2 = add nuw nsw <4 x i32> %1, <i32 32, i32 32, i32 32, i32 32>
  %3 = ashr <4 x i32> %2, <i32 6, i32 6, i32 6, i32 6>
  %4 = icmp sgt <4 x i32> %3, zeroinitializer
  %5 = select <4 x i1> %4, <4 x i32> %3, <4 x i32> zeroinitializer
  %6 = icmp ult <4 x i32> %5, <i32 65535, i32 65535, i32 65535, i32 65535>
  %7 = select <4 x i1> %6, <4 x i32> %5, <4 x i32> <i32 65535, i32 65535, i32 65535, i32 65535>
  %8 = trunc <4 x i32> %7 to <4 x i16>
  %9 = bitcast i8* %dst to <4 x i16>*
  store <4 x i16> %8, <4 x i16>* %9, align 1
  ret void
}



define void @test_urshr(i8* nocapture %dst, i8* nocapture readonly %pix1) {
; CHECK-LABEL: test_urshr:
; CHECK:    ldr
; NOROUNDING: ldr
; CHECK:    urshr
; NOROUNDING-NOT: urshr
; CHECK-NOT:   ushr
; CHECK:    str
; NOROUNDING: ret
; CHECK:    ret
entry:
  %0 = bitcast i8* %pix1 to <4 x i32>*
  %1 = load <4 x i32>, <4 x i32>* %0, align 1
  %2 = add nuw nsw <4 x i32> %1, <i32 32, i32 32, i32 32, i32 32>
  %3 = lshr <4 x i32> %2, <i32 6, i32 6, i32 6, i32 6>
  %4 = bitcast i8* %dst to <4 x i32>*
  store <4 x i32> %3, <4 x i32>* %4, align 1
  ret void
}


define void @test_sqrshrun2(i8* nocapture %dst, i8* nocapture readonly %pix1) {
; CHECK-LABEL: test_sqrshrun2:
; CHECK:    ldp
; NOROUNDING: ldp
; CHECK:    sqrshrun
; CHECK:    sqrshrun2
; NOROUNDING-NOT: sqrshrun
; NOROUNDING-NOT: sqrshrun2
; CHECK-NOT:   ushr
; CHECK:    str
; NOROUNDING: ret
; CHECK:    ret
entry:
  %0 = bitcast i8* %pix1 to <8 x i32>*
  %1 = load <8 x i32>, <8 x i32>* %0, align 1
  %2 = add nuw nsw <8 x i32> %1, <i32 32, i32 32, i32 32, i32 32, i32 32, i32 32, i32 32, i32 32>
  %3 = ashr <8 x i32> %2, <i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6>
  %4 = icmp sgt <8 x i32> %3, zeroinitializer
  %5 = select <8 x i1> %4, <8 x i32> %3, <8 x i32> zeroinitializer
  %6 = icmp ult <8 x i32> %5, <i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255>
  %7 = select <8 x i1> %6, <8 x i32> %5, <8 x i32> <i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255>
  %8 = trunc <8 x i32> %7 to <8 x i8>
  %9 = bitcast i8* %dst to <8 x i8>*
  store <8 x i8> %8, <8 x i8>* %9, align 1
  ret void
}

define void @test_uqrshrn(i8* nocapture %dst, i8* nocapture readonly %pix1) {
; CHECK-LABEL: test_uqrshrn:
; CHECK:    ldr
; NOROUNDING: ldr
; CHECK:    uqrshrn
; NOROUNDING-NOT: uqrshrn
; CHECK-NOT:   ushr
; CHECK:    str
; NOROUNDING: ret
; CHECK:    ret
entry:
  %0 = bitcast i8* %pix1 to <4 x i32>*
  %1 = load <4 x i32>, <4 x i32>* %0, align 1
  %2 = add nuw nsw <4 x i32> %1, <i32 32, i32 32, i32 32, i32 32>
  %3 = lshr <4 x i32> %2, <i32 6, i32 6, i32 6, i32 6>
  %4 = icmp sgt <4 x i32> %3, zeroinitializer
  %5 = select <4 x i1> %4, <4 x i32> %3, <4 x i32> zeroinitializer
  %6 = icmp ult <4 x i32> %5, <i32 65535, i32 65535, i32 65535, i32 65535>
  %7 = select <4 x i1> %6, <4 x i32> %5, <4 x i32> <i32 65535, i32 65535, i32 65535, i32 65535>
  %8 = trunc <4 x i32> %7 to <4 x i16>
  %9 = bitcast i8* %dst to <4 x i16>*
  store <4 x i16> %8, <4 x i16>* %9, align 1
  ret void
}

define void @test_srshr_long_shift(i8* nocapture %dst, i8* nocapture readonly %pix1) {
; CHECK-LABEL: test_srshr_long_shift:
; CHECK:    ldr
; NOROUNDING: ldr
; CHECK:    srshr
; NOROUNDING-NOT:: srshr
; CHECK-NOT:   sshr
; CHECK:    str
; NOROUNDING: ret
; CHECK:    ret
entry:
  %0 = bitcast i8* %pix1 to <4 x i32>*
  %1 = load <4 x i32>, <4 x i32>* %0, align 1
  %2 = add nuw nsw <4 x i32> %1, <i32 2097152, i32 2097152, i32 2097152, i32 2097152>
  %3 = ashr <4 x i32> %2, <i32 22, i32 22, i32 22, i32 22>
  %4 = bitcast i8* %dst to <4 x i32>*
  store <4 x i32> %3, <4 x i32>* %4, align 1
  ret void
}

define void @test_urshr_long_shift(i8* nocapture %dst, i8* nocapture readonly %pix1) {
; CHECK-LABEL: test_urshr_long_shift:
; CHECK:    ldr
; NOROUNDING: ldr
; CHECK:    urshr
; NOROUNDING-NOT: urshr
; CHECK-NOT:   ushr
; CHECK:    str
; NOROUNDING: ret
; CHECK:    ret
entry:
  %0 = bitcast i8* %pix1 to <4 x i32>*
  %1 = load <4 x i32>, <4 x i32>* %0, align 1
  %2 = add nuw nsw <4 x i32> %1, <i32 2097152, i32 2097152, i32 2097152, i32 2097152>
  %3 = lshr <4 x i32> %2, <i32 22, i32 22, i32 22, i32 22>
  %4 = bitcast i8* %dst to <4 x i32>*
  store <4 x i32> %3, <4 x i32>* %4, align 1
  ret void
}

; Negative test: Rounding+Truncation instruction doesn't support shift amount > input data with / 2
define void @test_srshr2_long_shift(i8* nocapture %dst, i8* nocapture readonly %pix1) {
; CHECK-LABEL: test_srshr2_long_shift:
; CHECK:    ldr
; NOROUNDING: ldr
; CHECK:   srshr
; CHECK-NOT:    sqrshrun
; CHECK-NOT:    uqxtn
; NOROUNDING-NOT: sqrshrun
; NOROUNDING-NOT: uqxtn
; CHECK:    str
; NOROUNDING: ret
; CHECK:    ret
entry:
  %0 = bitcast i8* %pix1 to <4 x i32>*
  %1 = load <4 x i32>, <4 x i32>* %0, align 1
  %2 = add nuw nsw <4 x i32> %1, <i32 2097152, i32 2097152, i32 2097152, i32 2097152>
  %3 = ashr <4 x i32> %2, <i32 22, i32 22, i32 22, i32 22>
  %4 = icmp sgt <4 x i32> %3, zeroinitializer
  %5 = select <4 x i1> %4, <4 x i32> %3, <4 x i32> zeroinitializer
  %6 = icmp ult <4 x i32> %5, <i32 255, i32 255, i32 255, i32 255>
  %7 = select <4 x i1> %6, <4 x i32> %5, <4 x i32> <i32 255, i32 255, i32 255, i32 255>
  %8 = trunc <4 x i32> %7 to <4 x i8>
  %9 = bitcast i8* %dst to <4 x i8>*
  store <4 x i8> %8, <4 x i8>* %9, align 1
  ret void
}


; Negative test: Rounding+Truncation instruction doesn't support shift amount > input data with / 2
define void @test_sqrshrun_long_shift(i8* nocapture %dst, i8* nocapture readonly %pix1) {
; CHECK-LABEL: test_sqrshrun_long_shift:
; CHECK:    ldr
; NOROUNDING: ldr
; CHECK:   srshr
; CHECK-NOT:    sqrshrun
; NOROUNDING-NOT: sqrshrun
; CHECK:   smax
; CHECK:    str
; NOROUNDING: ret
; CHECK:    ret
entry:
  %0 = bitcast i8* %pix1 to <4 x i32>*
  %1 = load <4 x i32>, <4 x i32>* %0, align 1
  %2 = add nuw nsw <4 x i32> %1, <i32 2097152, i32 2097152, i32 2097152, i32 2097152>
  %3 = ashr <4 x i32> %2, <i32 22, i32 22, i32 22, i32 22>
  %4 = icmp sgt <4 x i32> %3, zeroinitializer
  %5 = select <4 x i1> %4, <4 x i32> %3, <4 x i32> zeroinitializer
  %6 = icmp ult <4 x i32> %5, <i32 65535, i32 65535, i32 65535, i32 65535>
  %7 = select <4 x i1> %6, <4 x i32> %5, <4 x i32> <i32 65535, i32 65535, i32 65535, i32 65535>
  %8 = trunc <4 x i32> %7 to <4 x i16>
  %9 = bitcast i8* %dst to <4 x i16>*
  store <4 x i16> %8, <4 x i16>* %9, align 1
  ret void
}

; Negative test: Rounding+Truncation instruction doesn't support shift amount > input data with / 2
define void @test_sqrshrun2_long_shift(i8* nocapture %dst, i8* nocapture readonly %pix1) {
; CHECK-LABEL: test_sqrshrun2_long_shift:
; CHECK:    ldp
; CHECK:   srshr
; NOROUNDING: ldp
; CHECK-NOT:    sqrshrun
; CHECK-NOT:    sqrshrun2
; NOROUNDING-NOT: sqrshrun
; NOROUNDING-NOT: sqrshrun2
; CHECK:    str
; NOROUNDING: ret
; CHECK:    ret
entry:
  %0 = bitcast i8* %pix1 to <8 x i32>*
  %1 = load <8 x i32>, <8 x i32>* %0, align 1
  %2 = add nuw nsw <8 x i32> %1, <i32 2097152, i32 2097152, i32 2097152, i32 2097152, i32 2097152, i32 2097152, i32 2097152, i32 2097152>
  %3 = ashr <8 x i32> %2, <i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22>
  %4 = icmp sgt <8 x i32> %3, zeroinitializer
  %5 = select <8 x i1> %4, <8 x i32> %3, <8 x i32> zeroinitializer
  %6 = icmp ult <8 x i32> %5, <i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255>
  %7 = select <8 x i1> %6, <8 x i32> %5, <8 x i32> <i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255, i32 255>
  %8 = trunc <8 x i32> %7 to <8 x i8>
  %9 = bitcast i8* %dst to <8 x i8>*
  store <8 x i8> %8, <8 x i8>* %9, align 1
  ret void
}

; Negative test: Rounding+Truncation instruction doesn't support shift amount > input data with / 2
define void @test_uqrshrn_long_shift(i8* nocapture %dst, i8* nocapture readonly %pix1) {
; CHECK-LABEL: test_uqrshrn_long_shift:
; CHECK:    ldr
; NOROUNDING: ldr
; CHECK:   urshr
; CHECK-NOT:    uqrshrn
; NOROUNDING-NOT: uqrshrn
; CHECK:    str
; NOROUNDING: ret
; CHECK:    ret
entry:
  %0 = bitcast i8* %pix1 to <4 x i32>*
  %1 = load <4 x i32>, <4 x i32>* %0, align 1
  %2 = add nuw nsw <4 x i32> %1, <i32 2097152, i32 2097152, i32 2097152, i32 2097152>
  %3 = lshr <4 x i32> %2, <i32 22, i32 22, i32 22, i32 22>
  %4 = icmp sgt <4 x i32> %3, zeroinitializer
  %5 = select <4 x i1> %4, <4 x i32> %3, <4 x i32> zeroinitializer
  %6 = icmp ult <4 x i32> %5, <i32 65535, i32 65535, i32 65535, i32 65535>
  %7 = select <4 x i1> %6, <4 x i32> %5, <4 x i32> <i32 65535, i32 65535, i32 65535, i32 65535>
  %8 = trunc <4 x i32> %7 to <4 x i16>
  %9 = bitcast i8* %dst to <4 x i16>*
  store <4 x i16> %8, <4 x i16>* %9, align 1
  ret void
}
