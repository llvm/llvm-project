; RUN: llc < %s -march=arm64 -aarch64-neon-syntax=apple -mcpu=cyclone | FileCheck %s
%struct.X = type <{ i32, i64, i64 }>

define void @foo1(i32* %p, i64 %val) nounwind {
; CHECK-LABEL: foo1:
; CHECK: 	stur	w1, [x0, #-4]
; CHECK-NEXT: 	ret
  %tmp1 = trunc i64 %val to i32
  %ptr = getelementptr inbounds i32, i32* %p, i64 -1
  store i32 %tmp1, i32* %ptr, align 4
  ret void
}
define void @foo2(i16* %p, i64 %val) nounwind {
; CHECK-LABEL: foo2:
; CHECK: 	sturh	w1, [x0, #-2]
; CHECK-NEXT: 	ret
  %tmp1 = trunc i64 %val to i16
  %ptr = getelementptr inbounds i16, i16* %p, i64 -1
  store i16 %tmp1, i16* %ptr, align 2
  ret void
}
define void @foo3(i8* %p, i64 %val) nounwind {
; CHECK-LABEL: foo3:
; CHECK: 	sturb	w1, [x0, #-1]
; CHECK-NEXT: 	ret
  %tmp1 = trunc i64 %val to i8
  %ptr = getelementptr inbounds i8, i8* %p, i64 -1
  store i8 %tmp1, i8* %ptr, align 1
  ret void
}
define void @foo4(i16* %p, i32 %val) nounwind {
; CHECK-LABEL: foo4:
; CHECK: 	sturh	w1, [x0, #-2]
; CHECK-NEXT: 	ret
  %tmp1 = trunc i32 %val to i16
  %ptr = getelementptr inbounds i16, i16* %p, i32 -1
  store i16 %tmp1, i16* %ptr, align 2
  ret void
}
define void @foo5(i8* %p, i32 %val) nounwind {
; CHECK-LABEL: foo5:
; CHECK: 	sturb	w1, [x0, #-1]
; CHECK-NEXT: 	ret
  %tmp1 = trunc i32 %val to i8
  %ptr = getelementptr inbounds i8, i8* %p, i32 -1
  store i8 %tmp1, i8* %ptr, align 1
  ret void
}

define void @foo(%struct.X* nocapture %p) nounwind optsize ssp {
; CHECK-LABEL: foo:
; CHECK-NOT: str
; CHECK: stur    xzr, [x0, #12]
; CHECK-NEXT: stur    xzr, [x0, #4]
; CHECK-NEXT: ret
  %B = getelementptr inbounds %struct.X, %struct.X* %p, i64 0, i32 1
  %val = bitcast i64* %B to i8*
  call void @llvm.memset.p0i8.i64(i8* %val, i8 0, i64 16, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind

; Unaligned 16b stores are split into 8b stores for performance.
; radar://15424193

; CHECK-LABEL: unaligned:
; CHECK-NOT: str q0
; CHECK: str     d[[REG:[0-9]+]], [x0]
; CHECK: ext.16b v[[REG2:[0-9]+]], v[[REG]], v[[REG]], #8
; CHECK: str     d[[REG2]], [x0, #8]
define void @unaligned(<4 x i32>* %p, <4 x i32> %v) nounwind {
  store <4 x i32> %v, <4 x i32>* %p, align 4
  ret void
}

; CHECK-LABEL: aligned:
; CHECK: str q0
define void @aligned(<4 x i32>* %p, <4 x i32> %v) nounwind {
  store <4 x i32> %v, <4 x i32>* %p
  ret void
}

; Don't split one and two byte aligned stores.
; radar://16349308

; CHECK-LABEL: twobytealign:
; CHECK: str q0
define void @twobytealign(<4 x i32>* %p, <4 x i32> %v) nounwind {
  store <4 x i32> %v, <4 x i32>* %p, align 2
  ret void
}
; CHECK-LABEL: onebytealign:
; CHECK: str q0
define void @onebytealign(<4 x i32>* %p, <4 x i32> %v) nounwind {
  store <4 x i32> %v, <4 x i32>* %p, align 1
  ret void
}
