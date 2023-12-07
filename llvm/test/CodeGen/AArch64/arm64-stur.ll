; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -mcpu=cyclone -mattr=+slow-misaligned-128store | FileCheck %s
%struct.X = type <{ i32, i64, i64 }>

define void @foo1(ptr %p, i64 %val) nounwind {
; CHECK-LABEL: foo1:
; CHECK: 	stur	w1, [x0, #-4]
; CHECK-NEXT: 	ret
  %tmp1 = trunc i64 %val to i32
  %ptr = getelementptr inbounds i32, ptr %p, i64 -1
  store i32 %tmp1, ptr %ptr, align 4
  ret void
}
define void @foo2(ptr %p, i64 %val) nounwind {
; CHECK-LABEL: foo2:
; CHECK: 	sturh	w1, [x0, #-2]
; CHECK-NEXT: 	ret
  %tmp1 = trunc i64 %val to i16
  %ptr = getelementptr inbounds i16, ptr %p, i64 -1
  store i16 %tmp1, ptr %ptr, align 2
  ret void
}
define void @foo3(ptr %p, i64 %val) nounwind {
; CHECK-LABEL: foo3:
; CHECK: 	sturb	w1, [x0, #-1]
; CHECK-NEXT: 	ret
  %tmp1 = trunc i64 %val to i8
  %ptr = getelementptr inbounds i8, ptr %p, i64 -1
  store i8 %tmp1, ptr %ptr, align 1
  ret void
}
define void @foo4(ptr %p, i32 %val) nounwind {
; CHECK-LABEL: foo4:
; CHECK: 	sturh	w1, [x0, #-2]
; CHECK-NEXT: 	ret
  %tmp1 = trunc i32 %val to i16
  %ptr = getelementptr inbounds i16, ptr %p, i32 -1
  store i16 %tmp1, ptr %ptr, align 2
  ret void
}
define void @foo5(ptr %p, i32 %val) nounwind {
; CHECK-LABEL: foo5:
; CHECK: 	sturb	w1, [x0, #-1]
; CHECK-NEXT: 	ret
  %tmp1 = trunc i32 %val to i8
  %ptr = getelementptr inbounds i8, ptr %p, i32 -1
  store i8 %tmp1, ptr %ptr, align 1
  ret void
}

define void @foo(ptr nocapture %p) nounwind optsize ssp {
; CHECK-LABEL: foo:
; CHECK-NOT: str
; CHECK: stur    xzr, [x0, #12]
; CHECK-NEXT: stur    xzr, [x0, #4]
; CHECK-NEXT: ret
  %B = getelementptr inbounds %struct.X, ptr %p, i64 0, i32 1
  call void @llvm.memset.p0.i64(ptr %B, i8 0, i64 16, i1 false)
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) nounwind

; Unaligned 16b stores are split into 8b stores for performance.
; radar://15424193

; CHECK-LABEL: unaligned:
; CHECK-NOT: str q0
; CHECK: str     d[[REG:[0-9]+]], [x0]
; CHECK: ext.16b v[[REG2:[0-9]+]], v[[REG]], v[[REG]], #8
; CHECK: str     d[[REG2]], [x0, #8]
define void @unaligned(ptr %p, <4 x i32> %v) nounwind {
  store <4 x i32> %v, ptr %p, align 4
  ret void
}

; CHECK-LABEL: aligned:
; CHECK: str q0
define void @aligned(ptr %p, <4 x i32> %v) nounwind {
  store <4 x i32> %v, ptr %p
  ret void
}

; Don't split one and two byte aligned stores.
; radar://16349308

; CHECK-LABEL: twobytealign:
; CHECK: str q0
define void @twobytealign(ptr %p, <4 x i32> %v) nounwind {
  store <4 x i32> %v, ptr %p, align 2
  ret void
}
; CHECK-LABEL: onebytealign:
; CHECK: str q0
define void @onebytealign(ptr %p, <4 x i32> %v) nounwind {
  store <4 x i32> %v, ptr %p, align 1
  ret void
}
