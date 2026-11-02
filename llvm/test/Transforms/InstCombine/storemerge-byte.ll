; RUN: opt < %s -passes=instcombine -S -verify-each | FileCheck %s

target datalayout = "e-p:64:64"

; A byte value can contain pointer provenance or partially poisoned bits.
; Casting it to a non-byte type before storing need not preserve either.
define void @byte_int(ptr %p, i1 %c, b16 %a, i16 %b) {
; CHECK-LABEL: @byte_int(
; CHECK-NEXT: entry:
; CHECK-NEXT: br i1 %c, label %then, label %else
; CHECK: then:
; CHECK-NEXT: br label %join
; CHECK: else:
; CHECK-NEXT: [[BYTES:%.*]] = bitcast i16 %b to b16
; CHECK-NEXT: br label %join
; CHECK: join:
; CHECK-NEXT: [[MERGED:%.*]] = phi b16 [ %a, %then ], [ [[BYTES]], %else ]
; CHECK-NEXT: store b16 [[MERGED]], ptr %p, align 2
; CHECK-NEXT: ret void
entry:
  br i1 %c, label %then, label %else
then:
  store b16 %a, ptr %p
  br label %join
else:
  store i16 %b, ptr %p
  br label %join
join:
  ret void
}

define void @int_byte(ptr %p, i1 %c, i16 %a, b16 %b) {
; CHECK-LABEL: @int_byte(
; CHECK: [[CAST:%.*]] = bitcast i16 %a to b16
; CHECK: %storemerge = phi b16
; CHECK-NEXT: store b16 %storemerge, ptr %p, align 2
; CHECK-NEXT: ret void
entry:
  br i1 %c, label %then, label %else
then:
  store i16 %a, ptr %p
  br label %join
else:
  store b16 %b, ptr %p
  br label %join
join:
  ret void
}

define void @byte_ptr(ptr %p, i1 %c, b64 %a, ptr %b) {
; CHECK-LABEL: @byte_ptr(
; CHECK-NOT: bitcast b64
; CHECK: store b64
; CHECK-NOT: bitcast b64
; CHECK: ret void
entry:
  br i1 %c, label %then, label %else
then:
  store b64 %a, ptr %p, align 8
  br label %join
else:
  store ptr %b, ptr %p
  br label %join
join:
  ret void
}

define void @ptr_byte(ptr %p, i1 %c, ptr %a, b64 %b) {
; CHECK-LABEL: @ptr_byte(
; CHECK-NOT: bitcast b64
; CHECK: store b64
; CHECK-NOT: bitcast b64
; CHECK: ret void
entry:
  br i1 %c, label %then, label %else
then:
  store ptr %a, ptr %p
  br label %join
else:
  store b64 %b, ptr %p, align 8
  br label %join
join:
  ret void
}

define void @byte_int_vector(ptr %p, i1 %c, <2 x b16> %a, <2 x i16> %b) {
; CHECK-LABEL: @byte_int_vector(
; CHECK-NOT: bitcast <2 x b16>
; CHECK: store <2 x b16>
; CHECK-NOT: bitcast <2 x b16>
; CHECK: ret void
entry:
  br i1 %c, label %then, label %else
then:
  store <2 x b16> %a, ptr %p
  br label %join
else:
  store <2 x i16> %b, ptr %p
  br label %join
join:
  ret void
}

define void @int_byte_vector(ptr %p, i1 %c, <2 x i16> %a, <2 x b16> %b) {
; CHECK-LABEL: @int_byte_vector(
; CHECK: [[CAST:%.*]] = bitcast <2 x i16> %a to <2 x b16>
; CHECK: %storemerge = phi <2 x b16>
; CHECK-NEXT: store <2 x b16> %storemerge, ptr %p, align 4
; CHECK-NEXT: ret void
entry:
  br i1 %c, label %then, label %else
then:
  store <2 x i16> %a, ptr %p
  br label %join
else:
  store <2 x b16> %b, ptr %p
  br label %join
join:
  ret void
}

; Exercise the if/then form as well as the if/then/else form above.
define void @byte_int_triangle(ptr %p, i1 %c, b16 %a, i16 %b) {
; CHECK-LABEL: @byte_int_triangle(
; CHECK-NOT: bitcast b16
; CHECK: store b16
; CHECK-NOT: bitcast b16
; CHECK: ret void
entry:
  store b16 %a, ptr %p
  br i1 %c, label %then, label %join
then:
  store i16 %b, ptr %p
  br label %join
join:
  ret void
}

define void @int_byte_triangle(ptr %p, i1 %c, i16 %a, b16 %b) {
; CHECK-LABEL: @int_byte_triangle(
; CHECK: [[CAST:%.*]] = bitcast i16 %a to b16
; CHECK: %storemerge = phi b16
; CHECK-NEXT: store b16 %storemerge, ptr %p, align 2
; CHECK-NEXT: ret void
entry:
  store i16 %a, ptr %p
  br i1 %c, label %then, label %join
then:
  store b16 %b, ptr %p
  br label %join
join:
  ret void
}

; Merging byte values without a lossy cast is still allowed.
define void @byte_byte(ptr %p, i1 %c, b16 %a, b16 %b) {
; CHECK-LABEL: @byte_byte(
; CHECK: %storemerge = phi b16
; CHECK-NEXT: store b16 %storemerge, ptr %p, align 2
; CHECK-NEXT: ret void
entry:
  br i1 %c, label %then, label %else
then:
  store b16 %a, ptr %p
  br label %join
else:
  store b16 %b, ptr %p
  br label %join
join:
  ret void
}

define void @byte_byte_vector(ptr %p, i1 %c, <2 x b16> %a, <2 x b16> %b) {
; CHECK-LABEL: @byte_byte_vector(
; CHECK: %storemerge = phi <2 x b16>
; CHECK-NEXT: store <2 x b16> %storemerge, ptr %p, align 4
; CHECK-NEXT: ret void
entry:
  br i1 %c, label %then, label %else
then:
  store <2 x b16> %a, ptr %p
  br label %join
else:
  store <2 x b16> %b, ptr %p
  br label %join
join:
  ret void
}

define void @byte_float(ptr %p, i1 %c, b32 %a, float %b) {
; CHECK-LABEL: @byte_float(
; CHECK-NOT: bitcast b32
; CHECK: store b32
; CHECK-NOT: bitcast b32
; CHECK: ret void
entry:
  br i1 %c, label %then, label %else
then:
  store b32 %a, ptr %p
  br label %join
else:
  store float %b, ptr %p
  br label %join
join:
  ret void
}

; Changing the shape of bytes preserves their per-bit information.
define void @byte_reshape(ptr %p, i1 %c, b32 %a, <2 x b16> %b) {
; CHECK-LABEL: @byte_reshape(
; CHECK: bitcast b32 %a to <2 x b16>
; CHECK: %storemerge = phi <2 x b16>
; CHECK-NEXT: store <2 x b16> %storemerge, ptr %p, align 4
; CHECK-NEXT: ret void
entry:
  br i1 %c, label %then, label %else
then:
  store b32 %a, ptr %p
  br label %join
else:
  store <2 x b16> %b, ptr %p
  br label %join
join:
  ret void
}
