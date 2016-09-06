; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+sse2 -cost-model -analyze < %s | FileCheck --check-prefix=SSE2 %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+sse4.1 -cost-model -analyze < %s | FileCheck --check-prefix=SSE41 %s

define void @zext_v4i8_to_v4i64(<4 x i8>* %a) {
; SSE2: zext_v4i8_to_v4i64
; SSE2: cost of 4 {{.*}} zext
;
; SSE41: zext_v4i8_to_v4i64
; SSE41: cost of 2 {{.*}} zext
;
  %1 = load <4 x i8>, <4 x i8>* %a
  %2 = zext <4 x i8> %1 to <4 x i64>
  store <4 x i64> %2, <4 x i64>* undef, align 4
  ret void
}

define void @sext_v4i8_to_v4i64(<4 x i8>* %a) {
; SSE2: sext_v4i8_to_v4i64
; SSE2: cost of 8 {{.*}} sext
;
; SSE41: sext_v4i8_to_v4i64
; SSE41: cost of 2 {{.*}} sext
;
  %1 = load <4 x i8>, <4 x i8>* %a
  %2 = sext <4 x i8> %1 to <4 x i64>
  store <4 x i64> %2, <4 x i64>* undef, align 4
  ret void
}

define void @zext_v4i16_to_v4i64(<4 x i16>* %a) {
; SSE2: zext_v4i16_to_v4i64
; SSE2: cost of 3 {{.*}} zext
;
; SSE41: zext_v4i16_to_v4i64
; SSE41: cost of 2 {{.*}} zext
;
  %1 = load <4 x i16>, <4 x i16>* %a
  %2 = zext <4 x i16> %1 to <4 x i64>
  store <4 x i64> %2, <4 x i64>* undef, align 4
  ret void
}

define void @sext_v4i16_to_v4i64(<4 x i16>* %a) {
; SSE2: sext_v4i16_to_v4i64
; SSE2: cost of 10 {{.*}} sext
;
; SSE41: sext_v4i16_to_v4i64
; SSE41: cost of 2 {{.*}} sext
;
  %1 = load <4 x i16>, <4 x i16>* %a
  %2 = sext <4 x i16> %1 to <4 x i64>
  store <4 x i64> %2, <4 x i64>* undef, align 4
  ret void
}


define void @zext_v4i32_to_v4i64(<4 x i32>* %a) {
; SSE2: zext_v4i32_to_v4i64
; SSE2: cost of 3 {{.*}} zext
;
; SSE41: zext_v4i32_to_v4i64
; SSE41: cost of 2 {{.*}} zext
;
  %1 = load <4 x i32>, <4 x i32>* %a
  %2 = zext <4 x i32> %1 to <4 x i64>
  store <4 x i64> %2, <4 x i64>* undef, align 4
  ret void
}

define void @sext_v4i32_to_v4i64(<4 x i32>* %a) {
; SSE2: sext_v4i32_to_v4i64
; SSE2: cost of 5 {{.*}} sext
;
; SSE41: sext_v4i32_to_v4i64
; SSE41: cost of 2 {{.*}} sext
;
  %1 = load <4 x i32>, <4 x i32>* %a
  %2 = sext <4 x i32> %1 to <4 x i64>
  store <4 x i64> %2, <4 x i64>* undef, align 4
  ret void
}

define void @zext_v16i16_to_v16i32(<16 x i16>* %a) {
; SSE2: zext_v16i16_to_v16i32
; SSE2: cost of 6 {{.*}} zext
;
; SSE41: zext_v16i16_to_v16i32
; SSE41: cost of 4 {{.*}} zext
;
  %1 = load <16 x i16>, <16 x i16>* %a
  %2 = zext <16 x i16> %1 to <16 x i32>
  store <16 x i32> %2, <16 x i32>* undef, align 4
  ret void
}

define void @sext_v16i16_to_v16i32(<16 x i16>* %a) {
; SSE2: sext_v16i16_to_v16i32
; SSE2: cost of 8 {{.*}} sext
;
; SSE41: sext_v16i16_to_v16i32
; SSE41: cost of 4 {{.*}} sext
;
  %1 = load <16 x i16>, <16 x i16>* %a
  %2 = sext <16 x i16> %1 to <16 x i32>
  store <16 x i32> %2, <16 x i32>* undef, align 4
  ret void
}

define void @zext_v8i16_to_v8i32(<8 x i16>* %a) {
; SSE2: zext_v8i16_to_v8i32
; SSE2: cost of 3 {{.*}} zext
;
; SSE41: zext_v8i16_to_v8i32
; SSE41: cost of 2 {{.*}} zext
;
  %1 = load <8 x i16>, <8 x i16>* %a
  %2 = zext <8 x i16> %1 to <8 x i32>
  store <8 x i32> %2, <8 x i32>* undef, align 4
  ret void
}

define void @sext_v8i16_to_v8i32(<8 x i16>* %a) {
; SSE2: sext_v8i16_to_v8i32
; SSE2: cost of 4 {{.*}} sext
;
; SSE41: sext_v8i16_to_v8i32
; SSE41: cost of 2 {{.*}} sext
;
  %1 = load <8 x i16>, <8 x i16>* %a
  %2 = sext <8 x i16> %1 to <8 x i32>
  store <8 x i32> %2, <8 x i32>* undef, align 4
  ret void
}

define void @zext_v4i16_to_v4i32(<4 x i16>* %a) {
; SSE2: zext_v4i16_to_v4i32
; SSE2: cost of 1 {{.*}} zext
;
; SSE41: zext_v4i16_to_v4i32
; SSE41: cost of 1 {{.*}} zext
;
  %1 = load <4 x i16>, <4 x i16>* %a
  %2 = zext <4 x i16> %1 to <4 x i32>
  store <4 x i32> %2, <4 x i32>* undef, align 4
  ret void
}

define void @sext_v4i16_to_v4i32(<4 x i16>* %a) {
; SSE2: sext_v4i16_to_v4i32
; SSE2: cost of 2 {{.*}} sext
;
; SSE41: sext_v4i16_to_v4i32
; SSE41: cost of 1 {{.*}} sext
;
  %1 = load <4 x i16>, <4 x i16>* %a
  %2 = sext <4 x i16> %1 to <4 x i32>
  store <4 x i32> %2, <4 x i32>* undef, align 4
  ret void
}

define void @zext_v16i8_to_v16i32(<16 x i8>* %a) {
; SSE2: zext_v16i8_to_v16i32
; SSE2: cost of 9 {{.*}} zext
;
; SSE41: zext_v16i8_to_v16i32
; SSE41: cost of 4 {{.*}} zext
;
  %1 = load <16 x i8>, <16 x i8>* %a
  %2 = zext <16 x i8> %1 to <16 x i32>
  store <16 x i32> %2, <16 x i32>* undef, align 4
  ret void
}

define void @sext_v16i8_to_v16i32(<16 x i8>* %a) {
; SSE2: sext_v16i8_to_v16i32
; SSE2: cost of 12 {{.*}} sext
;
; SSE41: sext_v16i8_to_v16i32
; SSE41: cost of 4 {{.*}} sext
;
  %1 = load <16 x i8>, <16 x i8>* %a
  %2 = sext <16 x i8> %1 to <16 x i32>
  store <16 x i32> %2, <16 x i32>* undef, align 4
  ret void
}

define void @zext_v8i8_to_v8i32(<8 x i8>* %a) {
; SSE2: zext_v8i8_to_v8i32
; SSE2: cost of 6 {{.*}} zext
;
; SSE41: zext_v8i8_to_v8i32
; SSE41: cost of 2 {{.*}} zext
;
  %1 = load <8 x i8>, <8 x i8>* %a
  %2 = zext <8 x i8> %1 to <8 x i32>
  store <8 x i32> %2, <8 x i32>* undef, align 4
  ret void
}

define void @sext_v8i8_to_v8i32(<8 x i8>* %a) {
; SSE2: sext_v8i8_to_v8i32
; SSE2: cost of 6 {{.*}} sext
;
; SSE41: sext_v8i8_to_v8i32
; SSE41: cost of 2 {{.*}} sext
;
  %1 = load <8 x i8>, <8 x i8>* %a
  %2 = sext <8 x i8> %1 to <8 x i32>
  store <8 x i32> %2, <8 x i32>* undef, align 4
  ret void
}

define void @zext_v4i8_to_v4i32(<4 x i8>* %a) {
; SSE2: zext_v4i8_to_v4i32
; SSE2: cost of 2 {{.*}} zext
;
; SSE41: zext_v4i8_to_v4i32
; SSE41: cost of 1 {{.*}} zext
;
  %1 = load <4 x i8>, <4 x i8>* %a
  %2 = zext <4 x i8> %1 to <4 x i32>
  store <4 x i32> %2, <4 x i32>* undef, align 4
  ret void
}

define void @sext_v4i8_to_v4i32(<4 x i8>* %a) {
; SSE2: sext_v4i8_to_v4i32
; SSE2: cost of 3 {{.*}} sext
;
; SSE41: sext_v4i8_to_v4i32
; SSE41: cost of 1 {{.*}} sext
;
  %1 = load <4 x i8>, <4 x i8>* %a
  %2 = sext <4 x i8> %1 to <4 x i32>
  store <4 x i32> %2, <4 x i32>* undef, align 4
  ret void
}

define void @zext_v16i8_to_v16i16(<16 x i8>* %a) {
; SSE2: zext_v16i8_to_v16i16
; SSE2: cost of 3 {{.*}} zext
;
; SSE41: zext_v16i8_to_v16i16
; SSE41: cost of 2 {{.*}} zext
;
  %1 = load <16 x i8>, <16 x i8>* %a
  %2 = zext <16 x i8> %1 to <16 x i16>
  store <16 x i16> %2, <16 x i16>* undef, align 4
  ret void
}

define void @sext_v16i8_to_v16i16(<16 x i8>* %a) {
; SSE2: sext_v16i8_to_v16i16
; SSE2: cost of 4 {{.*}} sext
;
; SSE41: sext_v16i8_to_v16i16
; SSE41: cost of 2 {{.*}} sext
;
  %1 = load <16 x i8>, <16 x i8>* %a
  %2 = sext <16 x i8> %1 to <16 x i16>
  store <16 x i16> %2, <16 x i16>* undef, align 4
  ret void
}

define void @zext_v8i8_to_v8i16(<8 x i8>* %a) {
; SSE2: zext_v8i8_to_v8i16
; SSE2: cost of 1 {{.*}} zext
;
; SSE41: zext_v8i8_to_v8i16
; SSE41: cost of 1 {{.*}} zext
;
  %1 = load <8 x i8>, <8 x i8>* %a
  %2 = zext <8 x i8> %1 to <8 x i16>
  store <8 x i16> %2, <8 x i16>* undef, align 4
  ret void
}

define void @sext_v8i8_to_v8i16(<8 x i8>* %a) {
; SSE2: sext_v8i8_to_v8i16
; SSE2: cost of 2 {{.*}} sext
;
; SSE41: sext_v8i8_to_v8i16
; SSE41: cost of 1 {{.*}} sext
;
  %1 = load <8 x i8>, <8 x i8>* %a
  %2 = sext <8 x i8> %1 to <8 x i16>
  store <8 x i16> %2, <8 x i16>* undef, align 4
  ret void
}

define void @zext_v4i8_to_v4i16(<4 x i8>* %a) {
; SSE2: zext_v4i8_to_v4i16
; SSE2: cost of 1 {{.*}} zext
;
; SSE41: zext_v4i8_to_v4i16
; SSE41: cost of 1 {{.*}} zext
;
  %1 = load <4 x i8>, <4 x i8>* %a
  %2 = zext <4 x i8> %1 to <4 x i16>
  store <4 x i16> %2, <4 x i16>* undef, align 4
  ret void
}

define void @sext_v4i8_to_v4i16(<4 x i8>* %a) {
; SSE2: sext_v4i8_to_v4i16
; SSE2: cost of 6 {{.*}} sext
;
; SSE41: sext_v4i8_to_v4i16
; SSE41: cost of 2 {{.*}} sext
;
  %1 = load <4 x i8>, <4 x i8>* %a
  %2 = sext <4 x i8> %1 to <4 x i16>
  store <4 x i16> %2, <4 x i16>* undef, align 4
  ret void
}

define void @truncate_v16i32_to_v16i16(<16 x i32>* %a) {
; SSE2: truncate_v16i32_to_v16i16
; SSE2: cost of 10 {{.*}} trunc
;
; SSE41: truncate_v16i32_to_v16i16
; SSE41: cost of 6 {{.*}} trunc
;
  %1 = load <16 x i32>, <16 x i32>* %a
  %2 = trunc <16 x i32> %1 to <16 x i16>
  store <16 x i16> %2, <16 x i16>* undef, align 4
  ret void
}

define void @truncate_v8i32_to_v8i16(<8 x i32>* %a) {
; SSE2: truncate_v8i32_to_v8i16
; SSE2: cost of 5 {{.*}} trunc
;
; SSE41: truncate_v8i32_to_v8i16
; SSE41: cost of 3 {{.*}} trunc
;
  %1 = load <8 x i32>, <8 x i32>* %a
  %2 = trunc <8 x i32> %1 to <8 x i16>
  store <8 x i16> %2, <8 x i16>* undef, align 4
  ret void
}

define void @truncate_v4i32_to_v4i16(<4 x i32>* %a) {
; SSE2: truncate_v4i32_to_v4i16
; SSE2: cost of 3 {{.*}} trunc
;
; SSE41: truncate_v4i32_to_v4i16
; SSE41: cost of 1 {{.*}} trunc
;
  %1 = load <4 x i32>, <4 x i32>* %a
  %2 = trunc <4 x i32> %1 to <4 x i16>
  store <4 x i16> %2, <4 x i16>* undef, align 4
  ret void
}

define void @truncate_v16i32_to_v16i8(<16 x i32>* %a) {
; SSE2: truncate_v16i32_to_v16i8
; SSE2: cost of 7 {{.*}} trunc
;
; SSE41: truncate_v16i32_to_v16i8
; SSE41: cost of 7 {{.*}} trunc
;
  %1 = load <16 x i32>, <16 x i32>* %a
  %2 = trunc <16 x i32> %1 to <16 x i8>
  store <16 x i8> %2, <16 x i8>* undef, align 4
  ret void
}

define void @truncate_v8i32_to_v8i8(<8 x i32>* %a) {
; SSE2: truncate_v8i32_to_v8i8
; SSE2: cost of 4 {{.*}} trunc
;
; SSE41: truncate_v8i32_to_v8i8
; SSE41: cost of 3 {{.*}} trunc
;
  %1 = load <8 x i32>, <8 x i32>* %a
  %2 = trunc <8 x i32> %1 to <8 x i8>
  store <8 x i8> %2, <8 x i8>* undef, align 4
  ret void
}

define void @truncate_v4i32_to_v4i8(<4 x i32>* %a) {
; SSE2: truncate_v4i32_to_v4i8
; SSE2: cost of 3 {{.*}} trunc
;
; SSE41: truncate_v4i32_to_v4i8
; SSE41: cost of 1 {{.*}} trunc
;
  %1 = load <4 x i32>, <4 x i32>* %a
  %2 = trunc <4 x i32> %1 to <4 x i8>
  store <4 x i8> %2, <4 x i8>* undef, align 4
  ret void
}

define void @truncate_v16i16_to_v16i8(<16 x i16>* %a) {
; SSE2: truncate_v16i16_to_v16i8
; SSE2: cost of 3 {{.*}} trunc
;
; SSE41: truncate_v16i16_to_v16i8
; SSE41: cost of 3 {{.*}} trunc
;
  %1 = load <16 x i16>, <16 x i16>* %a
  %2 = trunc <16 x i16> %1 to <16 x i8>
  store <16 x i8> %2, <16 x i8>* undef, align 4
  ret void
}

define void @truncate_v8i16_to_v8i8(<8 x i16>* %a) {
; SSE2: truncate_v8i16_to_v8i8
; SSE2: cost of 2 {{.*}} trunc
;
; SSE41: truncate_v8i16_to_v8i8
; SSE41: cost of 1 {{.*}} trunc
;
  %1 = load <8 x i16>, <8 x i16>* %a
  %2 = trunc <8 x i16> %1 to <8 x i8>
  store <8 x i8> %2, <8 x i8>* undef, align 4
  ret void
}

define void @truncate_v4i16_to_v4i8(<4 x i16>* %a) {
; SSE2: truncate_v4i16_to_v4i8
; SSE2: cost of 4 {{.*}} trunc
;
; SSE41: truncate_v4i16_to_v4i8
; SSE41: cost of 2 {{.*}} trunc
;
  %1 = load <4 x i16>, <4 x i16>* %a
  %2 = trunc <4 x i16> %1 to <4 x i8>
  store <4 x i8> %2, <4 x i8>* undef, align 4
  ret void
}
