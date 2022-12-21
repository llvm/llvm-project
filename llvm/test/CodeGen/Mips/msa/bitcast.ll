; Test the bitcast operation for big-endian and little-endian.

; RUN: llc -march=mips -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck -check-prefix=BIGENDIAN %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64,+mips32r2 < %s | FileCheck -check-prefix=LITENDIAN %s

define void @v16i8_to_v16i8(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <16 x i8>, ptr %src
  %1 = tail call <16 x i8> @llvm.mips.addv.b(<16 x i8> %0, <16 x i8> %0)
  %2 = bitcast <16 x i8> %1 to <16 x i8>
  %3 = tail call <16 x i8> @llvm.mips.addv.b(<16 x i8> %2, <16 x i8> %2)
  store <16 x i8> %3, ptr %dst
  ret void
}

; LITENDIAN: v16i8_to_v16i8:
; LITENDIAN: ld.b [[R1:\$w[0-9]+]],
; LITENDIAN: addv.b [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.b [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.b [[R3]],
; LITENDIAN: .size v16i8_to_v16i8

; BIGENDIAN: v16i8_to_v16i8:
; BIGENDIAN: ld.b [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.b [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: addv.b [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; BIGENDIAN: st.b [[R3]],
; BIGENDIAN: .size v16i8_to_v16i8

define void @v16i8_to_v8i16(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <16 x i8>, ptr %src
  %1 = tail call <16 x i8> @llvm.mips.addv.b(<16 x i8> %0, <16 x i8> %0)
  %2 = bitcast <16 x i8> %1 to <8 x i16>
  %3 = tail call <8 x i16> @llvm.mips.addv.h(<8 x i16> %2, <8 x i16> %2)
  store <8 x i16> %3, ptr %dst
  ret void
}

; LITENDIAN: v16i8_to_v8i16:
; LITENDIAN: ld.b [[R1:\$w[0-9]+]],
; LITENDIAN: addv.b [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.h [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.h [[R3]],
; LITENDIAN: .size v16i8_to_v8i16

; BIGENDIAN: v16i8_to_v8i16:
; BIGENDIAN: ld.b [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.b [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.b [[R3:\$w[0-9]+]], [[R2]], 177
; BIGENDIAN: addv.h [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.h [[R4]],
; BIGENDIAN: .size v16i8_to_v8i16

; We can't prevent the (store (bitcast X), Y) DAG Combine here because there
; are no operations for v8f16 to put in the way.
define void @v16i8_to_v8f16(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <16 x i8>, ptr %src
  %1 = tail call <16 x i8> @llvm.mips.addv.b(<16 x i8> %0, <16 x i8> %0)
  %2 = bitcast <16 x i8> %1 to <8 x half>
  store <8 x half> %2, ptr %dst
  ret void
}

; LITENDIAN: v16i8_to_v8f16:
; LITENDIAN: ld.b [[R1:\$w[0-9]+]],
; LITENDIAN: addv.b [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: st.b [[R2]],
; LITENDIAN: .size v16i8_to_v8f16

; BIGENDIAN: v16i8_to_v8f16:
; BIGENDIAN: ld.b [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.b [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: st.b [[R2]],
; BIGENDIAN: .size v16i8_to_v8f16

define void @v16i8_to_v4i32(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <16 x i8>, ptr %src
  %1 = tail call <16 x i8> @llvm.mips.addv.b(<16 x i8> %0, <16 x i8> %0)
  %2 = bitcast <16 x i8> %1 to <4 x i32>
  %3 = tail call <4 x i32> @llvm.mips.addv.w(<4 x i32> %2, <4 x i32> %2)
  store <4 x i32> %3, ptr %dst
  ret void
}

; LITENDIAN: v16i8_to_v4i32:
; LITENDIAN: ld.b [[R1:\$w[0-9]+]],
; LITENDIAN: addv.b [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.w [[R3]],
; LITENDIAN: .size v16i8_to_v4i32

; BIGENDIAN: v16i8_to_v4i32:
; BIGENDIAN: ld.b [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.b [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.b [[R3:\$w[0-9]+]], [[R2]], 27
; BIGENDIAN: addv.w [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.w [[R4]],
; BIGENDIAN: .size v16i8_to_v4i32

define void @v16i8_to_v4f32(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <16 x i8>, ptr %src
  %1 = tail call <16 x i8> @llvm.mips.addv.b(<16 x i8> %0, <16 x i8> %0)
  %2 = bitcast <16 x i8> %1 to <4 x float>
  %3 = tail call <4 x float> @llvm.mips.fadd.w(<4 x float> %2, <4 x float> %2)
  store <4 x float> %3, ptr %dst
  ret void
}

; LITENDIAN: v16i8_to_v4f32:
; LITENDIAN: ld.b [[R1:\$w[0-9]+]],
; LITENDIAN: addv.b [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: fadd.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.w [[R3]],
; LITENDIAN: .size v16i8_to_v4f32

; BIGENDIAN: v16i8_to_v4f32:
; BIGENDIAN: ld.b [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.b [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.b [[R3:\$w[0-9]+]], [[R2]], 27
; BIGENDIAN: fadd.w [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.w [[R4]],
; BIGENDIAN: .size v16i8_to_v4f32

define void @v16i8_to_v2i64(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <16 x i8>, ptr %src
  %1 = tail call <16 x i8> @llvm.mips.addv.b(<16 x i8> %0, <16 x i8> %0)
  %2 = bitcast <16 x i8> %1 to <2 x i64>
  %3 = tail call <2 x i64> @llvm.mips.addv.d(<2 x i64> %2, <2 x i64> %2)
  store <2 x i64> %3, ptr %dst
  ret void
}

; LITENDIAN: v16i8_to_v2i64:
; LITENDIAN: ld.b [[R1:\$w[0-9]+]],
; LITENDIAN: addv.b [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.d [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.d [[R3]],
; LITENDIAN: .size v16i8_to_v2i64

; BIGENDIAN: v16i8_to_v2i64:
; BIGENDIAN: ld.b [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.b [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.b [[R3:\$w[0-9]+]], [[R2]], 27
; BIGENDIAN: shf.w [[R3:\$w[0-9]+]], [[R3]], 177
; BIGENDIAN: addv.d [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.d [[R4]],
; BIGENDIAN: .size v16i8_to_v2i64

define void @v16i8_to_v2f64(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <16 x i8>, ptr %src
  %1 = tail call <16 x i8> @llvm.mips.addv.b(<16 x i8> %0, <16 x i8> %0)
  %2 = bitcast <16 x i8> %1 to <2 x double>
  %3 = tail call <2 x double> @llvm.mips.fadd.d(<2 x double> %2, <2 x double> %2)
  store <2 x double> %3, ptr %dst
  ret void
}

; LITENDIAN: v16i8_to_v2f64:
; LITENDIAN: ld.b [[R1:\$w[0-9]+]],
; LITENDIAN: addv.b [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: fadd.d [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.d [[R3]],
; LITENDIAN: .size v16i8_to_v2f64

; BIGENDIAN: v16i8_to_v2f64:
; BIGENDIAN: ld.b [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.b [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.b [[R3:\$w[0-9]+]], [[R2]], 27
; BIGENDIAN: shf.w [[R3:\$w[0-9]+]], [[R3]], 177
; BIGENDIAN: fadd.d [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.d [[R4]],
; BIGENDIAN: .size v16i8_to_v2f64

define void @v8i16_to_v16i8(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <8 x i16>, ptr %src
  %1 = tail call <8 x i16> @llvm.mips.addv.h(<8 x i16> %0, <8 x i16> %0)
  %2 = bitcast <8 x i16> %1 to <16 x i8>
  %3 = tail call <16 x i8> @llvm.mips.addv.b(<16 x i8> %2, <16 x i8> %2)
  store <16 x i8> %3, ptr %dst
  ret void
}

; LITENDIAN: v8i16_to_v16i8:
; LITENDIAN: ld.h [[R1:\$w[0-9]+]],
; LITENDIAN: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.b [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.b [[R3]],
; LITENDIAN: .size v8i16_to_v16i8

; BIGENDIAN: v8i16_to_v16i8:
; BIGENDIAN: ld.h [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.b [[R3:\$w[0-9]+]], [[R2]], 177
; BIGENDIAN: addv.b [[R4:\$w[0-9]+]], [[R2]], [[R2]]
; BIGENDIAN: st.b [[R4]],
; BIGENDIAN: .size v8i16_to_v16i8

define void @v8i16_to_v8i16(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <8 x i16>, ptr %src
  %1 = tail call <8 x i16> @llvm.mips.addv.h(<8 x i16> %0, <8 x i16> %0)
  %2 = bitcast <8 x i16> %1 to <8 x i16>
  %3 = tail call <8 x i16> @llvm.mips.addv.h(<8 x i16> %2, <8 x i16> %2)
  store <8 x i16> %3, ptr %dst
  ret void
}

; LITENDIAN: v8i16_to_v8i16:
; LITENDIAN: ld.h [[R1:\$w[0-9]+]],
; LITENDIAN: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.h [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.h [[R3]],
; LITENDIAN: .size v8i16_to_v8i16

; BIGENDIAN: v8i16_to_v8i16:
; BIGENDIAN: ld.h [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: addv.h [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; BIGENDIAN: st.h [[R3]],
; BIGENDIAN: .size v8i16_to_v8i16

; We can't prevent the (store (bitcast X), Y) DAG Combine here because there
; are no operations for v8f16 to put in the way.
define void @v8i16_to_v8f16(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <8 x i16>, ptr %src
  %1 = tail call <8 x i16> @llvm.mips.addv.h(<8 x i16> %0, <8 x i16> %0)
  %2 = bitcast <8 x i16> %1 to <8 x half>
  store <8 x half> %2, ptr %dst
  ret void
}

; LITENDIAN: v8i16_to_v8f16:
; LITENDIAN: ld.h [[R1:\$w[0-9]+]],
; LITENDIAN: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: st.h [[R2]],
; LITENDIAN: .size v8i16_to_v8f16

; BIGENDIAN: v8i16_to_v8f16:
; BIGENDIAN: ld.h [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: st.h [[R2]],
; BIGENDIAN: .size v8i16_to_v8f16

define void @v8i16_to_v4i32(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <8 x i16>, ptr %src
  %1 = tail call <8 x i16> @llvm.mips.addv.h(<8 x i16> %0, <8 x i16> %0)
  %2 = bitcast <8 x i16> %1 to <4 x i32>
  %3 = tail call <4 x i32> @llvm.mips.addv.w(<4 x i32> %2, <4 x i32> %2)
  store <4 x i32> %3, ptr %dst
  ret void
}

; LITENDIAN: v8i16_to_v4i32:
; LITENDIAN: ld.h [[R1:\$w[0-9]+]],
; LITENDIAN: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.w [[R3]],
; LITENDIAN: .size v8i16_to_v4i32

; BIGENDIAN: v8i16_to_v4i32:
; BIGENDIAN: ld.h [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.h [[R3:\$w[0-9]+]], [[R2]], 177
; BIGENDIAN: addv.w [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.w [[R4]],
; BIGENDIAN: .size v8i16_to_v4i32

define void @v8i16_to_v4f32(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <8 x i16>, ptr %src
  %1 = tail call <8 x i16> @llvm.mips.addv.h(<8 x i16> %0, <8 x i16> %0)
  %2 = bitcast <8 x i16> %1 to <4 x float>
  %3 = tail call <4 x float> @llvm.mips.fadd.w(<4 x float> %2, <4 x float> %2)
  store <4 x float> %3, ptr %dst
  ret void
}

; LITENDIAN: v8i16_to_v4f32:
; LITENDIAN: ld.h [[R1:\$w[0-9]+]],
; LITENDIAN: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: fadd.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.w [[R3]],
; LITENDIAN: .size v8i16_to_v4f32

; BIGENDIAN: v8i16_to_v4f32:
; BIGENDIAN: ld.h [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.h [[R3:\$w[0-9]+]], [[R2]], 177
; BIGENDIAN: fadd.w [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.w [[R4]],
; BIGENDIAN: .size v8i16_to_v4f32

define void @v8i16_to_v2i64(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <8 x i16>, ptr %src
  %1 = tail call <8 x i16> @llvm.mips.addv.h(<8 x i16> %0, <8 x i16> %0)
  %2 = bitcast <8 x i16> %1 to <2 x i64>
  %3 = tail call <2 x i64> @llvm.mips.addv.d(<2 x i64> %2, <2 x i64> %2)
  store <2 x i64> %3, ptr %dst
  ret void
}

; LITENDIAN: v8i16_to_v2i64:
; LITENDIAN: ld.h [[R1:\$w[0-9]+]],
; LITENDIAN: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.d [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.d [[R3]],
; LITENDIAN: .size v8i16_to_v2i64

; BIGENDIAN: v8i16_to_v2i64:
; BIGENDIAN: ld.h [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.h [[R3:\$w[0-9]+]], [[R2]], 27
; BIGENDIAN: addv.d [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.d [[R4]],
; BIGENDIAN: .size v8i16_to_v2i64

define void @v8i16_to_v2f64(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <8 x i16>, ptr %src
  %1 = tail call <8 x i16> @llvm.mips.addv.h(<8 x i16> %0, <8 x i16> %0)
  %2 = bitcast <8 x i16> %1 to <2 x double>
  %3 = tail call <2 x double> @llvm.mips.fadd.d(<2 x double> %2, <2 x double> %2)
  store <2 x double> %3, ptr %dst
  ret void
}

; LITENDIAN: v8i16_to_v2f64:
; LITENDIAN: ld.h [[R1:\$w[0-9]+]],
; LITENDIAN: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: fadd.d [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.d [[R3]],
; LITENDIAN: .size v8i16_to_v2f64

; BIGENDIAN: v8i16_to_v2f64:
; BIGENDIAN: ld.h [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.h [[R3:\$w[0-9]+]], [[R2]], 27
; BIGENDIAN: fadd.d [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.d [[R4]],
; BIGENDIAN: .size v8i16_to_v2f64

;----
; We can't prevent the (bitcast (load X)) DAG Combine here because there
; are no operations for v8f16 to put in the way.
define void @v8f16_to_v16i8(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <8 x half>, ptr %src
  %1 = bitcast <8 x half> %0 to <16 x i8>
  %2 = tail call <16 x i8> @llvm.mips.addv.b(<16 x i8> %1, <16 x i8> %1)
  store <16 x i8> %2, ptr %dst
  ret void
}

; LITENDIAN: v8f16_to_v16i8:
; LITENDIAN: ld.b [[R1:\$w[0-9]+]],
; LITENDIAN: addv.b [[R3:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: st.b [[R3]],
; LITENDIAN: .size v8f16_to_v16i8

; BIGENDIAN: v8f16_to_v16i8:
; BIGENDIAN: ld.b [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.b [[R4:\$w[0-9]+]], [[R2]], [[R2]]
; BIGENDIAN: st.b [[R4]],
; BIGENDIAN: .size v8f16_to_v16i8

; We can't prevent the (bitcast (load X)) DAG Combine here because there
; are no operations for v8f16 to put in the way.
define void @v8f16_to_v8i16(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <8 x half>, ptr %src
  %1 = bitcast <8 x half> %0 to <8 x i16>
  %2 = tail call <8 x i16> @llvm.mips.addv.h(<8 x i16> %1, <8 x i16> %1)
  store <8 x i16> %2, ptr %dst
  ret void
}

; LITENDIAN: v8f16_to_v8i16:
; LITENDIAN: ld.h [[R1:\$w[0-9]+]],
; LITENDIAN: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: st.h [[R2]],
; LITENDIAN: .size v8f16_to_v8i16

; BIGENDIAN: v8f16_to_v8i16:
; BIGENDIAN: ld.h [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: st.h [[R2]],
; BIGENDIAN: .size v8f16_to_v8i16

; We can't prevent the (bitcast (load X)) DAG Combine here because there
; are no operations for v8f16 to put in the way.
; We can't prevent the (store (bitcast X), Y) DAG Combine here because there
; are no operations for v8f16 to put in the way.
define void @v8f16_to_v8f16(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <8 x half>, ptr %src
  %1 = bitcast <8 x half> %0 to <8 x half>
  store <8 x half> %1, ptr %dst
  ret void
}

; LITENDIAN: v8f16_to_v8f16:
; LITENDIAN: ld.h [[R1:\$w[0-9]+]],
; LITENDIAN: st.h [[R1]],
; LITENDIAN: .size v8f16_to_v8f16

; BIGENDIAN: v8f16_to_v8f16:
; BIGENDIAN: ld.h [[R1:\$w[0-9]+]],
; BIGENDIAN: st.h [[R1]],
; BIGENDIAN: .size v8f16_to_v8f16

; We can't prevent the (bitcast (load X)) DAG Combine here because there
; are no operations for v8f16 to put in the way.
define void @v8f16_to_v4i32(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <8 x half>, ptr %src
  %1 = bitcast <8 x half> %0 to <4 x i32>
  %2 = tail call <4 x i32> @llvm.mips.addv.w(<4 x i32> %1, <4 x i32> %1)
  store <4 x i32> %2, ptr %dst
  ret void
}

; LITENDIAN: v8f16_to_v4i32:
; LITENDIAN: ld.w [[R1:\$w[0-9]+]],
; LITENDIAN: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: st.w [[R2]],
; LITENDIAN: .size v8f16_to_v4i32

; BIGENDIAN: v8f16_to_v4i32:
; BIGENDIAN: ld.w [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; BIGENDIAN: st.w [[R3]],
; BIGENDIAN: .size v8f16_to_v4i32

; We can't prevent the (bitcast (load X)) DAG Combine here because there
; are no operations for v8f16 to put in the way.
define void @v8f16_to_v4f32(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <8 x half>, ptr %src
  %1 = bitcast <8 x half> %0 to <4 x float>
  %2 = tail call <4 x float> @llvm.mips.fadd.w(<4 x float> %1, <4 x float> %1)
  store <4 x float> %2, ptr %dst
  ret void
}

; LITENDIAN: v8f16_to_v4f32:
; LITENDIAN: ld.w [[R1:\$w[0-9]+]],
; LITENDIAN: fadd.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: st.w [[R2]],
; LITENDIAN: .size v8f16_to_v4f32

; BIGENDIAN: v8f16_to_v4f32:
; BIGENDIAN: ld.w [[R1:\$w[0-9]+]],
; BIGENDIAN: fadd.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; BIGENDIAN: st.w [[R3]],
; BIGENDIAN: .size v8f16_to_v4f32

; We can't prevent the (bitcast (load X)) DAG Combine here because there
; are no operations for v8f16 to put in the way.
define void @v8f16_to_v2i64(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <8 x half>, ptr %src
  %1 = bitcast <8 x half> %0 to <2 x i64>
  %2 = tail call <2 x i64> @llvm.mips.addv.d(<2 x i64> %1, <2 x i64> %1)
  store <2 x i64> %2, ptr %dst
  ret void
}

; LITENDIAN: v8f16_to_v2i64:
; LITENDIAN: ld.d [[R1:\$w[0-9]+]],
; LITENDIAN: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: st.d [[R2]],
; LITENDIAN: .size v8f16_to_v2i64

; BIGENDIAN: v8f16_to_v2i64:
; BIGENDIAN: ld.d [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.d [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; BIGENDIAN: st.d [[R3]],
; BIGENDIAN: .size v8f16_to_v2i64

; We can't prevent the (bitcast (load X)) DAG Combine here because there
; are no operations for v8f16 to put in the way.
define void @v8f16_to_v2f64(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <8 x half>, ptr %src
  %1 = bitcast <8 x half> %0 to <2 x double>
  %2 = tail call <2 x double> @llvm.mips.fadd.d(<2 x double> %1, <2 x double> %1)
  store <2 x double> %2, ptr %dst
  ret void
}

; LITENDIAN: v8f16_to_v2f64:
; LITENDIAN: ld.d [[R1:\$w[0-9]+]],
; LITENDIAN: fadd.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: st.d [[R2]],
; LITENDIAN: .size v8f16_to_v2f64

; BIGENDIAN: v8f16_to_v2f64:
; BIGENDIAN: ld.d [[R1:\$w[0-9]+]],
; BIGENDIAN: fadd.d [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; BIGENDIAN: st.d [[R3]],
; BIGENDIAN: .size v8f16_to_v2f64
;----

define void @v4i32_to_v16i8(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <4 x i32>, ptr %src
  %1 = tail call <4 x i32> @llvm.mips.addv.w(<4 x i32> %0, <4 x i32> %0)
  %2 = bitcast <4 x i32> %1 to <16 x i8>
  %3 = tail call <16 x i8> @llvm.mips.addv.b(<16 x i8> %2, <16 x i8> %2)
  store <16 x i8> %3, ptr %dst
  ret void
}

; LITENDIAN: v4i32_to_v16i8:
; LITENDIAN: ld.w [[R1:\$w[0-9]+]],
; LITENDIAN: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.b [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.b [[R3]],
; LITENDIAN: .size v4i32_to_v16i8

; BIGENDIAN: v4i32_to_v16i8:
; BIGENDIAN: ld.w [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.b [[R3:\$w[0-9]+]], [[R2]], 27
; BIGENDIAN: addv.b [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.b [[R4]],
; BIGENDIAN: .size v4i32_to_v16i8

define void @v4i32_to_v8i16(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <4 x i32>, ptr %src
  %1 = tail call <4 x i32> @llvm.mips.addv.w(<4 x i32> %0, <4 x i32> %0)
  %2 = bitcast <4 x i32> %1 to <8 x i16>
  %3 = tail call <8 x i16> @llvm.mips.addv.h(<8 x i16> %2, <8 x i16> %2)
  store <8 x i16> %3, ptr %dst
  ret void
}

; LITENDIAN: v4i32_to_v8i16:
; LITENDIAN: ld.w [[R1:\$w[0-9]+]],
; LITENDIAN: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.h [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.h [[R3]],
; LITENDIAN: .size v4i32_to_v8i16

; BIGENDIAN: v4i32_to_v8i16:
; BIGENDIAN: ld.w [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.h [[R3:\$w[0-9]+]], [[R2]], 177
; BIGENDIAN: addv.h [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.h [[R4]],
; BIGENDIAN: .size v4i32_to_v8i16

; We can't prevent the (store (bitcast X), Y) DAG Combine here because there
; are no operations for v8f16 to put in the way.
define void @v4i32_to_v8f16(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <4 x i32>, ptr %src
  %1 = tail call <4 x i32> @llvm.mips.addv.w(<4 x i32> %0, <4 x i32> %0)
  %2 = bitcast <4 x i32> %1 to <8 x half>
  store <8 x half> %2, ptr %dst
  ret void
}

; LITENDIAN: v4i32_to_v8f16:
; LITENDIAN: ld.w [[R1:\$w[0-9]+]],
; LITENDIAN: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: st.w [[R2]],
; LITENDIAN: .size v4i32_to_v8f16

; BIGENDIAN: v4i32_to_v8f16:
; BIGENDIAN: ld.w [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: st.w [[R2]],
; BIGENDIAN: .size v4i32_to_v8f16

define void @v4i32_to_v4i32(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <4 x i32>, ptr %src
  %1 = tail call <4 x i32> @llvm.mips.addv.w(<4 x i32> %0, <4 x i32> %0)
  %2 = bitcast <4 x i32> %1 to <4 x i32>
  %3 = tail call <4 x i32> @llvm.mips.addv.w(<4 x i32> %2, <4 x i32> %2)
  store <4 x i32> %3, ptr %dst
  ret void
}

; LITENDIAN: v4i32_to_v4i32:
; LITENDIAN: ld.w [[R1:\$w[0-9]+]],
; LITENDIAN: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.w [[R3]],
; LITENDIAN: .size v4i32_to_v4i32

; BIGENDIAN: v4i32_to_v4i32:
; BIGENDIAN: ld.w [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: addv.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; BIGENDIAN: st.w [[R3]],
; BIGENDIAN: .size v4i32_to_v4i32

define void @v4i32_to_v4f32(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <4 x i32>, ptr %src
  %1 = tail call <4 x i32> @llvm.mips.addv.w(<4 x i32> %0, <4 x i32> %0)
  %2 = bitcast <4 x i32> %1 to <4 x float>
  %3 = tail call <4 x float> @llvm.mips.fadd.w(<4 x float> %2, <4 x float> %2)
  store <4 x float> %3, ptr %dst
  ret void
}

; LITENDIAN: v4i32_to_v4f32:
; LITENDIAN: ld.w [[R1:\$w[0-9]+]],
; LITENDIAN: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: fadd.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.w [[R3]],
; LITENDIAN: .size v4i32_to_v4f32

; BIGENDIAN: v4i32_to_v4f32:
; BIGENDIAN: ld.w [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: fadd.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; BIGENDIAN: st.w [[R3]],
; BIGENDIAN: .size v4i32_to_v4f32

define void @v4i32_to_v2i64(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <4 x i32>, ptr %src
  %1 = tail call <4 x i32> @llvm.mips.addv.w(<4 x i32> %0, <4 x i32> %0)
  %2 = bitcast <4 x i32> %1 to <2 x i64>
  %3 = tail call <2 x i64> @llvm.mips.addv.d(<2 x i64> %2, <2 x i64> %2)
  store <2 x i64> %3, ptr %dst
  ret void
}

; LITENDIAN: v4i32_to_v2i64:
; LITENDIAN: ld.w [[R1:\$w[0-9]+]],
; LITENDIAN: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.d [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.d [[R3]],
; LITENDIAN: .size v4i32_to_v2i64

; BIGENDIAN: v4i32_to_v2i64:
; BIGENDIAN: ld.w [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.w [[R3:\$w[0-9]+]], [[R2]], 177
; BIGENDIAN: addv.d [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.d [[R4]],
; BIGENDIAN: .size v4i32_to_v2i64

define void @v4i32_to_v2f64(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <4 x i32>, ptr %src
  %1 = tail call <4 x i32> @llvm.mips.addv.w(<4 x i32> %0, <4 x i32> %0)
  %2 = bitcast <4 x i32> %1 to <2 x double>
  %3 = tail call <2 x double> @llvm.mips.fadd.d(<2 x double> %2, <2 x double> %2)
  store <2 x double> %3, ptr %dst
  ret void
}

; LITENDIAN: v4i32_to_v2f64:
; LITENDIAN: ld.w [[R1:\$w[0-9]+]],
; LITENDIAN: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: fadd.d [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.d [[R3]],
; LITENDIAN: .size v4i32_to_v2f64

; BIGENDIAN: v4i32_to_v2f64:
; BIGENDIAN: ld.w [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.w [[R3:\$w[0-9]+]], [[R2]], 177
; BIGENDIAN: fadd.d [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.d [[R4]],
; BIGENDIAN: .size v4i32_to_v2f64

define void @v4f32_to_v16i8(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <4 x float>, ptr %src
  %1 = tail call <4 x float> @llvm.mips.fadd.w(<4 x float> %0, <4 x float> %0)
  %2 = bitcast <4 x float> %1 to <16 x i8>
  %3 = tail call <16 x i8> @llvm.mips.addv.b(<16 x i8> %2, <16 x i8> %2)
  store <16 x i8> %3, ptr %dst
  ret void
}

; LITENDIAN: v4f32_to_v16i8:
; LITENDIAN: ld.w [[R1:\$w[0-9]+]],
; LITENDIAN: fadd.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.b [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.b [[R3]],
; LITENDIAN: .size v4f32_to_v16i8

; BIGENDIAN: v4f32_to_v16i8:
; BIGENDIAN: ld.w [[R1:\$w[0-9]+]],
; BIGENDIAN: fadd.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.b [[R3:\$w[0-9]+]], [[R2]], 27
; BIGENDIAN: addv.b [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.b [[R4]],
; BIGENDIAN: .size v4f32_to_v16i8

define void @v4f32_to_v8i16(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <4 x float>, ptr %src
  %1 = tail call <4 x float> @llvm.mips.fadd.w(<4 x float> %0, <4 x float> %0)
  %2 = bitcast <4 x float> %1 to <8 x i16>
  %3 = tail call <8 x i16> @llvm.mips.addv.h(<8 x i16> %2, <8 x i16> %2)
  store <8 x i16> %3, ptr %dst
  ret void
}

; LITENDIAN: v4f32_to_v8i16:
; LITENDIAN: ld.w [[R1:\$w[0-9]+]],
; LITENDIAN: fadd.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.h [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.h [[R3]],
; LITENDIAN: .size v4f32_to_v8i16

; BIGENDIAN: v4f32_to_v8i16:
; BIGENDIAN: ld.w [[R1:\$w[0-9]+]],
; BIGENDIAN: fadd.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.h [[R3:\$w[0-9]+]], [[R2]], 177
; BIGENDIAN: addv.h [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.h [[R4]],
; BIGENDIAN: .size v4f32_to_v8i16

; We can't prevent the (store (bitcast X), Y) DAG Combine here because there
; are no operations for v8f16 to put in the way.
define void @v4f32_to_v8f16(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <4 x float>, ptr %src
  %1 = tail call <4 x float> @llvm.mips.fadd.w(<4 x float> %0, <4 x float> %0)
  %2 = bitcast <4 x float> %1 to <8 x half>
  store <8 x half> %2, ptr %dst
  ret void
}

; LITENDIAN: v4f32_to_v8f16:
; LITENDIAN: ld.w [[R1:\$w[0-9]+]],
; LITENDIAN: fadd.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: st.w [[R2]],
; LITENDIAN: .size v4f32_to_v8f16

; BIGENDIAN: v4f32_to_v8f16:
; BIGENDIAN: ld.w [[R1:\$w[0-9]+]],
; BIGENDIAN: fadd.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: st.w [[R2]],
; BIGENDIAN: .size v4f32_to_v8f16

define void @v4f32_to_v4i32(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <4 x float>, ptr %src
  %1 = tail call <4 x float> @llvm.mips.fadd.w(<4 x float> %0, <4 x float> %0)
  %2 = bitcast <4 x float> %1 to <4 x i32>
  %3 = tail call <4 x i32> @llvm.mips.addv.w(<4 x i32> %2, <4 x i32> %2)
  store <4 x i32> %3, ptr %dst
  ret void
}

; LITENDIAN: v4f32_to_v4i32:
; LITENDIAN: ld.w [[R1:\$w[0-9]+]],
; LITENDIAN: fadd.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.w [[R3]],
; LITENDIAN: .size v4f32_to_v4i32

; BIGENDIAN: v4f32_to_v4i32:
; BIGENDIAN: ld.w [[R1:\$w[0-9]+]],
; BIGENDIAN: fadd.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: addv.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; BIGENDIAN: st.w [[R3]],
; BIGENDIAN: .size v4f32_to_v4i32

define void @v4f32_to_v4f32(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <4 x float>, ptr %src
  %1 = tail call <4 x float> @llvm.mips.fadd.w(<4 x float> %0, <4 x float> %0)
  %2 = bitcast <4 x float> %1 to <4 x float>
  %3 = tail call <4 x float> @llvm.mips.fadd.w(<4 x float> %2, <4 x float> %2)
  store <4 x float> %3, ptr %dst
  ret void
}

; LITENDIAN: v4f32_to_v4f32:
; LITENDIAN: ld.w [[R1:\$w[0-9]+]],
; LITENDIAN: fadd.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: fadd.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.w [[R3]],
; LITENDIAN: .size v4f32_to_v4f32

; BIGENDIAN: v4f32_to_v4f32:
; BIGENDIAN: ld.w [[R1:\$w[0-9]+]],
; BIGENDIAN: fadd.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: fadd.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; BIGENDIAN: st.w [[R3]],
; BIGENDIAN: .size v4f32_to_v4f32

define void @v4f32_to_v2i64(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <4 x float>, ptr %src
  %1 = tail call <4 x float> @llvm.mips.fadd.w(<4 x float> %0, <4 x float> %0)
  %2 = bitcast <4 x float> %1 to <2 x i64>
  %3 = tail call <2 x i64> @llvm.mips.addv.d(<2 x i64> %2, <2 x i64> %2)
  store <2 x i64> %3, ptr %dst
  ret void
}

; LITENDIAN: v4f32_to_v2i64:
; LITENDIAN: ld.w [[R1:\$w[0-9]+]],
; LITENDIAN: fadd.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.d [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.d [[R3]],
; LITENDIAN: .size v4f32_to_v2i64

; BIGENDIAN: v4f32_to_v2i64:
; BIGENDIAN: ld.w [[R1:\$w[0-9]+]],
; BIGENDIAN: fadd.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.w [[R3:\$w[0-9]+]], [[R2]], 177
; BIGENDIAN: addv.d [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.d [[R4]],
; BIGENDIAN: .size v4f32_to_v2i64

define void @v4f32_to_v2f64(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <4 x float>, ptr %src
  %1 = tail call <4 x float> @llvm.mips.fadd.w(<4 x float> %0, <4 x float> %0)
  %2 = bitcast <4 x float> %1 to <2 x double>
  %3 = tail call <2 x double> @llvm.mips.fadd.d(<2 x double> %2, <2 x double> %2)
  store <2 x double> %3, ptr %dst
  ret void
}

; LITENDIAN: v4f32_to_v2f64:
; LITENDIAN: ld.w [[R1:\$w[0-9]+]],
; LITENDIAN: fadd.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: fadd.d [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.d [[R3]],
; LITENDIAN: .size v4f32_to_v2f64

; BIGENDIAN: v4f32_to_v2f64:
; BIGENDIAN: ld.w [[R1:\$w[0-9]+]],
; BIGENDIAN: fadd.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.w [[R3:\$w[0-9]+]], [[R2]], 177
; BIGENDIAN: fadd.d [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.d [[R4]],
; BIGENDIAN: .size v4f32_to_v2f64

define void @v2i64_to_v16i8(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <2 x i64>, ptr %src
  %1 = tail call <2 x i64> @llvm.mips.addv.d(<2 x i64> %0, <2 x i64> %0)
  %2 = bitcast <2 x i64> %1 to <16 x i8>
  %3 = tail call <16 x i8> @llvm.mips.addv.b(<16 x i8> %2, <16 x i8> %2)
  store <16 x i8> %3, ptr %dst
  ret void
}

; LITENDIAN: v2i64_to_v16i8:
; LITENDIAN: ld.d [[R1:\$w[0-9]+]],
; LITENDIAN: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.b [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.b [[R3]],
; LITENDIAN: .size v2i64_to_v16i8

; BIGENDIAN: v2i64_to_v16i8:
; BIGENDIAN: ld.d [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.b [[R3:\$w[0-9]+]], [[R2]], 27
; BIGENDIAN: shf.w [[R3:\$w[0-9]+]], [[R3]], 177
; BIGENDIAN: addv.b [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.b [[R4]],
; BIGENDIAN: .size v2i64_to_v16i8

define void @v2i64_to_v8i16(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <2 x i64>, ptr %src
  %1 = tail call <2 x i64> @llvm.mips.addv.d(<2 x i64> %0, <2 x i64> %0)
  %2 = bitcast <2 x i64> %1 to <8 x i16>
  %3 = tail call <8 x i16> @llvm.mips.addv.h(<8 x i16> %2, <8 x i16> %2)
  store <8 x i16> %3, ptr %dst
  ret void
}

; LITENDIAN: v2i64_to_v8i16:
; LITENDIAN: ld.d [[R1:\$w[0-9]+]],
; LITENDIAN: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.h [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.h [[R3]],
; LITENDIAN: .size v2i64_to_v8i16

; BIGENDIAN: v2i64_to_v8i16:
; BIGENDIAN: ld.d [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.h [[R3:\$w[0-9]+]], [[R2]], 27
; BIGENDIAN: addv.h [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.h [[R4]],
; BIGENDIAN: .size v2i64_to_v8i16

; We can't prevent the (store (bitcast X), Y) DAG Combine here because there
; are no operations for v8f16 to put in the way.
define void @v2i64_to_v8f16(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <2 x i64>, ptr %src
  %1 = tail call <2 x i64> @llvm.mips.addv.d(<2 x i64> %0, <2 x i64> %0)
  %2 = bitcast <2 x i64> %1 to <8 x half>
  store <8 x half> %2, ptr %dst
  ret void
}

; LITENDIAN: v2i64_to_v8f16:
; LITENDIAN: ld.d [[R1:\$w[0-9]+]],
; LITENDIAN: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: st.d [[R2]],
; LITENDIAN: .size v2i64_to_v8f16

; BIGENDIAN: v2i64_to_v8f16:
; BIGENDIAN: ld.d [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: st.d [[R2]],
; BIGENDIAN: .size v2i64_to_v8f16

define void @v2i64_to_v4i32(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <2 x i64>, ptr %src
  %1 = tail call <2 x i64> @llvm.mips.addv.d(<2 x i64> %0, <2 x i64> %0)
  %2 = bitcast <2 x i64> %1 to <4 x i32>
  %3 = tail call <4 x i32> @llvm.mips.addv.w(<4 x i32> %2, <4 x i32> %2)
  store <4 x i32> %3, ptr %dst
  ret void
}

; LITENDIAN: v2i64_to_v4i32:
; LITENDIAN: ld.d [[R1:\$w[0-9]+]],
; LITENDIAN: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.w [[R3]],
; LITENDIAN: .size v2i64_to_v4i32

; BIGENDIAN: v2i64_to_v4i32:
; BIGENDIAN: ld.d [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.w [[R3:\$w[0-9]+]], [[R2]], 177
; BIGENDIAN: addv.w [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.w [[R4]],
; BIGENDIAN: .size v2i64_to_v4i32

define void @v2i64_to_v4f32(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <2 x i64>, ptr %src
  %1 = tail call <2 x i64> @llvm.mips.addv.d(<2 x i64> %0, <2 x i64> %0)
  %2 = bitcast <2 x i64> %1 to <4 x float>
  %3 = tail call <4 x float> @llvm.mips.fadd.w(<4 x float> %2, <4 x float> %2)
  store <4 x float> %3, ptr %dst
  ret void
}

; LITENDIAN: v2i64_to_v4f32:
; LITENDIAN: ld.d [[R1:\$w[0-9]+]],
; LITENDIAN: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: fadd.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.w [[R3]],
; LITENDIAN: .size v2i64_to_v4f32

; BIGENDIAN: v2i64_to_v4f32:
; BIGENDIAN: ld.d [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.w [[R3:\$w[0-9]+]], [[R2]], 177
; BIGENDIAN: fadd.w [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.w [[R4]],
; BIGENDIAN: .size v2i64_to_v4f32

define void @v2i64_to_v2i64(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <2 x i64>, ptr %src
  %1 = tail call <2 x i64> @llvm.mips.addv.d(<2 x i64> %0, <2 x i64> %0)
  %2 = bitcast <2 x i64> %1 to <2 x i64>
  %3 = tail call <2 x i64> @llvm.mips.addv.d(<2 x i64> %2, <2 x i64> %2)
  store <2 x i64> %3, ptr %dst
  ret void
}

; LITENDIAN: v2i64_to_v2i64:
; LITENDIAN: ld.d [[R1:\$w[0-9]+]],
; LITENDIAN: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.d [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.d [[R3]],
; LITENDIAN: .size v2i64_to_v2i64

; BIGENDIAN: v2i64_to_v2i64:
; BIGENDIAN: ld.d [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: addv.d [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; BIGENDIAN: st.d [[R3]],
; BIGENDIAN: .size v2i64_to_v2i64

define void @v2i64_to_v2f64(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <2 x i64>, ptr %src
  %1 = tail call <2 x i64> @llvm.mips.addv.d(<2 x i64> %0, <2 x i64> %0)
  %2 = bitcast <2 x i64> %1 to <2 x double>
  %3 = tail call <2 x double> @llvm.mips.fadd.d(<2 x double> %2, <2 x double> %2)
  store <2 x double> %3, ptr %dst
  ret void
}

; LITENDIAN: v2i64_to_v2f64:
; LITENDIAN: ld.d [[R1:\$w[0-9]+]],
; LITENDIAN: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: fadd.d [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.d [[R3]],
; LITENDIAN: .size v2i64_to_v2f64

; BIGENDIAN: v2i64_to_v2f64:
; BIGENDIAN: ld.d [[R1:\$w[0-9]+]],
; BIGENDIAN: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: fadd.d [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; BIGENDIAN: st.d [[R3]],
; BIGENDIAN: .size v2i64_to_v2f64

define void @v2f64_to_v16i8(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <2 x double>, ptr %src
  %1 = tail call <2 x double> @llvm.mips.fadd.d(<2 x double> %0, <2 x double> %0)
  %2 = bitcast <2 x double> %1 to <16 x i8>
  %3 = tail call <16 x i8> @llvm.mips.addv.b(<16 x i8> %2, <16 x i8> %2)
  store <16 x i8> %3, ptr %dst
  ret void
}

; LITENDIAN: v2f64_to_v16i8:
; LITENDIAN: ld.d [[R1:\$w[0-9]+]],
; LITENDIAN: fadd.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.b [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.b [[R3]],
; LITENDIAN: .size v2f64_to_v16i8

; BIGENDIAN: v2f64_to_v16i8:
; BIGENDIAN: ld.d [[R1:\$w[0-9]+]],
; BIGENDIAN: fadd.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.b [[R3:\$w[0-9]+]], [[R2]], 27
; BIGENDIAN: shf.w [[R3:\$w[0-9]+]], [[R3]], 177
; BIGENDIAN: addv.b [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.b [[R4]],
; BIGENDIAN: .size v2f64_to_v16i8

define void @v2f64_to_v8i16(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <2 x double>, ptr %src
  %1 = tail call <2 x double> @llvm.mips.fadd.d(<2 x double> %0, <2 x double> %0)
  %2 = bitcast <2 x double> %1 to <8 x i16>
  %3 = tail call <8 x i16> @llvm.mips.addv.h(<8 x i16> %2, <8 x i16> %2)
  store <8 x i16> %3, ptr %dst
  ret void
}

; LITENDIAN: v2f64_to_v8i16:
; LITENDIAN: ld.d [[R1:\$w[0-9]+]],
; LITENDIAN: fadd.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.h [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.h [[R3]],
; LITENDIAN: .size v2f64_to_v8i16

; BIGENDIAN: v2f64_to_v8i16:
; BIGENDIAN: ld.d [[R1:\$w[0-9]+]],
; BIGENDIAN: fadd.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.h [[R3:\$w[0-9]+]], [[R2]], 27
; BIGENDIAN: addv.h [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.h [[R4]],
; BIGENDIAN: .size v2f64_to_v8i16

; We can't prevent the (store (bitcast X), Y) DAG Combine here because there
; are no operations for v8f16 to put in the way.
define void @v2f64_to_v8f16(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <2 x double>, ptr %src
  %1 = tail call <2 x double> @llvm.mips.fadd.d(<2 x double> %0, <2 x double> %0)
  %2 = bitcast <2 x double> %1 to <8 x half>
  store <8 x half> %2, ptr %dst
  ret void
}

; LITENDIAN: v2f64_to_v8f16:
; LITENDIAN: ld.d [[R1:\$w[0-9]+]],
; LITENDIAN: fadd.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: st.d [[R2]],
; LITENDIAN: .size v2f64_to_v8f16

; BIGENDIAN: v2f64_to_v8f16:
; BIGENDIAN: ld.d [[R1:\$w[0-9]+]],
; BIGENDIAN: fadd.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: st.d [[R2]],
; BIGENDIAN: .size v2f64_to_v8f16

define void @v2f64_to_v4i32(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <2 x double>, ptr %src
  %1 = tail call <2 x double> @llvm.mips.fadd.d(<2 x double> %0, <2 x double> %0)
  %2 = bitcast <2 x double> %1 to <4 x i32>
  %3 = tail call <4 x i32> @llvm.mips.addv.w(<4 x i32> %2, <4 x i32> %2)
  store <4 x i32> %3, ptr %dst
  ret void
}

; LITENDIAN: v2f64_to_v4i32:
; LITENDIAN: ld.d [[R1:\$w[0-9]+]],
; LITENDIAN: fadd.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.w [[R3]],
; LITENDIAN: .size v2f64_to_v4i32

; BIGENDIAN: v2f64_to_v4i32:
; BIGENDIAN: ld.d [[R1:\$w[0-9]+]],
; BIGENDIAN: fadd.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.w [[R3:\$w[0-9]+]], [[R2]], 177
; BIGENDIAN: addv.w [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.w [[R4]],
; BIGENDIAN: .size v2f64_to_v4i32

define void @v2f64_to_v4f32(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <2 x double>, ptr %src
  %1 = tail call <2 x double> @llvm.mips.fadd.d(<2 x double> %0, <2 x double> %0)
  %2 = bitcast <2 x double> %1 to <4 x float>
  %3 = tail call <4 x float> @llvm.mips.fadd.w(<4 x float> %2, <4 x float> %2)
  store <4 x float> %3, ptr %dst
  ret void
}

; LITENDIAN: v2f64_to_v4f32:
; LITENDIAN: ld.d [[R1:\$w[0-9]+]],
; LITENDIAN: fadd.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: fadd.w [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.w [[R3]],
; LITENDIAN: .size v2f64_to_v4f32

; BIGENDIAN: v2f64_to_v4f32:
; BIGENDIAN: ld.d [[R1:\$w[0-9]+]],
; BIGENDIAN: fadd.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: shf.w [[R3:\$w[0-9]+]], [[R2]], 177
; BIGENDIAN: fadd.w [[R4:\$w[0-9]+]], [[R3]], [[R3]]
; BIGENDIAN: st.w [[R4]],
; BIGENDIAN: .size v2f64_to_v4f32

define void @v2f64_to_v2i64(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <2 x double>, ptr %src
  %1 = tail call <2 x double> @llvm.mips.fadd.d(<2 x double> %0, <2 x double> %0)
  %2 = bitcast <2 x double> %1 to <2 x i64>
  %3 = tail call <2 x i64> @llvm.mips.addv.d(<2 x i64> %2, <2 x i64> %2)
  store <2 x i64> %3, ptr %dst
  ret void
}

; LITENDIAN: v2f64_to_v2i64:
; LITENDIAN: ld.d [[R1:\$w[0-9]+]],
; LITENDIAN: fadd.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: addv.d [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.d [[R3]],
; LITENDIAN: .size v2f64_to_v2i64

; BIGENDIAN: v2f64_to_v2i64:
; BIGENDIAN: ld.d [[R1:\$w[0-9]+]],
; BIGENDIAN: fadd.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: addv.d [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; BIGENDIAN: st.d [[R3]],
; BIGENDIAN: .size v2f64_to_v2i64

define void @v2f64_to_v2f64(ptr %src, ptr %dst) nounwind {
entry:
  %0 = load volatile <2 x double>, ptr %src
  %1 = tail call <2 x double> @llvm.mips.fadd.d(<2 x double> %0, <2 x double> %0)
  %2 = bitcast <2 x double> %1 to <2 x double>
  %3 = tail call <2 x double> @llvm.mips.fadd.d(<2 x double> %2, <2 x double> %2)
  store <2 x double> %3, ptr %dst
  ret void
}

; LITENDIAN: v2f64_to_v2f64:
; LITENDIAN: ld.d [[R1:\$w[0-9]+]],
; LITENDIAN: fadd.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; LITENDIAN: fadd.d [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; LITENDIAN: st.d [[R3]],
; LITENDIAN: .size v2f64_to_v2f64

; BIGENDIAN: v2f64_to_v2f64:
; BIGENDIAN: ld.d [[R1:\$w[0-9]+]],
; BIGENDIAN: fadd.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]
; BIGENDIAN: fadd.d [[R3:\$w[0-9]+]], [[R2]], [[R2]]
; BIGENDIAN: st.d [[R3]],
; BIGENDIAN: .size v2f64_to_v2f64

declare <16 x i8> @llvm.mips.addv.b(<16 x i8>, <16 x i8>) nounwind
declare <8 x i16> @llvm.mips.addv.h(<8 x i16>, <8 x i16>) nounwind
declare <4 x i32> @llvm.mips.addv.w(<4 x i32>, <4 x i32>) nounwind
declare <2 x i64> @llvm.mips.addv.d(<2 x i64>, <2 x i64>) nounwind
declare <4 x float> @llvm.mips.fadd.w(<4 x float>, <4 x float>) nounwind
declare <2 x double> @llvm.mips.fadd.d(<2 x double>, <2 x double>) nounwind
