; RUN: llc -mtriple aarch64-arm-none-eabi -asm-verbose=1 -mattr=+bf16 %s -o - | FileCheck %s

%struct.bfloat16x4x2_t = type { [2 x <4 x bfloat>] }
%struct.bfloat16x8x2_t = type { [2 x <8 x bfloat>] }
%struct.bfloat16x4x3_t = type { [3 x <4 x bfloat>] }
%struct.bfloat16x8x3_t = type { [3 x <8 x bfloat>] }
%struct.bfloat16x4x4_t = type { [4 x <4 x bfloat>] }
%struct.bfloat16x8x4_t = type { [4 x <8 x bfloat>] }

define <4 x bfloat> @test_vld1_bf16(bfloat* nocapture readonly %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld1_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ldr d0, [x0]
; CHECK-NEXT:    ret
entry:
  %0 = bitcast bfloat* %ptr to <4 x bfloat>*
  %1 = load <4 x bfloat>, <4 x bfloat>* %0, align 2
  ret <4 x bfloat> %1
}

define <8 x bfloat> @test_vld1q_bf16(bfloat* nocapture readonly %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld1q_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ldr q0, [x0]
; CHECK-NEXT:    ret
entry:
  %0 = bitcast bfloat* %ptr to <8 x bfloat>*
  %1 = load <8 x bfloat>, <8 x bfloat>* %0, align 2
  ret <8 x bfloat> %1
}

define <4 x bfloat> @test_vld1_lane_bf16(bfloat* nocapture readonly %ptr, <4 x bfloat> %src) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld1_lane_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK:    ld1 { v0.h }[0], [x0]
; CHECK:    ret
entry:
  %0 = load bfloat, bfloat* %ptr, align 2
  %vld1_lane = insertelement <4 x bfloat> %src, bfloat %0, i32 0
  ret <4 x bfloat> %vld1_lane
}

define <8 x bfloat> @test_vld1q_lane_bf16(bfloat* nocapture readonly %ptr, <8 x bfloat> %src) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld1q_lane_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld1 { v0.h }[7], [x0]
; CHECK-NEXT:    ret
entry:
  %0 = load bfloat, bfloat* %ptr, align 2
  %vld1_lane = insertelement <8 x bfloat> %src, bfloat %0, i32 7
  ret <8 x bfloat> %vld1_lane
}

define <4 x bfloat> @test_vld1_dup_bf16(bfloat* nocapture readonly %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld1_dup_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld1r { v0.4h }, [x0]
; CHECK-NEXT:    ret
entry:
  %0 = load bfloat, bfloat* %ptr, align 2
  %1 = insertelement <4 x bfloat> undef, bfloat %0, i32 0
  %lane = shufflevector <4 x bfloat> %1, <4 x bfloat> undef, <4 x i32> zeroinitializer
  ret <4 x bfloat> %lane
}

define %struct.bfloat16x4x2_t @test_vld1_bf16_x2(bfloat* %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld1_bf16_x2:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld1 { v0.4h, v1.4h }, [x0]
; CHECK-NEXT:    ret
entry:
  %vld1xN = tail call { <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld1x2.v4bf16.p0bf16(bfloat* %ptr)
  %vld1xN.fca.0.extract = extractvalue { <4 x bfloat>, <4 x bfloat> } %vld1xN, 0
  %vld1xN.fca.1.extract = extractvalue { <4 x bfloat>, <4 x bfloat> } %vld1xN, 1
  %.fca.0.0.insert = insertvalue %struct.bfloat16x4x2_t undef, <4 x bfloat> %vld1xN.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x4x2_t %.fca.0.0.insert, <4 x bfloat> %vld1xN.fca.1.extract, 0, 1
  ret %struct.bfloat16x4x2_t %.fca.0.1.insert
}

declare { <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld1x2.v4bf16.p0bf16(bfloat*) nounwind

define %struct.bfloat16x8x2_t @test_vld1q_bf16_x2(bfloat* %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld1q_bf16_x2:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld1 { v0.8h, v1.8h }, [x0]
; CHECK-NEXT:    ret
entry:
  %vld1xN = tail call { <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld1x2.v8bf16.p0bf16(bfloat* %ptr)
  %vld1xN.fca.0.extract = extractvalue { <8 x bfloat>, <8 x bfloat> } %vld1xN, 0
  %vld1xN.fca.1.extract = extractvalue { <8 x bfloat>, <8 x bfloat> } %vld1xN, 1
  %.fca.0.0.insert = insertvalue %struct.bfloat16x8x2_t undef, <8 x bfloat> %vld1xN.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x8x2_t %.fca.0.0.insert, <8 x bfloat> %vld1xN.fca.1.extract, 0, 1
  ret %struct.bfloat16x8x2_t %.fca.0.1.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld1x2.v8bf16.p0bf16(bfloat*) nounwind

define %struct.bfloat16x4x3_t @test_vld1_bf16_x3(bfloat* %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld1_bf16_x3:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld1 { v0.4h, v1.4h, v2.4h }, [x0]
; CHECK-NEXT:    ret
entry:
  %vld1xN = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld1x3.v4bf16.p0bf16(bfloat* %ptr)
  %vld1xN.fca.0.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld1xN, 0
  %vld1xN.fca.1.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld1xN, 1
  %vld1xN.fca.2.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld1xN, 2
  %.fca.0.0.insert = insertvalue %struct.bfloat16x4x3_t undef, <4 x bfloat> %vld1xN.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x4x3_t %.fca.0.0.insert, <4 x bfloat> %vld1xN.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.bfloat16x4x3_t %.fca.0.1.insert, <4 x bfloat> %vld1xN.fca.2.extract, 0, 2
  ret %struct.bfloat16x4x3_t %.fca.0.2.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld1x3.v4bf16.p0bf16(bfloat*) nounwind

define %struct.bfloat16x8x3_t @test_vld1q_bf16_x3(bfloat* %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld1q_bf16_x3:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld1 { v0.8h, v1.8h, v2.8h }, [x0]
; CHECK-NEXT:    ret
entry:
  %vld1xN = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld1x3.v8bf16.p0bf16(bfloat* %ptr)
  %vld1xN.fca.0.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld1xN, 0
  %vld1xN.fca.1.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld1xN, 1
  %vld1xN.fca.2.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld1xN, 2
  %.fca.0.0.insert = insertvalue %struct.bfloat16x8x3_t undef, <8 x bfloat> %vld1xN.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x8x3_t %.fca.0.0.insert, <8 x bfloat> %vld1xN.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.bfloat16x8x3_t %.fca.0.1.insert, <8 x bfloat> %vld1xN.fca.2.extract, 0, 2
  ret %struct.bfloat16x8x3_t %.fca.0.2.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld1x3.v8bf16.p0bf16(bfloat*) nounwind

define %struct.bfloat16x4x4_t @test_vld1_bf16_x4(bfloat* %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld1_bf16_x4:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld1 { v0.4h, v1.4h, v2.4h, v3.4h }, [x0]
; CHECK-NEXT:    ret
entry:
  %vld1xN = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld1x4.v4bf16.p0bf16(bfloat* %ptr)
  %vld1xN.fca.0.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld1xN, 0
  %vld1xN.fca.1.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld1xN, 1
  %vld1xN.fca.2.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld1xN, 2
  %vld1xN.fca.3.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld1xN, 3
  %.fca.0.0.insert = insertvalue %struct.bfloat16x4x4_t undef, <4 x bfloat> %vld1xN.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x4x4_t %.fca.0.0.insert, <4 x bfloat> %vld1xN.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.bfloat16x4x4_t %.fca.0.1.insert, <4 x bfloat> %vld1xN.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.bfloat16x4x4_t %.fca.0.2.insert, <4 x bfloat> %vld1xN.fca.3.extract, 0, 3
  ret %struct.bfloat16x4x4_t %.fca.0.3.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld1x4.v4bf16.p0bf16(bfloat*) nounwind

define %struct.bfloat16x8x4_t @test_vld1q_bf16_x4(bfloat* %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld1q_bf16_x4:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld1 { v0.8h, v1.8h, v2.8h, v3.8h }, [x0]
; CHECK-NEXT:    ret
entry:
  %vld1xN = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld1x4.v8bf16.p0bf16(bfloat* %ptr)
  %vld1xN.fca.0.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld1xN, 0
  %vld1xN.fca.1.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld1xN, 1
  %vld1xN.fca.2.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld1xN, 2
  %vld1xN.fca.3.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld1xN, 3
  %.fca.0.0.insert = insertvalue %struct.bfloat16x8x4_t undef, <8 x bfloat> %vld1xN.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x8x4_t %.fca.0.0.insert, <8 x bfloat> %vld1xN.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.bfloat16x8x4_t %.fca.0.1.insert, <8 x bfloat> %vld1xN.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.bfloat16x8x4_t %.fca.0.2.insert, <8 x bfloat> %vld1xN.fca.3.extract, 0, 3
  ret %struct.bfloat16x8x4_t %.fca.0.3.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld1x4.v8bf16.p0bf16(bfloat*) nounwind

define <8 x bfloat> @test_vld1q_dup_bf16(bfloat* nocapture readonly %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld1q_dup_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld1r { v0.8h }, [x0]
; CHECK-NEXT:    ret
entry:
  %0 = load bfloat, bfloat* %ptr, align 2
  %1 = insertelement <8 x bfloat> undef, bfloat %0, i32 0
  %lane = shufflevector <8 x bfloat> %1, <8 x bfloat> undef, <8 x i32> zeroinitializer
  ret <8 x bfloat> %lane
}

define %struct.bfloat16x4x2_t @test_vld2_bf16(bfloat* %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld2_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld2 { v0.4h, v1.4h }, [x0]
; CHECK-NEXT:    ret
entry:
  %0 = bitcast bfloat* %ptr to <4 x bfloat>*
  %vld2 = tail call { <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld2.v4bf16.p0v4bf16(<4 x bfloat>* %0)
  %vld2.fca.0.extract = extractvalue { <4 x bfloat>, <4 x bfloat> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <4 x bfloat>, <4 x bfloat> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.bfloat16x4x2_t undef, <4 x bfloat> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x4x2_t %.fca.0.0.insert, <4 x bfloat> %vld2.fca.1.extract, 0, 1
  ret %struct.bfloat16x4x2_t %.fca.0.1.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld2.v4bf16.p0v4bf16(<4 x bfloat>*) nounwind

define %struct.bfloat16x8x2_t @test_vld2q_bf16(bfloat* %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld2q_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld2 { v0.8h, v1.8h }, [x0]
; CHECK-NEXT:    ret
entry:
  %0 = bitcast bfloat* %ptr to <8 x bfloat>*
  %vld2 = tail call { <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld2.v8bf16.p0v8bf16(<8 x bfloat>* %0)
  %vld2.fca.0.extract = extractvalue { <8 x bfloat>, <8 x bfloat> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <8 x bfloat>, <8 x bfloat> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.bfloat16x8x2_t undef, <8 x bfloat> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x8x2_t %.fca.0.0.insert, <8 x bfloat> %vld2.fca.1.extract, 0, 1
  ret %struct.bfloat16x8x2_t %.fca.0.1.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld2.v8bf16.p0v8bf16(<8 x bfloat>*) nounwind
define %struct.bfloat16x4x2_t @test_vld2_lane_bf16(bfloat* %ptr, [2 x <4 x bfloat>] %src.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld2_lane_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK:    ld2 { v0.h, v1.h }[1], [x0]
; CHECK:    ret
entry:
  %src.coerce.fca.0.extract = extractvalue [2 x <4 x bfloat>] %src.coerce, 0
  %src.coerce.fca.1.extract = extractvalue [2 x <4 x bfloat>] %src.coerce, 1
  %0 = bitcast bfloat* %ptr to i8*
  %vld2_lane = tail call { <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld2lane.v4bf16.p0i8(<4 x bfloat> %src.coerce.fca.0.extract, <4 x bfloat> %src.coerce.fca.1.extract, i64 1, i8* %0)
  %vld2_lane.fca.0.extract = extractvalue { <4 x bfloat>, <4 x bfloat> } %vld2_lane, 0
  %vld2_lane.fca.1.extract = extractvalue { <4 x bfloat>, <4 x bfloat> } %vld2_lane, 1
  %.fca.0.0.insert = insertvalue %struct.bfloat16x4x2_t undef, <4 x bfloat> %vld2_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x4x2_t %.fca.0.0.insert, <4 x bfloat> %vld2_lane.fca.1.extract, 0, 1
  ret %struct.bfloat16x4x2_t %.fca.0.1.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld2lane.v4bf16.p0i8(<4 x bfloat>, <4 x bfloat>, i64, i8*) nounwind

define %struct.bfloat16x8x2_t @test_vld2q_lane_bf16(bfloat* %ptr, [2 x <8 x bfloat>] %src.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld2q_lane_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK:    ld2 { v0.h, v1.h }[7], [x0]
; CHECK:    ret
entry:
  %src.coerce.fca.0.extract = extractvalue [2 x <8 x bfloat>] %src.coerce, 0
  %src.coerce.fca.1.extract = extractvalue [2 x <8 x bfloat>] %src.coerce, 1
  %0 = bitcast bfloat* %ptr to i8*
  %vld2_lane = tail call { <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld2lane.v8bf16.p0i8(<8 x bfloat> %src.coerce.fca.0.extract, <8 x bfloat> %src.coerce.fca.1.extract, i64 7, i8* %0)
  %vld2_lane.fca.0.extract = extractvalue { <8 x bfloat>, <8 x bfloat> } %vld2_lane, 0
  %vld2_lane.fca.1.extract = extractvalue { <8 x bfloat>, <8 x bfloat> } %vld2_lane, 1
  %.fca.0.0.insert = insertvalue %struct.bfloat16x8x2_t undef, <8 x bfloat> %vld2_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x8x2_t %.fca.0.0.insert, <8 x bfloat> %vld2_lane.fca.1.extract, 0, 1
  ret %struct.bfloat16x8x2_t %.fca.0.1.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld2lane.v8bf16.p0i8(<8 x bfloat>, <8 x bfloat>, i64, i8*) nounwind

define %struct.bfloat16x4x3_t @test_vld3_bf16(bfloat* %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld3_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld3 { v0.4h, v1.4h, v2.4h }, [x0]
; CHECK-NEXT:    ret
entry:
  %0 = bitcast bfloat* %ptr to <4 x bfloat>*
  %vld3 = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld3.v4bf16.p0v4bf16(<4 x bfloat>* %0)
  %vld3.fca.0.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.bfloat16x4x3_t undef, <4 x bfloat> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x4x3_t %.fca.0.0.insert, <4 x bfloat> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.bfloat16x4x3_t %.fca.0.1.insert, <4 x bfloat> %vld3.fca.2.extract, 0, 2
  ret %struct.bfloat16x4x3_t %.fca.0.2.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld3.v4bf16.p0v4bf16(<4 x bfloat>*) nounwind

define %struct.bfloat16x8x3_t @test_vld3q_bf16(bfloat* %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld3q_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld3 { v0.8h, v1.8h, v2.8h }, [x0]
; CHECK-NEXT:    ret
entry:
  %0 = bitcast bfloat* %ptr to <8 x bfloat>*
  %vld3 = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld3.v8bf16.p0v8bf16(<8 x bfloat>* %0)
  %vld3.fca.0.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.bfloat16x8x3_t undef, <8 x bfloat> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x8x3_t %.fca.0.0.insert, <8 x bfloat> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.bfloat16x8x3_t %.fca.0.1.insert, <8 x bfloat> %vld3.fca.2.extract, 0, 2
  ret %struct.bfloat16x8x3_t %.fca.0.2.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld3.v8bf16.p0v8bf16(<8 x bfloat>*) nounwind

define %struct.bfloat16x4x3_t @test_vld3_lane_bf16(bfloat* %ptr, [3 x <4 x bfloat>] %src.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld3_lane_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK:    ld3 { v0.h, v1.h, v2.h }[1], [x0]
; CHECK:    ret
entry:
  %src.coerce.fca.0.extract = extractvalue [3 x <4 x bfloat>] %src.coerce, 0
  %src.coerce.fca.1.extract = extractvalue [3 x <4 x bfloat>] %src.coerce, 1
  %src.coerce.fca.2.extract = extractvalue [3 x <4 x bfloat>] %src.coerce, 2
  %0 = bitcast bfloat* %ptr to i8*
  %vld3_lane = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld3lane.v4bf16.p0i8(<4 x bfloat> %src.coerce.fca.0.extract, <4 x bfloat> %src.coerce.fca.1.extract, <4 x bfloat> %src.coerce.fca.2.extract, i64 1, i8* %0)
  %vld3_lane.fca.0.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld3_lane, 2
  %.fca.0.0.insert = insertvalue %struct.bfloat16x4x3_t undef, <4 x bfloat> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x4x3_t %.fca.0.0.insert, <4 x bfloat> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.bfloat16x4x3_t %.fca.0.1.insert, <4 x bfloat> %vld3_lane.fca.2.extract, 0, 2
  ret %struct.bfloat16x4x3_t %.fca.0.2.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld3lane.v4bf16.p0i8(<4 x bfloat>, <4 x bfloat>, <4 x bfloat>, i64, i8*) nounwind

define %struct.bfloat16x8x3_t @test_vld3q_lane_bf16(bfloat* %ptr, [3 x <8 x bfloat>] %src.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld3q_lane_bf16:
; CHECK:       // %bb.0: // %entry
; CHECKT:    ld3 { v0.h, v1.h, v2.h }[7], [x0]
; CHECKT:    ret
entry:
  %src.coerce.fca.0.extract = extractvalue [3 x <8 x bfloat>] %src.coerce, 0
  %src.coerce.fca.1.extract = extractvalue [3 x <8 x bfloat>] %src.coerce, 1
  %src.coerce.fca.2.extract = extractvalue [3 x <8 x bfloat>] %src.coerce, 2
  %0 = bitcast bfloat* %ptr to i8*
  %vld3_lane = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld3lane.v8bf16.p0i8(<8 x bfloat> %src.coerce.fca.0.extract, <8 x bfloat> %src.coerce.fca.1.extract, <8 x bfloat> %src.coerce.fca.2.extract, i64 7, i8* %0)
  %vld3_lane.fca.0.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld3_lane, 0
  %vld3_lane.fca.1.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld3_lane, 1
  %vld3_lane.fca.2.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld3_lane, 2
  %.fca.0.0.insert = insertvalue %struct.bfloat16x8x3_t undef, <8 x bfloat> %vld3_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x8x3_t %.fca.0.0.insert, <8 x bfloat> %vld3_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.bfloat16x8x3_t %.fca.0.1.insert, <8 x bfloat> %vld3_lane.fca.2.extract, 0, 2
  ret %struct.bfloat16x8x3_t %.fca.0.2.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld3lane.v8bf16.p0i8(<8 x bfloat>, <8 x bfloat>, <8 x bfloat>, i64, i8*) nounwind

define %struct.bfloat16x4x4_t @test_vld4_bf16(bfloat* %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld4_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld4 { v0.4h, v1.4h, v2.4h, v3.4h }, [x0]
; CHECK-NEXT:    ret
entry:
  %0 = bitcast bfloat* %ptr to <4 x bfloat>*
  %vld4 = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld4.v4bf16.p0v4bf16(<4 x bfloat>* %0)
  %vld4.fca.0.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.bfloat16x4x4_t undef, <4 x bfloat> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x4x4_t %.fca.0.0.insert, <4 x bfloat> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.bfloat16x4x4_t %.fca.0.1.insert, <4 x bfloat> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.bfloat16x4x4_t %.fca.0.2.insert, <4 x bfloat> %vld4.fca.3.extract, 0, 3
  ret %struct.bfloat16x4x4_t %.fca.0.3.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld4.v4bf16.p0v4bf16(<4 x bfloat>*) nounwind

define %struct.bfloat16x8x4_t @test_vld4q_bf16(bfloat* %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld4q_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld4 { v0.8h, v1.8h, v2.8h, v3.8h }, [x0]
; CHECK-NEXT:    ret
entry:
  %0 = bitcast bfloat* %ptr to <8 x bfloat>*
  %vld4 = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld4.v8bf16.p0v8bf16(<8 x bfloat>* %0)
  %vld4.fca.0.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.bfloat16x8x4_t undef, <8 x bfloat> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x8x4_t %.fca.0.0.insert, <8 x bfloat> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.bfloat16x8x4_t %.fca.0.1.insert, <8 x bfloat> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.bfloat16x8x4_t %.fca.0.2.insert, <8 x bfloat> %vld4.fca.3.extract, 0, 3
  ret %struct.bfloat16x8x4_t %.fca.0.3.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld4.v8bf16.p0v8bf16(<8 x bfloat>*) nounwind

define %struct.bfloat16x4x4_t @test_vld4_lane_bf16(bfloat* %ptr, [4 x <4 x bfloat>] %src.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld4_lane_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK:    ld4 { v0.h, v1.h, v2.h, v3.h }[1], [x0]
; CHECK:    ret
entry:
  %src.coerce.fca.0.extract = extractvalue [4 x <4 x bfloat>] %src.coerce, 0
  %src.coerce.fca.1.extract = extractvalue [4 x <4 x bfloat>] %src.coerce, 1
  %src.coerce.fca.2.extract = extractvalue [4 x <4 x bfloat>] %src.coerce, 2
  %src.coerce.fca.3.extract = extractvalue [4 x <4 x bfloat>] %src.coerce, 3
  %0 = bitcast bfloat* %ptr to i8*
  %vld4_lane = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld4lane.v4bf16.p0i8(<4 x bfloat> %src.coerce.fca.0.extract, <4 x bfloat> %src.coerce.fca.1.extract, <4 x bfloat> %src.coerce.fca.2.extract, <4 x bfloat> %src.coerce.fca.3.extract, i64 1, i8* %0)
  %vld4_lane.fca.0.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld4_lane, 0
  %vld4_lane.fca.1.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld4_lane, 1
  %vld4_lane.fca.2.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld4_lane, 2
  %vld4_lane.fca.3.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld4_lane, 3
  %.fca.0.0.insert = insertvalue %struct.bfloat16x4x4_t undef, <4 x bfloat> %vld4_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x4x4_t %.fca.0.0.insert, <4 x bfloat> %vld4_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.bfloat16x4x4_t %.fca.0.1.insert, <4 x bfloat> %vld4_lane.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.bfloat16x4x4_t %.fca.0.2.insert, <4 x bfloat> %vld4_lane.fca.3.extract, 0, 3
  ret %struct.bfloat16x4x4_t %.fca.0.3.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld4lane.v4bf16.p0i8(<4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, i64, i8*) nounwind

define %struct.bfloat16x8x4_t @test_vld4q_lane_bf16(bfloat* %ptr, [4 x <8 x bfloat>] %src.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld4q_lane_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK:    ld4 { v0.h, v1.h, v2.h, v3.h }[7], [x0]
; CHECK:    ret
entry:
  %src.coerce.fca.0.extract = extractvalue [4 x <8 x bfloat>] %src.coerce, 0
  %src.coerce.fca.1.extract = extractvalue [4 x <8 x bfloat>] %src.coerce, 1
  %src.coerce.fca.2.extract = extractvalue [4 x <8 x bfloat>] %src.coerce, 2
  %src.coerce.fca.3.extract = extractvalue [4 x <8 x bfloat>] %src.coerce, 3
  %0 = bitcast bfloat* %ptr to i8*
  %vld4_lane = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld4lane.v8bf16.p0i8(<8 x bfloat> %src.coerce.fca.0.extract, <8 x bfloat> %src.coerce.fca.1.extract, <8 x bfloat> %src.coerce.fca.2.extract, <8 x bfloat> %src.coerce.fca.3.extract, i64 7, i8* %0)
  %vld4_lane.fca.0.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld4_lane, 0
  %vld4_lane.fca.1.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld4_lane, 1
  %vld4_lane.fca.2.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld4_lane, 2
  %vld4_lane.fca.3.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld4_lane, 3
  %.fca.0.0.insert = insertvalue %struct.bfloat16x8x4_t undef, <8 x bfloat> %vld4_lane.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x8x4_t %.fca.0.0.insert, <8 x bfloat> %vld4_lane.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.bfloat16x8x4_t %.fca.0.1.insert, <8 x bfloat> %vld4_lane.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.bfloat16x8x4_t %.fca.0.2.insert, <8 x bfloat> %vld4_lane.fca.3.extract, 0, 3
  ret %struct.bfloat16x8x4_t %.fca.0.3.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld4lane.v8bf16.p0i8(<8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, i64, i8*) nounwind

define %struct.bfloat16x4x2_t @test_vld2_dup_bf16(bfloat* %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld2_dup_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld2r { v0.4h, v1.4h }, [x0]
; CHECK-NEXT:    ret
entry:
  %vld2 = tail call { <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld2r.v4bf16.p0bf16(bfloat* %ptr)
  %vld2.fca.0.extract = extractvalue { <4 x bfloat>, <4 x bfloat> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <4 x bfloat>, <4 x bfloat> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.bfloat16x4x2_t undef, <4 x bfloat> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x4x2_t %.fca.0.0.insert, <4 x bfloat> %vld2.fca.1.extract, 0, 1
  ret %struct.bfloat16x4x2_t %.fca.0.1.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld2r.v4bf16.p0bf16(bfloat*) nounwind

define %struct.bfloat16x8x2_t @test_vld2q_dup_bf16(bfloat* %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld2q_dup_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld2r { v0.8h, v1.8h }, [x0]
; CHECK-NEXT:    ret
entry:
  %vld2 = tail call { <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld2r.v8bf16.p0bf16(bfloat* %ptr)
  %vld2.fca.0.extract = extractvalue { <8 x bfloat>, <8 x bfloat> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <8 x bfloat>, <8 x bfloat> } %vld2, 1
  %.fca.0.0.insert = insertvalue %struct.bfloat16x8x2_t undef, <8 x bfloat> %vld2.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x8x2_t %.fca.0.0.insert, <8 x bfloat> %vld2.fca.1.extract, 0, 1
  ret %struct.bfloat16x8x2_t %.fca.0.1.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld2r.v8bf16.p0bf16(bfloat*) nounwind

define %struct.bfloat16x4x3_t @test_vld3_dup_bf16(bfloat* %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld3_dup_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld3r { v0.4h, v1.4h, v2.4h }, [x0]
; CHECK-NEXT:    ret
entry:
  %vld3 = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld3r.v4bf16.p0bf16(bfloat* %ptr)
  %vld3.fca.0.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.bfloat16x4x3_t undef, <4 x bfloat> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x4x3_t %.fca.0.0.insert, <4 x bfloat> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.bfloat16x4x3_t %.fca.0.1.insert, <4 x bfloat> %vld3.fca.2.extract, 0, 2
  ret %struct.bfloat16x4x3_t %.fca.0.2.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld3r.v4bf16.p0bf16(bfloat*) nounwind

define %struct.bfloat16x8x3_t @test_vld3q_dup_bf16(bfloat* %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld3q_dup_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld3r { v0.8h, v1.8h, v2.8h }, [x0]
; CHECK-NEXT:    ret
entry:
  %vld3 = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld3r.v8bf16.p0bf16(bfloat* %ptr)
  %vld3.fca.0.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld3, 1
  %vld3.fca.2.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld3, 2
  %.fca.0.0.insert = insertvalue %struct.bfloat16x8x3_t undef, <8 x bfloat> %vld3.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x8x3_t %.fca.0.0.insert, <8 x bfloat> %vld3.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.bfloat16x8x3_t %.fca.0.1.insert, <8 x bfloat> %vld3.fca.2.extract, 0, 2
  ret %struct.bfloat16x8x3_t %.fca.0.2.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld3r.v8bf16.p0bf16(bfloat*) nounwind

define %struct.bfloat16x4x4_t @test_vld4_dup_bf16(bfloat* %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld4_dup_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld4r { v0.4h, v1.4h, v2.4h, v3.4h }, [x0]
; CHECK-NEXT:    ret
entry:
  %vld4 = tail call { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld4r.v4bf16.p0bf16(bfloat* %ptr)
  %vld4.fca.0.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.bfloat16x4x4_t undef, <4 x bfloat> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x4x4_t %.fca.0.0.insert, <4 x bfloat> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.bfloat16x4x4_t %.fca.0.1.insert, <4 x bfloat> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.bfloat16x4x4_t %.fca.0.2.insert, <4 x bfloat> %vld4.fca.3.extract, 0, 3
  ret %struct.bfloat16x4x4_t %.fca.0.3.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat> } @llvm.aarch64.neon.ld4r.v4bf16.p0bf16(bfloat*) nounwind

define %struct.bfloat16x8x4_t @test_vld4q_dup_bf16(bfloat* %ptr) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vld4q_dup_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ld4r { v0.8h, v1.8h, v2.8h, v3.8h }, [x0]
; CHECK-NEXT:    ret
entry:
  %vld4 = tail call { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld4r.v8bf16.p0bf16(bfloat* %ptr)
  %vld4.fca.0.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld4, 0
  %vld4.fca.1.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld4, 1
  %vld4.fca.2.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld4, 2
  %vld4.fca.3.extract = extractvalue { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } %vld4, 3
  %.fca.0.0.insert = insertvalue %struct.bfloat16x8x4_t undef, <8 x bfloat> %vld4.fca.0.extract, 0, 0
  %.fca.0.1.insert = insertvalue %struct.bfloat16x8x4_t %.fca.0.0.insert, <8 x bfloat> %vld4.fca.1.extract, 0, 1
  %.fca.0.2.insert = insertvalue %struct.bfloat16x8x4_t %.fca.0.1.insert, <8 x bfloat> %vld4.fca.2.extract, 0, 2
  %.fca.0.3.insert = insertvalue %struct.bfloat16x8x4_t %.fca.0.2.insert, <8 x bfloat> %vld4.fca.3.extract, 0, 3
  ret %struct.bfloat16x8x4_t %.fca.0.3.insert
}

; Function Attrs: argmemonly nounwind readonly
declare { <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat> } @llvm.aarch64.neon.ld4r.v8bf16.p0bf16(bfloat*) nounwind

define void @test_vst1_bf16(bfloat* nocapture %ptr, <4 x bfloat> %val) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst1_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    str d0, [x0]
; CHECK-NEXT:    ret
entry:
  %0 = bitcast bfloat* %ptr to <4 x bfloat>*
  store <4 x bfloat> %val, <4 x bfloat>* %0, align 8
  ret void
}

define void @test_vst1q_bf16(bfloat* nocapture %ptr, <8 x bfloat> %val) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst1q_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    str q0, [x0]
; CHECK-NEXT:    ret
entry:
  %0 = bitcast bfloat* %ptr to <8 x bfloat>*
  store <8 x bfloat> %val, <8 x bfloat>* %0, align 16
  ret void
}

define void @test_vst1_lane_bf16(bfloat* nocapture %ptr, <4 x bfloat> %val) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst1_lane_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK:    st1 { v0.h }[1], [x0]
; CHECK:    ret
entry:
  %0 = extractelement <4 x bfloat> %val, i32 1
  store bfloat %0, bfloat* %ptr, align 2
  ret void
}

define void @test_vst1q_lane_bf16(bfloat* nocapture %ptr, <8 x bfloat> %val) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst1q_lane_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    st1 { v0.h }[7], [x0]
; CHECK-NEXT:    ret
entry:
  %0 = extractelement <8 x bfloat> %val, i32 7
  store bfloat %0, bfloat* %ptr, align 2
  ret void
}

define void @test_vst1_bf16_x2(bfloat* nocapture %ptr, [2 x <4 x bfloat>] %val.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst1_bf16_x2:
; CHECK:       // %bb.0: // %entry
; CHECK:    st1 { v0.4h, v1.4h }, [x0]
; CHECK:    ret
entry:
  %val.coerce.fca.0.extract = extractvalue [2 x <4 x bfloat>] %val.coerce, 0
  %val.coerce.fca.1.extract = extractvalue [2 x <4 x bfloat>] %val.coerce, 1
  tail call void @llvm.aarch64.neon.st1x2.v4bf16.p0bf16(<4 x bfloat> %val.coerce.fca.0.extract, <4 x bfloat> %val.coerce.fca.1.extract, bfloat* %ptr)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.aarch64.neon.st1x2.v4bf16.p0bf16(<4 x bfloat>, <4 x bfloat>, bfloat* nocapture) nounwind

define void @test_vst1q_bf16_x2(bfloat* nocapture %ptr, [2 x <8 x bfloat>] %val.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst1q_bf16_x2:
; CHECK:       // %bb.0: // %entry
; CHECK:    st1 { v0.8h, v1.8h }, [x0]
; CHECK:    ret
entry:
  %val.coerce.fca.0.extract = extractvalue [2 x <8 x bfloat>] %val.coerce, 0
  %val.coerce.fca.1.extract = extractvalue [2 x <8 x bfloat>] %val.coerce, 1
  tail call void @llvm.aarch64.neon.st1x2.v8bf16.p0bf16(<8 x bfloat> %val.coerce.fca.0.extract, <8 x bfloat> %val.coerce.fca.1.extract, bfloat* %ptr)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.aarch64.neon.st1x2.v8bf16.p0bf16(<8 x bfloat>, <8 x bfloat>, bfloat* nocapture) nounwind

define void @test_vst1_bf16_x3(bfloat* nocapture %ptr, [3 x <4 x bfloat>] %val.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst1_bf16_x3:
; CHECK:       // %bb.0: // %entry
; CHECK:    st1 { v0.4h, v1.4h, v2.4h }, [x0]
; CHECK:    ret
entry:
  %val.coerce.fca.0.extract = extractvalue [3 x <4 x bfloat>] %val.coerce, 0
  %val.coerce.fca.1.extract = extractvalue [3 x <4 x bfloat>] %val.coerce, 1
  %val.coerce.fca.2.extract = extractvalue [3 x <4 x bfloat>] %val.coerce, 2
  tail call void @llvm.aarch64.neon.st1x3.v4bf16.p0bf16(<4 x bfloat> %val.coerce.fca.0.extract, <4 x bfloat> %val.coerce.fca.1.extract, <4 x bfloat> %val.coerce.fca.2.extract, bfloat* %ptr)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.aarch64.neon.st1x3.v4bf16.p0bf16(<4 x bfloat>, <4 x bfloat>, <4 x bfloat>, bfloat* nocapture) nounwind

define void @test_vst1q_bf16_x3(bfloat* nocapture %ptr, [3 x <8 x bfloat>] %val.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst1q_bf16_x3:
; CHECK:       // %bb.0: // %entry
; CHECK:    st1 { v0.8h, v1.8h, v2.8h }, [x0]
; CHECK:    ret
entry:
  %val.coerce.fca.0.extract = extractvalue [3 x <8 x bfloat>] %val.coerce, 0
  %val.coerce.fca.1.extract = extractvalue [3 x <8 x bfloat>] %val.coerce, 1
  %val.coerce.fca.2.extract = extractvalue [3 x <8 x bfloat>] %val.coerce, 2
  tail call void @llvm.aarch64.neon.st1x3.v8bf16.p0bf16(<8 x bfloat> %val.coerce.fca.0.extract, <8 x bfloat> %val.coerce.fca.1.extract, <8 x bfloat> %val.coerce.fca.2.extract, bfloat* %ptr)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.aarch64.neon.st1x3.v8bf16.p0bf16(<8 x bfloat>, <8 x bfloat>, <8 x bfloat>, bfloat* nocapture) nounwind

; Function Attrs: nounwind
define void @test_vst1_bf16_x4(bfloat* nocapture %ptr, [4 x <4 x bfloat>] %val.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst1_bf16_x4:
; CHECK:       // %bb.0: // %entry
; CHECK:    st1 { v0.4h, v1.4h, v2.4h, v3.4h }, [x0]
; CHECK:    ret
entry:
  %val.coerce.fca.0.extract = extractvalue [4 x <4 x bfloat>] %val.coerce, 0
  %val.coerce.fca.1.extract = extractvalue [4 x <4 x bfloat>] %val.coerce, 1
  %val.coerce.fca.2.extract = extractvalue [4 x <4 x bfloat>] %val.coerce, 2
  %val.coerce.fca.3.extract = extractvalue [4 x <4 x bfloat>] %val.coerce, 3
  tail call void @llvm.aarch64.neon.st1x4.v4bf16.p0bf16(<4 x bfloat> %val.coerce.fca.0.extract, <4 x bfloat> %val.coerce.fca.1.extract, <4 x bfloat> %val.coerce.fca.2.extract, <4 x bfloat> %val.coerce.fca.3.extract, bfloat* %ptr)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.aarch64.neon.st1x4.v4bf16.p0bf16(<4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, bfloat* nocapture) nounwind

define void @test_vst1q_bf16_x4(bfloat* nocapture %ptr, [4 x <8 x bfloat>] %val.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst1q_bf16_x4:
; CHECK:       // %bb.0: // %entry
; CHECK:    st1 { v0.8h, v1.8h, v2.8h, v3.8h }, [x0]
; CHECK:    ret
entry:
  %val.coerce.fca.0.extract = extractvalue [4 x <8 x bfloat>] %val.coerce, 0
  %val.coerce.fca.1.extract = extractvalue [4 x <8 x bfloat>] %val.coerce, 1
  %val.coerce.fca.2.extract = extractvalue [4 x <8 x bfloat>] %val.coerce, 2
  %val.coerce.fca.3.extract = extractvalue [4 x <8 x bfloat>] %val.coerce, 3
  tail call void @llvm.aarch64.neon.st1x4.v8bf16.p0bf16(<8 x bfloat> %val.coerce.fca.0.extract, <8 x bfloat> %val.coerce.fca.1.extract, <8 x bfloat> %val.coerce.fca.2.extract, <8 x bfloat> %val.coerce.fca.3.extract, bfloat* %ptr)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.aarch64.neon.st1x4.v8bf16.p0bf16(<8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, bfloat* nocapture) nounwind

define void @test_vst2_bf16(bfloat* nocapture %ptr, [2 x <4 x bfloat>] %val.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst2_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK:    st2 { v0.4h, v1.4h }, [x0]
; CHECK:    ret
entry:
  %val.coerce.fca.0.extract = extractvalue [2 x <4 x bfloat>] %val.coerce, 0
  %val.coerce.fca.1.extract = extractvalue [2 x <4 x bfloat>] %val.coerce, 1
  %0 = bitcast bfloat* %ptr to i8*
  tail call void @llvm.aarch64.neon.st2.v4bf16.p0i8(<4 x bfloat> %val.coerce.fca.0.extract, <4 x bfloat> %val.coerce.fca.1.extract, i8* %0)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.aarch64.neon.st2.v4bf16.p0i8(<4 x bfloat>, <4 x bfloat>, i8* nocapture) nounwind

define void @test_vst2q_bf16(bfloat* nocapture %ptr, [2 x <8 x bfloat>] %val.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst2q_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK:    st2 { v0.8h, v1.8h }, [x0]
; CHECK:    ret
entry:
  %val.coerce.fca.0.extract = extractvalue [2 x <8 x bfloat>] %val.coerce, 0
  %val.coerce.fca.1.extract = extractvalue [2 x <8 x bfloat>] %val.coerce, 1
  %0 = bitcast bfloat* %ptr to i8*
  tail call void @llvm.aarch64.neon.st2.v8bf16.p0i8(<8 x bfloat> %val.coerce.fca.0.extract, <8 x bfloat> %val.coerce.fca.1.extract, i8* %0)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.aarch64.neon.st2.v8bf16.p0i8(<8 x bfloat>, <8 x bfloat>, i8* nocapture) nounwind

define void @test_vst2_lane_bf16(bfloat* nocapture %ptr, [2 x <4 x bfloat>] %val.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst2_lane_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK:    st2 { v0.h, v1.h }[1], [x0]
; CHECK:    ret
entry:
  %val.coerce.fca.0.extract = extractvalue [2 x <4 x bfloat>] %val.coerce, 0
  %val.coerce.fca.1.extract = extractvalue [2 x <4 x bfloat>] %val.coerce, 1
  %0 = bitcast bfloat* %ptr to i8*
  tail call void @llvm.aarch64.neon.st2lane.v4bf16.p0i8(<4 x bfloat> %val.coerce.fca.0.extract, <4 x bfloat> %val.coerce.fca.1.extract, i64 1, i8* %0)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.aarch64.neon.st2lane.v4bf16.p0i8(<4 x bfloat>, <4 x bfloat>, i64, i8* nocapture) nounwind

; Function Attrs: nounwind
define void @test_vst2q_lane_bf16(bfloat* nocapture %ptr, [2 x <8 x bfloat>] %val.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst2q_lane_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK:    st2 { v0.h, v1.h }[7], [x0]
; CHECK:    ret
entry:
  %val.coerce.fca.0.extract = extractvalue [2 x <8 x bfloat>] %val.coerce, 0
  %val.coerce.fca.1.extract = extractvalue [2 x <8 x bfloat>] %val.coerce, 1
  %0 = bitcast bfloat* %ptr to i8*
  tail call void @llvm.aarch64.neon.st2lane.v8bf16.p0i8(<8 x bfloat> %val.coerce.fca.0.extract, <8 x bfloat> %val.coerce.fca.1.extract, i64 7, i8* %0)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.aarch64.neon.st2lane.v8bf16.p0i8(<8 x bfloat>, <8 x bfloat>, i64, i8* nocapture) nounwind

; Function Attrs: nounwind
define void @test_vst3_bf16(bfloat* nocapture %ptr, [3 x <4 x bfloat>] %val.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst3_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK:    st3 { v0.4h, v1.4h, v2.4h }, [x0]
; CHECK:    ret
entry:
  %val.coerce.fca.0.extract = extractvalue [3 x <4 x bfloat>] %val.coerce, 0
  %val.coerce.fca.1.extract = extractvalue [3 x <4 x bfloat>] %val.coerce, 1
  %val.coerce.fca.2.extract = extractvalue [3 x <4 x bfloat>] %val.coerce, 2
  %0 = bitcast bfloat* %ptr to i8*
  tail call void @llvm.aarch64.neon.st3.v4bf16.p0i8(<4 x bfloat> %val.coerce.fca.0.extract, <4 x bfloat> %val.coerce.fca.1.extract, <4 x bfloat> %val.coerce.fca.2.extract, i8* %0)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.aarch64.neon.st3.v4bf16.p0i8(<4 x bfloat>, <4 x bfloat>, <4 x bfloat>, i8* nocapture) nounwind

; Function Attrs: nounwind
define void @test_vst3q_bf16(bfloat* nocapture %ptr, [3 x <8 x bfloat>] %val.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst3q_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK:    st3 { v0.8h, v1.8h, v2.8h }, [x0]
; CHECK:    ret
entry:
  %val.coerce.fca.0.extract = extractvalue [3 x <8 x bfloat>] %val.coerce, 0
  %val.coerce.fca.1.extract = extractvalue [3 x <8 x bfloat>] %val.coerce, 1
  %val.coerce.fca.2.extract = extractvalue [3 x <8 x bfloat>] %val.coerce, 2
  %0 = bitcast bfloat* %ptr to i8*
  tail call void @llvm.aarch64.neon.st3.v8bf16.p0i8(<8 x bfloat> %val.coerce.fca.0.extract, <8 x bfloat> %val.coerce.fca.1.extract, <8 x bfloat> %val.coerce.fca.2.extract, i8* %0)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.aarch64.neon.st3.v8bf16.p0i8(<8 x bfloat>, <8 x bfloat>, <8 x bfloat>, i8* nocapture) nounwind

; Function Attrs: nounwind
define void @test_vst3_lane_bf16(bfloat* nocapture %ptr, [3 x <4 x bfloat>] %val.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst3_lane_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK:    st3 { v0.h, v1.h, v2.h }[1], [x0]
; CHECK:    ret
entry:
  %val.coerce.fca.0.extract = extractvalue [3 x <4 x bfloat>] %val.coerce, 0
  %val.coerce.fca.1.extract = extractvalue [3 x <4 x bfloat>] %val.coerce, 1
  %val.coerce.fca.2.extract = extractvalue [3 x <4 x bfloat>] %val.coerce, 2
  %0 = bitcast bfloat* %ptr to i8*
  tail call void @llvm.aarch64.neon.st3lane.v4bf16.p0i8(<4 x bfloat> %val.coerce.fca.0.extract, <4 x bfloat> %val.coerce.fca.1.extract, <4 x bfloat> %val.coerce.fca.2.extract, i64 1, i8* %0)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.aarch64.neon.st3lane.v4bf16.p0i8(<4 x bfloat>, <4 x bfloat>, <4 x bfloat>, i64, i8* nocapture) nounwind

; Function Attrs: nounwind
define void @test_vst3q_lane_bf16(bfloat* nocapture %ptr, [3 x <8 x bfloat>] %val.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst3q_lane_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK:    st3 { v0.h, v1.h, v2.h }[7], [x0]
; CHECK:    ret
entry:
  %val.coerce.fca.0.extract = extractvalue [3 x <8 x bfloat>] %val.coerce, 0
  %val.coerce.fca.1.extract = extractvalue [3 x <8 x bfloat>] %val.coerce, 1
  %val.coerce.fca.2.extract = extractvalue [3 x <8 x bfloat>] %val.coerce, 2
  %0 = bitcast bfloat* %ptr to i8*
  tail call void @llvm.aarch64.neon.st3lane.v8bf16.p0i8(<8 x bfloat> %val.coerce.fca.0.extract, <8 x bfloat> %val.coerce.fca.1.extract, <8 x bfloat> %val.coerce.fca.2.extract, i64 7, i8* %0)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.aarch64.neon.st3lane.v8bf16.p0i8(<8 x bfloat>, <8 x bfloat>, <8 x bfloat>, i64, i8* nocapture) nounwind

; Function Attrs: nounwind
define void @test_vst4_bf16(bfloat* nocapture %ptr, [4 x <4 x bfloat>] %val.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst4_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK:    st4 { v0.4h, v1.4h, v2.4h, v3.4h }, [x0]
; CHECK:    ret
entry:
  %val.coerce.fca.0.extract = extractvalue [4 x <4 x bfloat>] %val.coerce, 0
  %val.coerce.fca.1.extract = extractvalue [4 x <4 x bfloat>] %val.coerce, 1
  %val.coerce.fca.2.extract = extractvalue [4 x <4 x bfloat>] %val.coerce, 2
  %val.coerce.fca.3.extract = extractvalue [4 x <4 x bfloat>] %val.coerce, 3
  %0 = bitcast bfloat* %ptr to i8*
  tail call void @llvm.aarch64.neon.st4.v4bf16.p0i8(<4 x bfloat> %val.coerce.fca.0.extract, <4 x bfloat> %val.coerce.fca.1.extract, <4 x bfloat> %val.coerce.fca.2.extract, <4 x bfloat> %val.coerce.fca.3.extract, i8* %0)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.aarch64.neon.st4.v4bf16.p0i8(<4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, i8* nocapture) nounwind

; Function Attrs: nounwind
define void @test_vst4q_bf16(bfloat* nocapture %ptr, [4 x <8 x bfloat>] %val.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst4q_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK:    st4 { v0.8h, v1.8h, v2.8h, v3.8h }, [x0]
; CHECK:    ret
entry:
  %val.coerce.fca.0.extract = extractvalue [4 x <8 x bfloat>] %val.coerce, 0
  %val.coerce.fca.1.extract = extractvalue [4 x <8 x bfloat>] %val.coerce, 1
  %val.coerce.fca.2.extract = extractvalue [4 x <8 x bfloat>] %val.coerce, 2
  %val.coerce.fca.3.extract = extractvalue [4 x <8 x bfloat>] %val.coerce, 3
  %0 = bitcast bfloat* %ptr to i8*
  tail call void @llvm.aarch64.neon.st4.v8bf16.p0i8(<8 x bfloat> %val.coerce.fca.0.extract, <8 x bfloat> %val.coerce.fca.1.extract, <8 x bfloat> %val.coerce.fca.2.extract, <8 x bfloat> %val.coerce.fca.3.extract, i8* %0)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.aarch64.neon.st4.v8bf16.p0i8(<8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, i8* nocapture) nounwind

; Function Attrs: nounwind
define void @test_vst4_lane_bf16(bfloat* nocapture %ptr, [4 x <4 x bfloat>] %val.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst4_lane_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK:    st4 { v0.h, v1.h, v2.h, v3.h }[1], [x0]
; CHECK:    ret
entry:
  %val.coerce.fca.0.extract = extractvalue [4 x <4 x bfloat>] %val.coerce, 0
  %val.coerce.fca.1.extract = extractvalue [4 x <4 x bfloat>] %val.coerce, 1
  %val.coerce.fca.2.extract = extractvalue [4 x <4 x bfloat>] %val.coerce, 2
  %val.coerce.fca.3.extract = extractvalue [4 x <4 x bfloat>] %val.coerce, 3
  %0 = bitcast bfloat* %ptr to i8*
  tail call void @llvm.aarch64.neon.st4lane.v4bf16.p0i8(<4 x bfloat> %val.coerce.fca.0.extract, <4 x bfloat> %val.coerce.fca.1.extract, <4 x bfloat> %val.coerce.fca.2.extract, <4 x bfloat> %val.coerce.fca.3.extract, i64 1, i8* %0)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.aarch64.neon.st4lane.v4bf16.p0i8(<4 x bfloat>, <4 x bfloat>, <4 x bfloat>, <4 x bfloat>, i64, i8* nocapture) nounwind

; Function Attrs: nounwind
define void @test_vst4q_lane_bf16(bfloat* nocapture %ptr, [4 x <8 x bfloat>] %val.coerce) local_unnamed_addr nounwind {
; CHECK-LABEL: test_vst4q_lane_bf16:
; CHECK:       // %bb.0: // %entry
; CHECK:    st4 { v0.h, v1.h, v2.h, v3.h }[7], [x0]
; CHECK:    ret
entry:
  %val.coerce.fca.0.extract = extractvalue [4 x <8 x bfloat>] %val.coerce, 0
  %val.coerce.fca.1.extract = extractvalue [4 x <8 x bfloat>] %val.coerce, 1
  %val.coerce.fca.2.extract = extractvalue [4 x <8 x bfloat>] %val.coerce, 2
  %val.coerce.fca.3.extract = extractvalue [4 x <8 x bfloat>] %val.coerce, 3
  %0 = bitcast bfloat* %ptr to i8*
  tail call void @llvm.aarch64.neon.st4lane.v8bf16.p0i8(<8 x bfloat> %val.coerce.fca.0.extract, <8 x bfloat> %val.coerce.fca.1.extract, <8 x bfloat> %val.coerce.fca.2.extract, <8 x bfloat> %val.coerce.fca.3.extract, i64 7, i8* %0)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.aarch64.neon.st4lane.v8bf16.p0i8(<8 x bfloat>, <8 x bfloat>, <8 x bfloat>, <8 x bfloat>, i64, i8* nocapture) nounwind


