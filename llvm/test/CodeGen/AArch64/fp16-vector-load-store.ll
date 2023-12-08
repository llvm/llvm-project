; RUN: llc < %s -mtriple=aarch64 | FileCheck %s

; Simple load of v4i16
define <4 x half> @load_64(ptr nocapture readonly %a) #0 {
; CHECK-LABEL: load_64:
; CHECK: ldr d0, [x0]
entry:
  %0 = load <4 x half>, ptr %a, align 8
  ret <4 x half> %0
}

; Simple load of v8i16
define <8 x half> @load_128(ptr nocapture readonly %a) #0 {
; CHECK-LABEL: load_128:
; CHECK: ldr q0, [x0]
entry:
  %0 = load <8 x half>, ptr %a, align 16
  ret <8 x half> %0
}

; Duplicating load to v4i16
define <4 x half> @load_dup_64(ptr nocapture readonly %a) #0 {
; CHECK-LABEL: load_dup_64:
; CHECK: ld1r { v0.4h }, [x0]
entry:
  %0 = load half, ptr %a, align 2
  %1 = insertelement <4 x half> undef, half %0, i32 0
  %2 = shufflevector <4 x half> %1, <4 x half> undef, <4 x i32> zeroinitializer
  ret <4 x half> %2
}

; Duplicating load to v8i16
define <8 x half> @load_dup_128(ptr nocapture readonly %a) #0 {
; CHECK-LABEL: load_dup_128:
; CHECK: ld1r { v0.8h }, [x0]
entry:
  %0 = load half, ptr %a, align 2
  %1 = insertelement <8 x half> undef, half %0, i32 0
  %2 = shufflevector <8 x half> %1, <8 x half> undef, <8 x i32> zeroinitializer
  ret <8 x half> %2
}

; Load to one lane of v4f16
define <4 x half> @load_lane_64(ptr nocapture readonly %a, <4 x half> %b) #0 {
; CHECK-LABEL: load_lane_64:
; CHECK: ld1 { v0.h }[2], [x0]
entry:
  %0 = load half, ptr %a, align 2
  %1 = insertelement <4 x half> %b, half %0, i32 2
  ret <4 x half> %1
}

; Load to one lane of v8f16
define <8 x half> @load_lane_128(ptr nocapture readonly %a, <8 x half> %b) #0 {
; CHECK-LABEL: load_lane_128:
; CHECK: ld1 { v0.h }[5], [x0]
entry:
  %0 = load half, ptr %a, align 2
  %1 = insertelement <8 x half> %b, half %0, i32 5
  ret <8 x half> %1
}

; Simple store of v4f16
define void @store_64(ptr nocapture %a, <4 x half> %b) #1 {
; CHECK-LABEL: store_64:
; CHECK: str d0, [x0]
entry:
  store <4 x half> %b, ptr %a, align 8
  ret void
}

; Simple store of v8f16
define void @store_128(ptr nocapture %a, <8 x half> %b) #1 {
; CHECK-LABEL: store_128:
; CHECK: str q0, [x0]
entry:
  store <8 x half> %b, ptr %a, align 16
  ret void
}

; Store from one lane of v4f16
define void @store_lane_64(ptr nocapture %a, <4 x half> %b) #1 {
; CHECK-LABEL: store_lane_64:
; CHECK: st1 { v0.h }[2], [x0]
entry:
  %0 = extractelement <4 x half> %b, i32 2
  store half %0, ptr %a, align 2
  ret void
}

define void @store_lane0_64(ptr nocapture %a, <4 x half> %b) #1 {
; CHECK-LABEL: store_lane0_64:
; CHECK: str h0, [x0]
entry:
  %0 = extractelement <4 x half> %b, i32 0
  store half %0, ptr %a, align 2
  ret void
}

define void @storeu_lane0_64(ptr nocapture %a, <4 x half> %b) #1 {
; CHECK-LABEL: storeu_lane0_64:
; CHECK: stur h0, [x{{[0-9]+}}, #-2]
entry:
  %0 = getelementptr half, ptr %a, i64 -1
  %1 = extractelement <4 x half> %b, i32 0
  store half %1, ptr %0, align 2
  ret void
}

define void @storero_lane_64(ptr nocapture %a, <4 x half> %b, i64 %c) #1 {
; CHECK-LABEL: storero_lane_64:
; CHECK: st1 { v0.h }[2], [x{{[0-9]+}}]
entry:
  %0 = getelementptr half, ptr %a, i64 %c
  %1 = extractelement <4 x half> %b, i32 2
  store half %1, ptr %0, align 2
  ret void
}

define void @storero_lane0_64(ptr nocapture %a, <4 x half> %b, i64 %c) #1 {
; CHECK-LABEL: storero_lane0_64:
; CHECK: str h0, [x0, x1, lsl #1]
entry:
  %0 = getelementptr half, ptr %a, i64 %c
  %1 = extractelement <4 x half> %b, i32 0
  store half %1, ptr %0, align 2
  ret void
}

; Store from one lane of v8f16
define void @store_lane_128(ptr nocapture %a, <8 x half> %b) #1 {
; CHECK-LABEL: store_lane_128:
; CHECK: st1 { v0.h }[5], [x0]
entry:
  %0 = extractelement <8 x half> %b, i32 5
  store half %0, ptr %a, align 2
  ret void
}

define void @store_lane0_128(ptr nocapture %a, <8 x half> %b) #1 {
; CHECK-LABEL: store_lane0_128:
; CHECK: str h0, [x0]
entry:
  %0 = extractelement <8 x half> %b, i32 0
  store half %0, ptr %a, align 2
  ret void
}

define void @storeu_lane0_128(ptr nocapture %a, <8 x half> %b) #1 {
; CHECK-LABEL: storeu_lane0_128:
; CHECK: stur h0, [x{{[0-9]+}}, #-2]
entry:
  %0 = getelementptr half, ptr %a, i64 -1
  %1 = extractelement <8 x half> %b, i32 0
  store half %1, ptr %0, align 2
  ret void
}

define void @storero_lane_128(ptr nocapture %a, <8 x half> %b, i64 %c) #1 {
; CHECK-LABEL: storero_lane_128:
; CHECK: st1 { v0.h }[4], [x{{[0-9]+}}]
entry:
  %0 = getelementptr half, ptr %a, i64 %c
  %1 = extractelement <8 x half> %b, i32 4
  store half %1, ptr %0, align 2
  ret void
}

define void @storero_lane0_128(ptr nocapture %a, <8 x half> %b, i64 %c) #1 {
; CHECK-LABEL: storero_lane0_128:
; CHECK: str h0, [x0, x1, lsl #1]
entry:
  %0 = getelementptr half, ptr %a, i64 %c
  %1 = extractelement <8 x half> %b, i32 0
  store half %1, ptr %0, align 2
  ret void
}

; NEON intrinsics - (de-)interleaving loads and stores
declare { <4 x half>, <4 x half> } @llvm.aarch64.neon.ld2.v4f16.p0(ptr)
declare { <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld3.v4f16.p0(ptr)
declare { <4 x half>, <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld4.v4f16.p0(ptr)
declare void @llvm.aarch64.neon.st2.v4f16.p0(<4 x half>, <4 x half>, ptr)
declare void @llvm.aarch64.neon.st3.v4f16.p0(<4 x half>, <4 x half>, <4 x half>, ptr)
declare void @llvm.aarch64.neon.st4.v4f16.p0(<4 x half>, <4 x half>, <4 x half>, <4 x half>, ptr)
declare { <8 x half>, <8 x half> } @llvm.aarch64.neon.ld2.v8f16.p0(ptr)
declare { <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld3.v8f16.p0(ptr)
declare { <8 x half>, <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld4.v8f16.p0(ptr)
declare void @llvm.aarch64.neon.st2.v8f16.p0(<8 x half>, <8 x half>, ptr)
declare void @llvm.aarch64.neon.st3.v8f16.p0(<8 x half>, <8 x half>, <8 x half>, ptr)
declare void @llvm.aarch64.neon.st4.v8f16.p0(<8 x half>, <8 x half>, <8 x half>, <8 x half>, ptr)

; Load 2 x v4f16 with de-interleaving
define { <4 x half>, <4 x half> } @load_interleave_64_2(ptr %a) #0 {
; CHECK-LABEL: load_interleave_64_2:
; CHECK: ld2 { v0.4h, v1.4h }, [x0]
entry:
  %0 = tail call { <4 x half>, <4 x half> } @llvm.aarch64.neon.ld2.v4f16.p0(ptr %a)
  ret { <4 x half>, <4 x half> } %0
}

; Load 3 x v4f16 with de-interleaving
define { <4 x half>, <4 x half>, <4 x half> } @load_interleave_64_3(ptr %a) #0 {
; CHECK-LABEL: load_interleave_64_3:
; CHECK: ld3 { v0.4h, v1.4h, v2.4h }, [x0]
entry:
  %0 = tail call { <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld3.v4f16.p0(ptr %a)
  ret { <4 x half>, <4 x half>, <4 x half> } %0
}

; Load 4 x v4f16 with de-interleaving
define { <4 x half>, <4 x half>, <4 x half>, <4 x half> } @load_interleave_64_4(ptr %a) #0 {
; CHECK-LABEL: load_interleave_64_4:
; CHECK: ld4 { v0.4h, v1.4h, v2.4h, v3.4h }, [x0]
entry:
  %0 = tail call { <4 x half>, <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld4.v4f16.p0(ptr %a)
  ret { <4 x half>, <4 x half>, <4 x half>, <4 x half> } %0
}

; Store 2 x v4f16 with interleaving
define void @store_interleave_64_2(ptr %a, <4 x half> %b, <4 x half> %c) #0 {
; CHECK-LABEL: store_interleave_64_2:
; CHECK: st2 { v0.4h, v1.4h }, [x0]
entry:
  tail call void @llvm.aarch64.neon.st2.v4f16.p0(<4 x half> %b, <4 x half> %c, ptr %a)
  ret void
}

; Store 3 x v4f16 with interleaving
define void @store_interleave_64_3(ptr %a, <4 x half> %b, <4 x half> %c, <4 x half> %d) #0 {
; CHECK-LABEL: store_interleave_64_3:
; CHECK: st3 { v0.4h, v1.4h, v2.4h }, [x0]
entry:
  tail call void @llvm.aarch64.neon.st3.v4f16.p0(<4 x half> %b, <4 x half> %c, <4 x half> %d, ptr %a)
  ret void
}

; Store 4 x v4f16 with interleaving
define void @store_interleave_64_4(ptr %a, <4 x half> %b, <4 x half> %c, <4 x half> %d, <4 x half> %e) #0 {
; CHECK-LABEL: store_interleave_64_4:
; CHECK: st4 { v0.4h, v1.4h, v2.4h, v3.4h }, [x0]
entry:
  tail call void @llvm.aarch64.neon.st4.v4f16.p0(<4 x half> %b, <4 x half> %c, <4 x half> %d, <4 x half> %e, ptr %a)
  ret void
}

; Load 2 x v8f16 with de-interleaving
define { <8 x half>, <8 x half> } @load_interleave_128_2(ptr %a) #0 {
; CHECK-LABEL: load_interleave_128_2:
; CHECK: ld2 { v0.8h, v1.8h }, [x0]
entry:
  %0 = tail call { <8 x half>, <8 x half> } @llvm.aarch64.neon.ld2.v8f16.p0(ptr %a)
  ret { <8 x half>, <8 x half> } %0
}

; Load 3 x v8f16 with de-interleaving
define { <8 x half>, <8 x half>, <8 x half> } @load_interleave_128_3(ptr %a) #0 {
; CHECK-LABEL: load_interleave_128_3:
; CHECK: ld3 { v0.8h, v1.8h, v2.8h }, [x0]
entry:
  %0 = tail call { <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld3.v8f16.p0(ptr %a)
  ret { <8 x half>, <8 x half>, <8 x half> } %0
}

; Load 8 x v8f16 with de-interleaving
define { <8 x half>, <8 x half>, <8 x half>, <8 x half> } @load_interleave_128_4(ptr %a) #0 {
; CHECK-LABEL: load_interleave_128_4:
; CHECK: ld4 { v0.8h, v1.8h, v2.8h, v3.8h }, [x0]
entry:
  %0 = tail call { <8 x half>, <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld4.v8f16.p0(ptr %a)
  ret { <8 x half>, <8 x half>, <8 x half>, <8 x half> } %0
}

; Store 2 x v8f16 with interleaving
define void @store_interleave_128_2(ptr %a, <8 x half> %b, <8 x half> %c) #0 {
; CHECK-LABEL: store_interleave_128_2:
; CHECK: st2 { v0.8h, v1.8h }, [x0]
entry:
  tail call void @llvm.aarch64.neon.st2.v8f16.p0(<8 x half> %b, <8 x half> %c, ptr %a)
  ret void
}

; Store 3 x v8f16 with interleaving
define void @store_interleave_128_3(ptr %a, <8 x half> %b, <8 x half> %c, <8 x half> %d) #0 {
; CHECK-LABEL: store_interleave_128_3:
; CHECK: st3 { v0.8h, v1.8h, v2.8h }, [x0]
entry:
  tail call void @llvm.aarch64.neon.st3.v8f16.p0(<8 x half> %b, <8 x half> %c, <8 x half> %d, ptr %a)
  ret void
}

; Store 8 x v8f16 with interleaving
define void @store_interleave_128_4(ptr %a, <8 x half> %b, <8 x half> %c, <8 x half> %d, <8 x half> %e) #0 {
; CHECK-LABEL: store_interleave_128_4:
; CHECK: st4 { v0.8h, v1.8h, v2.8h, v3.8h }, [x0]
entry:
  tail call void @llvm.aarch64.neon.st4.v8f16.p0(<8 x half> %b, <8 x half> %c, <8 x half> %d, <8 x half> %e, ptr %a)
  ret void
}

; NEON intrinsics - duplicating loads
declare { <4 x half>, <4 x half> } @llvm.aarch64.neon.ld2r.v4f16.p0(ptr)
declare { <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld3r.v4f16.p0(ptr)
declare { <4 x half>, <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld4r.v4f16.p0(ptr)
declare { <8 x half>, <8 x half> } @llvm.aarch64.neon.ld2r.v8f16.p0(ptr)
declare { <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld3r.v8f16.p0(ptr)
declare { <8 x half>, <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld4r.v8f16.p0(ptr)

; Load 2 x v4f16 with duplication
define { <4 x half>, <4 x half> } @load_dup_64_2(ptr %a) #0 {
; CHECK-LABEL: load_dup_64_2:
; CHECK: ld2r { v0.4h, v1.4h }, [x0]
entry:
  %0 = tail call { <4 x half>, <4 x half> } @llvm.aarch64.neon.ld2r.v4f16.p0(ptr %a)
  ret { <4 x half>, <4 x half> } %0
}

; Load 3 x v4f16 with duplication
define { <4 x half>, <4 x half>, <4 x half> } @load_dup_64_3(ptr %a) #0 {
; CHECK-LABEL: load_dup_64_3:
; CHECK: ld3r { v0.4h, v1.4h, v2.4h }, [x0]
entry:
  %0 = tail call { <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld3r.v4f16.p0(ptr %a)
  ret { <4 x half>, <4 x half>, <4 x half> } %0
}

; Load 4 x v4f16 with duplication
define { <4 x half>, <4 x half>, <4 x half>, <4 x half> } @load_dup_64_4(ptr %a) #0 {
; CHECK-LABEL: load_dup_64_4:
; CHECK: ld4r { v0.4h, v1.4h, v2.4h, v3.4h }, [x0]
entry:
  %0 = tail call { <4 x half>, <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld4r.v4f16.p0(ptr %a)
  ret { <4 x half>, <4 x half>, <4 x half>, <4 x half> } %0
}

; Load 2 x v8f16 with duplication
define { <8 x half>, <8 x half> } @load_dup_128_2(ptr %a) #0 {
; CHECK-LABEL: load_dup_128_2:
; CHECK: ld2r { v0.8h, v1.8h }, [x0]
entry:
  %0 = tail call { <8 x half>, <8 x half> } @llvm.aarch64.neon.ld2r.v8f16.p0(ptr %a)
  ret { <8 x half>, <8 x half> } %0
}

; Load 3 x v8f16 with duplication
define { <8 x half>, <8 x half>, <8 x half> } @load_dup_128_3(ptr %a) #0 {
; CHECK-LABEL: load_dup_128_3:
; CHECK: ld3r { v0.8h, v1.8h, v2.8h }, [x0]
entry:
  %0 = tail call { <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld3r.v8f16.p0(ptr %a)
  ret { <8 x half>, <8 x half>, <8 x half> } %0
}

; Load 8 x v8f16 with duplication
define { <8 x half>, <8 x half>, <8 x half>, <8 x half> } @load_dup_128_4(ptr %a) #0 {
; CHECK-LABEL: load_dup_128_4:
; CHECK: ld4r { v0.8h, v1.8h, v2.8h, v3.8h }, [x0]
entry:
  %0 = tail call { <8 x half>, <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld4r.v8f16.p0(ptr %a)
  ret { <8 x half>, <8 x half>, <8 x half>, <8 x half> } %0
}


; NEON intrinsics - loads and stores to/from one lane
declare { <4 x half>, <4 x half> } @llvm.aarch64.neon.ld2lane.v4f16.p0(<4 x half>, <4 x half>, i64, ptr)
declare { <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld3lane.v4f16.p0(<4 x half>, <4 x half>, <4 x half>, i64, ptr)
declare { <4 x half>, <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld4lane.v4f16.p0(<4 x half>, <4 x half>, <4 x half>, <4 x half>, i64, ptr)
declare void @llvm.aarch64.neon.st2lane.v4f16.p0(<4 x half>, <4 x half>, i64, ptr)
declare void @llvm.aarch64.neon.st3lane.v4f16.p0(<4 x half>, <4 x half>, <4 x half>, i64, ptr)
declare void @llvm.aarch64.neon.st4lane.v4f16.p0(<4 x half>, <4 x half>, <4 x half>, <4 x half>, i64, ptr)
declare { <8 x half>, <8 x half> } @llvm.aarch64.neon.ld2lane.v8f16.p0(<8 x half>, <8 x half>, i64, ptr)
declare { <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld3lane.v8f16.p0(<8 x half>, <8 x half>, <8 x half>, i64, ptr)
declare { <8 x half>, <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld4lane.v8f16.p0(<8 x half>, <8 x half>, <8 x half>, <8 x half>, i64, ptr)
declare void @llvm.aarch64.neon.st2lane.v8f16.p0(<8 x half>, <8 x half>, i64, ptr)
declare void @llvm.aarch64.neon.st3lane.v8f16.p0(<8 x half>, <8 x half>, <8 x half>, i64, ptr)
declare void @llvm.aarch64.neon.st4lane.v8f16.p0(<8 x half>, <8 x half>, <8 x half>, <8 x half>, i64, ptr)

; Load one lane of 2 x v4f16
define { <4 x half>, <4 x half> } @load_lane_64_2(ptr %a, <4 x half> %b, <4 x half> %c) #0 {
; CHECK-LABEL: load_lane_64_2:
; CHECK: ld2 { v0.h, v1.h }[2], [x0]
entry:
  %0 = tail call { <4 x half>, <4 x half> } @llvm.aarch64.neon.ld2lane.v4f16.p0(<4 x half> %b, <4 x half> %c, i64 2, ptr %a)
  ret { <4 x half>, <4 x half> } %0
}

; Load one lane of 3 x v4f16
define { <4 x half>, <4 x half>, <4 x half> } @load_lane_64_3(ptr %a, <4 x half> %b, <4 x half> %c, <4 x half> %d) #0 {
; CHECK-LABEL: load_lane_64_3:
; CHECK: ld3 { v0.h, v1.h, v2.h }[2], [x0]
entry:
  %0 = tail call { <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld3lane.v4f16.p0(<4 x half> %b, <4 x half> %c, <4 x half> %d, i64 2, ptr %a)
  ret { <4 x half>, <4 x half>, <4 x half> } %0
}

; Load one lane of 4 x v4f16
define { <4 x half>, <4 x half>, <4 x half>, <4 x half> } @load_lane_64_4(ptr %a, <4 x half> %b, <4 x half> %c, <4 x half> %d, <4 x half> %e) #0 {
; CHECK-LABEL: load_lane_64_4:
; CHECK: ld4 { v0.h, v1.h, v2.h, v3.h }[2], [x0]
entry:
  %0 = tail call { <4 x half>, <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld4lane.v4f16.p0(<4 x half> %b, <4 x half> %c, <4 x half> %d, <4 x half> %e, i64 2, ptr %a)
  ret { <4 x half>, <4 x half>, <4 x half>, <4 x half> } %0
}

; Store one lane of 2 x v4f16
define void @store_lane_64_2(ptr %a, <4 x half> %b, <4 x half> %c) #0 {
; CHECK-LABEL: store_lane_64_2:
; CHECK: st2 { v0.h, v1.h }[2], [x0]
entry:
  tail call void @llvm.aarch64.neon.st2lane.v4f16.p0(<4 x half> %b, <4 x half> %c, i64 2, ptr %a)
  ret void
}

; Store one lane of 3 x v4f16
define void @store_lane_64_3(ptr %a, <4 x half> %b, <4 x half> %c, <4 x half> %d) #0 {
; CHECK-LABEL: store_lane_64_3:
; CHECK: st3 { v0.h, v1.h, v2.h }[2], [x0]
entry:
  tail call void @llvm.aarch64.neon.st3lane.v4f16.p0(<4 x half> %b, <4 x half> %c, <4 x half> %d, i64 2, ptr %a)
  ret void
}

; Store one lane of 4 x v4f16
define void @store_lane_64_4(ptr %a, <4 x half> %b, <4 x half> %c, <4 x half> %d, <4 x half> %e) #0 {
; CHECK-LABEL: store_lane_64_4:
; CHECK: st4 { v0.h, v1.h, v2.h, v3.h }[2], [x0]
entry:
  tail call void @llvm.aarch64.neon.st4lane.v4f16.p0(<4 x half> %b, <4 x half> %c, <4 x half> %d, <4 x half> %e, i64 2, ptr %a)
  ret void
}

; Load one lane of 2 x v8f16
define { <8 x half>, <8 x half> } @load_lane_128_2(ptr %a, <8 x half> %b, <8 x half> %c) #0 {
; CHECK-LABEL: load_lane_128_2:
; CHECK: ld2 { v0.h, v1.h }[2], [x0]
entry:
  %0 = tail call { <8 x half>, <8 x half> } @llvm.aarch64.neon.ld2lane.v8f16.p0(<8 x half> %b, <8 x half> %c, i64 2, ptr %a)
  ret { <8 x half>, <8 x half> } %0
}

; Load one lane of 3 x v8f16
define { <8 x half>, <8 x half>, <8 x half> } @load_lane_128_3(ptr %a, <8 x half> %b, <8 x half> %c, <8 x half> %d) #0 {
; CHECK-LABEL: load_lane_128_3:
; CHECK: ld3 { v0.h, v1.h, v2.h }[2], [x0]
entry:
  %0 = tail call { <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld3lane.v8f16.p0(<8 x half> %b, <8 x half> %c, <8 x half> %d, i64 2, ptr %a)
  ret { <8 x half>, <8 x half>, <8 x half> } %0
}

; Load one lane of 8 x v8f16
define { <8 x half>, <8 x half>, <8 x half>, <8 x half> } @load_lane_128_4(ptr %a, <8 x half> %b, <8 x half> %c, <8 x half> %d, <8 x half> %e) #0 {
; CHECK-LABEL: load_lane_128_4:
; CHECK: ld4 { v0.h, v1.h, v2.h, v3.h }[2], [x0]
entry:
  %0 = tail call { <8 x half>, <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld4lane.v8f16.p0(<8 x half> %b, <8 x half> %c, <8 x half> %d, <8 x half> %e, i64 2, ptr %a)
  ret { <8 x half>, <8 x half>, <8 x half>, <8 x half> } %0
}

; Store one lane of 2 x v8f16
define void @store_lane_128_2(ptr %a, <8 x half> %b, <8 x half> %c) #0 {
; CHECK-LABEL: store_lane_128_2:
; CHECK: st2 { v0.h, v1.h }[2], [x0]
entry:
  tail call void @llvm.aarch64.neon.st2lane.v8f16.p0(<8 x half> %b, <8 x half> %c, i64 2, ptr %a)
  ret void
}

; Store one lane of 3 x v8f16
define void @store_lane_128_3(ptr %a, <8 x half> %b, <8 x half> %c, <8 x half> %d) #0 {
; CHECK-LABEL: store_lane_128_3:
; CHECK: st3 { v0.h, v1.h, v2.h }[2], [x0]
entry:
  tail call void @llvm.aarch64.neon.st3lane.v8f16.p0(<8 x half> %b, <8 x half> %c, <8 x half> %d, i64 2, ptr %a)
  ret void
}

; Store one lane of 8 x v8f16
define void @store_lane_128_4(ptr %a, <8 x half> %b, <8 x half> %c, <8 x half> %d, <8 x half> %e) #0 {
; CHECK-LABEL: store_lane_128_4:
; CHECK: st4 { v0.h, v1.h, v2.h, v3.h }[2], [x0]
entry:
  tail call void @llvm.aarch64.neon.st4lane.v8f16.p0(<8 x half> %b, <8 x half> %c, <8 x half> %d, <8 x half> %e, i64 2, ptr %a)
  ret void
}

; NEON intrinsics - load/store without interleaving
declare { <4 x half>, <4 x half> } @llvm.aarch64.neon.ld1x2.v4f16.p0(ptr)
declare { <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld1x3.v4f16.p0(ptr)
declare { <4 x half>, <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld1x4.v4f16.p0(ptr)
declare void @llvm.aarch64.neon.st1x2.v4f16.p0(<4 x half>, <4 x half>, ptr)
declare void @llvm.aarch64.neon.st1x3.v4f16.p0(<4 x half>, <4 x half>, <4 x half>, ptr)
declare void @llvm.aarch64.neon.st1x4.v4f16.p0(<4 x half>, <4 x half>, <4 x half>, <4 x half>, ptr)
declare { <8 x half>, <8 x half> } @llvm.aarch64.neon.ld1x2.v8f16.p0(ptr)
declare { <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld1x3.v8f16.p0(ptr)
declare { <8 x half>, <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld1x4.v8f16.p0(ptr)
declare void @llvm.aarch64.neon.st1x2.v8f16.p0(<8 x half>, <8 x half>, ptr)
declare void @llvm.aarch64.neon.st1x3.v8f16.p0(<8 x half>, <8 x half>, <8 x half>, ptr)
declare void @llvm.aarch64.neon.st1x4.v8f16.p0(<8 x half>, <8 x half>, <8 x half>, <8 x half>, ptr)

; Load 2 x v4f16 without de-interleaving
define { <4 x half>, <4 x half> } @load_64_2(ptr %a) #0 {
; CHECK-LABEL: load_64_2:
; CHECK: ld1 { v0.4h, v1.4h }, [x0]
entry:
  %0 = tail call { <4 x half>, <4 x half> } @llvm.aarch64.neon.ld1x2.v4f16.p0(ptr %a)
  ret { <4 x half>, <4 x half> } %0
}

; Load 3 x v4f16 without de-interleaving
define { <4 x half>, <4 x half>, <4 x half> } @load_64_3(ptr %a) #0 {
; CHECK-LABEL: load_64_3:
; CHECK: ld1 { v0.4h, v1.4h, v2.4h }, [x0]
entry:
  %0 = tail call { <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld1x3.v4f16.p0(ptr %a)
  ret { <4 x half>, <4 x half>, <4 x half> } %0
}

; Load 4 x v4f16 without de-interleaving
define { <4 x half>, <4 x half>, <4 x half>, <4 x half> } @load_64_4(ptr %a) #0 {
; CHECK-LABEL: load_64_4:
; CHECK: ld1 { v0.4h, v1.4h, v2.4h, v3.4h }, [x0]
entry:
  %0 = tail call { <4 x half>, <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld1x4.v4f16.p0(ptr %a)
  ret { <4 x half>, <4 x half>, <4 x half>, <4 x half> } %0
}

; Store 2 x v4f16 without interleaving
define void @store_64_2(ptr %a, <4 x half> %b, <4 x half> %c) #0 {
; CHECK-LABEL: store_64_2:
; CHECK: st1 { v0.4h, v1.4h }, [x0]
entry:
  tail call void @llvm.aarch64.neon.st1x2.v4f16.p0(<4 x half> %b, <4 x half> %c, ptr %a)
  ret void
}

; Store 3 x v4f16 without interleaving
define void @store_64_3(ptr %a, <4 x half> %b, <4 x half> %c, <4 x half> %d) #0 {
; CHECK-LABEL: store_64_3:
; CHECK: st1 { v0.4h, v1.4h, v2.4h }, [x0]
entry:
  tail call void @llvm.aarch64.neon.st1x3.v4f16.p0(<4 x half> %b, <4 x half> %c, <4 x half> %d, ptr %a)
  ret void
}

; Store 4 x v4f16 without interleaving
define void @store_64_4(ptr %a, <4 x half> %b, <4 x half> %c, <4 x half> %d, <4 x half> %e) #0 {
; CHECK-LABEL: store_64_4:
; CHECK: st1 { v0.4h, v1.4h, v2.4h, v3.4h }, [x0]
entry:
  tail call void @llvm.aarch64.neon.st1x4.v4f16.p0(<4 x half> %b, <4 x half> %c, <4 x half> %d, <4 x half> %e, ptr %a)
  ret void
}

; Load 2 x v8f16 without de-interleaving
define { <8 x half>, <8 x half> } @load_128_2(ptr %a) #0 {
; CHECK-LABEL: load_128_2:
; CHECK: ld1 { v0.8h, v1.8h }, [x0]
entry:
  %0 = tail call { <8 x half>, <8 x half> } @llvm.aarch64.neon.ld1x2.v8f16.p0(ptr %a)
  ret { <8 x half>, <8 x half> } %0
}

; Load 3 x v8f16 without de-interleaving
define { <8 x half>, <8 x half>, <8 x half> } @load_128_3(ptr %a) #0 {
; CHECK-LABEL: load_128_3:
; CHECK: ld1 { v0.8h, v1.8h, v2.8h }, [x0]
entry:
  %0 = tail call { <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld1x3.v8f16.p0(ptr %a)
  ret { <8 x half>, <8 x half>, <8 x half> } %0
}

; Load 8 x v8f16 without de-interleaving
define { <8 x half>, <8 x half>, <8 x half>, <8 x half> } @load_128_4(ptr %a) #0 {
; CHECK-LABEL: load_128_4:
; CHECK: ld1 { v0.8h, v1.8h, v2.8h, v3.8h }, [x0]
entry:
  %0 = tail call { <8 x half>, <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld1x4.v8f16.p0(ptr %a)
  ret { <8 x half>, <8 x half>, <8 x half>, <8 x half> } %0
}

; Store 2 x v8f16 without interleaving
define void @store_128_2(ptr %a, <8 x half> %b, <8 x half> %c) #0 {
; CHECK-LABEL: store_128_2:
; CHECK: st1 { v0.8h, v1.8h }, [x0]
entry:
  tail call void @llvm.aarch64.neon.st1x2.v8f16.p0(<8 x half> %b, <8 x half> %c, ptr %a)
  ret void
}

; Store 3 x v8f16 without interleaving
define void @store_128_3(ptr %a, <8 x half> %b, <8 x half> %c, <8 x half> %d) #0 {
; CHECK-LABEL: store_128_3:
; CHECK: st1 { v0.8h, v1.8h, v2.8h }, [x0]
entry:
  tail call void @llvm.aarch64.neon.st1x3.v8f16.p0(<8 x half> %b, <8 x half> %c, <8 x half> %d, ptr %a)
  ret void
}

; Store 8 x v8f16 without interleaving
define void @store_128_4(ptr %a, <8 x half> %b, <8 x half> %c, <8 x half> %d, <8 x half> %e) #0 {
; CHECK-LABEL: store_128_4:
; CHECK: st1 { v0.8h, v1.8h, v2.8h, v3.8h }, [x0]
entry:
  tail call void @llvm.aarch64.neon.st1x4.v8f16.p0(<8 x half> %b, <8 x half> %c, <8 x half> %d, <8 x half> %e, ptr %a)
  ret void
}
