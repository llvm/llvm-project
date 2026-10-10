; RUN: opt -passes=vector-combine -vector-combine-guard-masked-regions -verify-each -S %s | FileCheck %s
; RUN: opt -passes='vector-combine,vector-combine' -vector-combine-guard-masked-regions -verify-each -S %s | FileCheck %s
; RUN: opt -passes='vector-combine,instcombine,simplifycfg,vector-combine' -vector-combine-guard-masked-regions -verify-each -S %s | FileCheck %s --check-prefix=CANON
; RUN: opt -passes=vector-combine -S %s | FileCheck %s --check-prefix=OFF
; RUN: opt -passes='debugify,function(vector-combine),check-debugify' -vector-combine-guard-masked-regions -disable-output %s 2>&1 | FileCheck %s --check-prefix=DEBUG

; DEBUG-NOT: WARNING
; DEBUG: CheckModuleDebugify: PASS
; OFF-NOT: br i1
; CANON-LABEL: define void @interleaved(
; CANON: br i1
; CANON-NOT: br i1
; CANON: ret void

target triple = "x86_64-unknown-linux-gnu"

declare <8 x float> @llvm.masked.load.v8f32.p0(ptr, <8 x i1>, <8 x float>)
declare void @llvm.masked.store.v8f32.p0(<8 x float>, ptr, <8 x i1>)
declare <4 x float> @llvm.masked.load.v4f32.p0(ptr, <4 x i1>, <4 x float>)
declare void @llvm.masked.store.v4f32.p0(<4 x float>, ptr, <4 x i1>)
declare void @side_effect()

; Guard both interleaved load/store pairs together, skipping them only when
; both masks are inactive. Each access retains its own frozen mask.
; CHECK-LABEL: define void @interleaved(
; CHECK: [[A:%.*]] = freeze <8 x i1> %a
; CHECK: [[B:%.*]] = freeze <8 x i1> %b
; CHECK: [[AB:%.*]] = or <8 x i1> [[A]], [[B]]
; CHECK: [[ANY:%.*]] = call i1 @llvm.vector.reduce.or.v8i1(<8 x i1> [[AB]])
; CHECK: br i1 [[ANY]], label %[[BODY:.*]], label %[[EXIT:.*]]
; CHECK: [[BODY]]:
; CHECK-NOT: br i1
; CHECK: [[VA:%.*]] = call <8 x float> @llvm.masked.load.v8f32.p0(ptr align 4 %srca, <8 x i1> [[A]], <8 x float> zeroinitializer)
; CHECK: [[VB:%.*]] = call <8 x float> @llvm.masked.load.v8f32.p0(ptr align 4 %srcb, <8 x i1> [[B]], <8 x float> zeroinitializer)
; CHECK: call void @llvm.masked.store.v8f32.p0(<8 x float> [[VA]], ptr align 4 %dsta, <8 x i1> [[A]])
; CHECK: call void @llvm.masked.store.v8f32.p0(<8 x float> [[VB]], ptr align 4 %dstb, <8 x i1> [[B]])
; CHECK: br label %[[EXIT]]
; CHECK: [[EXIT]]:
; CHECK: ret void
define void @interleaved(ptr %srca, ptr %srcb, ptr %dsta, ptr %dstb, <8 x i1> %a, <8 x i1> %b) #0 {
  %va = call <8 x float> @llvm.masked.load.v8f32.p0(ptr align 4 %srca, <8 x i1> %a, <8 x float> zeroinitializer)
  %vb = call <8 x float> @llvm.masked.load.v8f32.p0(ptr align 4 %srcb, <8 x i1> %b, <8 x float> zeroinitializer)
  call void @llvm.masked.store.v8f32.p0(<8 x float> %va, ptr align 4 %dsta, <8 x i1> %a)
  call void @llvm.masked.store.v8f32.p0(<8 x float> %vb, ptr align 4 %dstb, <8 x i1> %b)
  ret void
}

; Reject a mask computed from a load inside the region: it is not available
; before the region to form the combined guard.
; CHECK-LABEL: define void @mask_inside_region(
; CHECK-NOT: br i1
; CHECK: ret void
define void @mask_inside_region(ptr %src, ptr %dst, <8 x i1> %a) #0 {
  %v = call <8 x float> @llvm.masked.load.v8f32.p0(ptr align 4 %src, <8 x i1> %a, <8 x float> zeroinitializer)
  %b = fcmp ogt <8 x float> %v, zeroinitializer
  call void @llvm.masked.store.v8f32.p0(<8 x float> %v, ptr align 4 %dst, <8 x i1> %b)
  ret void
}

; Reject a loaded value used by an unmasked store after the region. Skipping
; the region would leave that external use without its value.
; CHECK-LABEL: define void @escaping_multimask(
; CHECK-NOT: br i1
; CHECK: ret void
define void @escaping_multimask(ptr %srca, ptr %srcb, ptr %dst, ptr %other, <8 x i1> %a, <8 x i1> %b) #0 {
  %va = call <8 x float> @llvm.masked.load.v8f32.p0(ptr align 4 %srca, <8 x i1> %a, <8 x float> zeroinitializer)
  %vb = call <8 x float> @llvm.masked.load.v8f32.p0(ptr align 4 %srcb, <8 x i1> %b, <8 x float> zeroinitializer)
  call void @llvm.masked.store.v8f32.p0(<8 x float> %vb, ptr align 4 %dst, <8 x i1> %b)
  store <8 x float> %va, ptr %other
  ret void
}

; Reject an intervening call whose effects must occur even when both masks
; are inactive.
; CHECK-LABEL: define void @multimask_side_effect(
; CHECK-NOT: br i1
; CHECK: ret void
define void @multimask_side_effect(ptr %src, ptr %dst, <8 x i1> %a, <8 x i1> %b) #0 {
  %v = call <8 x float> @llvm.masked.load.v8f32.p0(ptr align 4 %src, <8 x i1> %a, <8 x float> zeroinitializer)
  call void @side_effect()
  call void @llvm.masked.store.v8f32.p0(<8 x float> %v, ptr align 4 %dst, <8 x i1> %b)
  ret void
}

; Reject masks with different lane counts: the combined-mask construction
; requires matching fixed-vector types.
; CHECK-LABEL: define void @different_widths(
; CHECK-NOT: br i1
; CHECK: ret void
define void @different_widths(ptr %srca, ptr %srcb, ptr %dsta, ptr %dstb, <8 x i1> %a, <4 x i1> %b) #0 {
  %va = call <8 x float> @llvm.masked.load.v8f32.p0(ptr align 4 %srca, <8 x i1> %a, <8 x float> zeroinitializer)
  %vb = call <4 x float> @llvm.masked.load.v4f32.p0(ptr align 4 %srcb, <4 x i1> %b, <4 x float> zeroinitializer)
  call void @llvm.masked.store.v8f32.p0(<8 x float> %va, ptr align 4 %dsta, <8 x i1> %a)
  call void @llvm.masked.store.v4f32.p0(<4 x float> %vb, ptr align 4 %dstb, <4 x i1> %b)
  ret void
}

; Combine all four store masks into one guard while preserving individual
; masks. Recognize the guard after simplification to avoid adding another.
; CHECK-LABEL: define void @four_masks(
; CHECK: [[A4:%.*]] = freeze <8 x i1> %a
; CHECK: [[B4:%.*]] = freeze <8 x i1> %b
; CHECK: [[AB4:%.*]] = or <8 x i1> [[A4]], [[B4]]
; CHECK: [[C4:%.*]] = freeze <8 x i1> %c
; CHECK: [[ABC4:%.*]] = or <8 x i1> [[AB4]], [[C4]]
; CHECK: [[D4:%.*]] = freeze <8 x i1> %d
; CHECK: [[ABCD4:%.*]] = or <8 x i1> [[ABC4]], [[D4]]
; CHECK: [[ANY4:%.*]] = call i1 @llvm.vector.reduce.or.v8i1(<8 x i1> [[ABCD4]])
; CHECK: br i1 [[ANY4]], label %[[BODY4:.*]], label %[[EXIT4:.*]]
; CHECK: [[BODY4]]:
; CHECK-NOT: br i1
; CHECK: call void @llvm.masked.store.v8f32.p0(<8 x float> %v, ptr align 4 %p, <8 x i1> [[A4]])
; CHECK: call void @llvm.masked.store.v8f32.p0(<8 x float> %v, ptr align 4 %q, <8 x i1> [[B4]])
; CHECK: call void @llvm.masked.store.v8f32.p0(<8 x float> %v, ptr align 4 %r, <8 x i1> [[C4]])
; CHECK: call void @llvm.masked.store.v8f32.p0(<8 x float> %v, ptr align 4 %s, <8 x i1> [[D4]])
; CHECK: ret void
; CANON-LABEL: define void @four_masks(
; CANON: br i1
; CANON-NOT: br i1
; CANON: ret void
define void @four_masks(ptr %p, ptr %q, ptr %r, ptr %s, <8 x float> %v, <8 x i1> %a, <8 x i1> %b, <8 x i1> %c, <8 x i1> %d) #0 {
  call void @llvm.masked.store.v8f32.p0(<8 x float> %v, ptr align 4 %p, <8 x i1> %a)
  call void @llvm.masked.store.v8f32.p0(<8 x float> %v, ptr align 4 %q, <8 x i1> %b)
  call void @llvm.masked.store.v8f32.p0(<8 x float> %v, ptr align 4 %r, <8 x i1> %c)
  call void @llvm.masked.store.v8f32.p0(<8 x float> %v, ptr align 4 %s, <8 x i1> %d)
  ret void
}

attributes #0 = { "target-features"="+avx2" }