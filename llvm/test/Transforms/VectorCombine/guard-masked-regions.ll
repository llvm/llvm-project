; RUN: opt -passes=vector-combine -vector-combine-guard-masked-regions -verify-each -S %s | FileCheck %s
; RUN: opt -passes='vector-combine,vector-combine' -vector-combine-guard-masked-regions -verify-each -S %s | FileCheck %s
; RUN: opt -passes=vector-combine -S %s | FileCheck %s --check-prefix=OFF
; RUN: opt -passes=vector-combine -disable-vector-combine -vector-combine-guard-masked-regions -S %s | FileCheck %s --check-prefix=OFF
; RUN: opt -passes='vector-combine,instcombine,simplifycfg,vector-combine' -vector-combine-guard-masked-regions -verify-each -S %s | FileCheck %s --check-prefix=CANON
; RUN: opt -passes='debugify,function(vector-combine),check-debugify' -vector-combine-guard-masked-regions -disable-output %s 2>&1 | FileCheck %s --check-prefix=DEBUG

; DEBUG-NOT: WARNING
; DEBUG: CheckModuleDebugify: PASS
; CANON-LABEL: define void @guard(
; CANON: br i1
; CANON-NOT: br i1
; CANON: ret void

target triple = "x86_64-unknown-linux-gnu"

declare <8 x float> @llvm.masked.load.v8f32.p0(ptr, i32 immarg, <8 x i1>, <8 x float>)
declare void @llvm.masked.store.v8f32.p0(<8 x float>, ptr, i32 immarg, <8 x i1>)
declare void @side_effect()
declare <8 x float> @collective(<8 x float>) convergent nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.entry()

; Skip the load, arithmetic and store when all lanes are inactive. Freeze the
; mask so the guard and memory operations agree even for undefined lanes.
; OFF-NOT: br i1
; CHECK-LABEL: define void @guard(
; CHECK: [[MASK:%.*]] = freeze <8 x i1> %mask
; CHECK-NEXT: [[ANY:%.*]] = call i1 @llvm.vector.reduce.or.v8i1(<8 x i1> [[MASK]])
; CHECK-NEXT: br i1 [[ANY]], label %[[BODY:.*]], label %[[EXIT:.*]]
; CHECK: [[BODY]]:
; CHECK-NOT: br i1
; CHECK: [[LOAD:%.*]] = call <8 x float> @llvm.masked.load.v8f32.p0(ptr align 4 %src, <8 x i1> [[MASK]], <8 x float> poison)
; CHECK: [[SUM:%.*]] = fadd <8 x float> [[LOAD]], [[LOAD]]
; CHECK: call void @llvm.masked.store.v8f32.p0(<8 x float> [[SUM]], ptr align 4 %dst, <8 x i1> [[MASK]])
; CHECK-NEXT: br label %[[EXIT]]
; CHECK: [[EXIT]]:
; CHECK-NEXT: ret void
define void @guard(ptr %src, ptr %dst, <8 x i1> %mask) #0 {
  %v = call <8 x float> @llvm.masked.load.v8f32.p0(ptr %src, i32 4, <8 x i1> %mask, <8 x float> poison)
  %sum = fadd <8 x float> %v, %v
  call void @llvm.masked.store.v8f32.p0(<8 x float> %sum, ptr %dst, i32 4, <8 x i1> %mask)
  ret void
}

; Reject a region whose loaded value escapes through the return: skipping the
; region would require providing a value for that external use.
; CHECK-LABEL: define <8 x float> @escaping_value(
; CHECK-NOT: br i1
; CHECK: ret <8 x float>
define <8 x float> @escaping_value(ptr %src, ptr %dst, <8 x i1> %mask) #0 {
  %v = call <8 x float> @llvm.masked.load.v8f32.p0(ptr %src, i32 4, <8 x i1> %mask, <8 x float> poison)
  call void @llvm.masked.store.v8f32.p0(<8 x float> %v, ptr %dst, i32 4, <8 x i1> %mask)
  ret <8 x float> %v
}

; Skip only when both masks are inactive. Combine frozen masks for the guard,
; but retain each operation's own mask rather than enabling extra accesses.
; CHECK-LABEL: define void @different_masks(
; CHECK: [[A:%.*]] = freeze <8 x i1> %a
; CHECK: [[B:%.*]] = freeze <8 x i1> %b
; CHECK: [[AB:%.*]] = or <8 x i1> [[A]], [[B]]
; CHECK: [[ANYAB:%.*]] = call i1 @llvm.vector.reduce.or.v8i1(<8 x i1> [[AB]])
; CHECK: br i1 [[ANYAB]], label %[[BODYAB:.*]], label %[[EXITAB:.*]]
; CHECK: [[BODYAB]]:
; CHECK-NOT: br i1
; CHECK: [[V:%.*]] = call <8 x float> @llvm.masked.load.v8f32.p0(ptr align 4 %src, <8 x i1> [[A]], <8 x float> poison)
; CHECK: call void @llvm.masked.store.v8f32.p0(<8 x float> [[V]], ptr align 4 %dst, <8 x i1> [[B]])
; CHECK: ret void
define void @different_masks(ptr %src, ptr %dst, <8 x i1> %a, <8 x i1> %b) #0 {
  %v = call <8 x float> @llvm.masked.load.v8f32.p0(ptr %src, i32 4, <8 x i1> %a, <8 x float> poison)
  call void @llvm.masked.store.v8f32.p0(<8 x float> %v, ptr %dst, i32 4, <8 x i1> %b)
  ret void
}

; Reject an intervening call that may have observable effects even when all
; mask lanes are inactive.
; CHECK-LABEL: define void @side_effects(
; CHECK-NOT: br i1
; CHECK: ret void
define void @side_effects(ptr %src, ptr %dst, <8 x i1> %mask) #0 {
  %v = call <8 x float> @llvm.masked.load.v8f32.p0(ptr %src, i32 4, <8 x i1> %mask, <8 x float> poison)
  call void @side_effect()
  call void @llvm.masked.store.v8f32.p0(<8 x float> %v, ptr %dst, i32 4, <8 x i1> %mask)
  ret void
}

; Reject the region because the unmasked store must execute independently of
; the vector mask.
; CHECK-LABEL: define void @unmasked_store(
; CHECK-NOT: br i1
; CHECK: ret void
define void @unmasked_store(ptr %src, ptr %dst, ptr %other, <8 x i1> %mask) #0 {
  %v = call <8 x float> @llvm.masked.load.v8f32.p0(ptr %src, i32 4, <8 x i1> %mask, <8 x float> poison)
  store i32 1, ptr %other
  call void @llvm.masked.store.v8f32.p0(<8 x float> %v, ptr %dst, i32 4, <8 x i1> %mask)
  ret void
}

; Avoid adding guard control flow when the function is optimized for code size.
; CHECK-LABEL: define void @size_optimized(
; CHECK-NOT: br i1
; CHECK: ret void
define void @size_optimized(ptr %src, ptr %dst, <8 x i1> %mask) #1 {
  %v = call <8 x float> @llvm.masked.load.v8f32.p0(ptr %src, i32 4, <8 x i1> %mask, <8 x float> poison)
  call void @llvm.masked.store.v8f32.p0(<8 x float> %v, ptr %dst, i32 4, <8 x i1> %mask)
  ret void
}

; Guarding a memory-free convergent call can change collective participation.
; CHECK-LABEL: define void @convergent_call(
; CHECK-NOT: br i1
; CHECK: ret void
; CANON-LABEL: define void @convergent_call(
; CANON-NOT: br i1
; CANON: ret void
define void @convergent_call(ptr %src, ptr %dst, <8 x i1> %mask) convergent #0 {
  %v = call <8 x float> @llvm.masked.load.v8f32.p0(ptr %src, i32 4, <8 x i1> %mask, <8 x float> zeroinitializer)
  %sum = call <8 x float> @collective(<8 x float> %v)
  call void @llvm.masked.store.v8f32.p0(<8 x float> %sum, ptr %dst, i32 4, <8 x i1> %mask)
  ret void
}

; An explicit convergence token does not make the extra guard safe.
; CHECK-LABEL: define void @controlled_convergent_call(
; CHECK-NOT: br i1
; CHECK: ret void
; CANON-LABEL: define void @controlled_convergent_call(
; CANON-NOT: br i1
; CANON: ret void
define void @controlled_convergent_call(ptr %src, ptr %dst, <8 x i1> %mask) convergent #0 {
  %token = call token @llvm.experimental.convergence.entry()
  %v = call <8 x float> @llvm.masked.load.v8f32.p0(ptr %src, i32 4, <8 x i1> %mask, <8 x float> zeroinitializer)
  %sum = call <8 x float> @collective(<8 x float> %v) [ "convergencectrl"(token %token) ]
  call void @llvm.masked.store.v8f32.p0(<8 x float> %sum, ptr %dst, i32 4, <8 x i1> %mask)
  ret void
}

attributes #0 = { "target-features"="+avx2" }
attributes #1 = { optsize "target-features"="+avx2" }