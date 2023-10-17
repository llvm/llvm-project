; Check that if option prefer-no-gather/scatter can disable gather/scatter instructions.
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2,+fast-gather %s -o - | FileCheck %s --check-prefixes=GATHER
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2,+fast-gather,+prefer-no-gather %s -o - | FileCheck %s --check-prefixes=NO-GATHER
; RUN: llc -mtriple=x86_64-unknown-linux-gnu  -mattr=+avx512vl,+avx512dq < %s | FileCheck %s --check-prefix=SCATTER
; RUN: llc -mtriple=x86_64-unknown-linux-gnu  -mattr=+avx512vl,+avx512dq,+prefer-no-gather < %s | FileCheck %s --check-prefix=SCATTER-NO-GATHER
; RUN: llc -mtriple=x86_64-unknown-linux-gnu  -mattr=+avx512vl,+avx512dq,+prefer-no-scatter < %s | FileCheck %s --check-prefix=GATHER-NO-SCATTER
; RUN: llc -mtriple=x86_64-unknown-linux-gnu  -mattr=+avx512vl,+avx512dq,+prefer-no-gather,+prefer-no-scatter < %s | FileCheck %s --check-prefix=NO-SCATTER-GATHER

@A = global [1024 x i8] zeroinitializer, align 128
@B = global [1024 x i64] zeroinitializer, align 128
@C = global [1024 x i64] zeroinitializer, align 128

; This tests the function that if prefer-no-gather can disable lowerMGather
define void @test() #0 {
; GATHER-LABEL: test:
; GATHER: vpgatherdq
;
; NO-GATHER-LABEL: test:
; NO-GATHER-NOT: vpgatherdq
;
; GATHER-NO-SCATTER-LABEL: test:
; GATHER-NO-SCATTER: vpgatherdq
;
; NO-SCATTER-GATHER-LABEL: test:
; NO-SCATTER-GATHER-NOT: vpgatherdq
iter.check:
  br i1 false, label %vec.epilog.scalar.ph, label %vector.main.loop.iter.check

vector.main.loop.iter.check:                      ; preds = %iter.check
  br i1 false, label %vec.epilog.ph, label %vector.ph

vector.ph:                                        ; preds = %vector.main.loop.iter.check
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = add i64 %index, 0
  %1 = getelementptr inbounds [1024 x i8], ptr @A, i64 0, i64 %0
  %2 = getelementptr inbounds i8, ptr %1, i32 0
  %wide.load = load <32 x i8>, ptr %2, align 1
  %3 = sext <32 x i8> %wide.load to <32 x i64>
  %4 = getelementptr inbounds [1024 x i64], ptr @B, i64 0, <32 x i64> %3
  %wide.masked.gather = call <32 x i64> @llvm.masked.gather.v32i64.v32p0(<32 x ptr> %4, i32 8, <32 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <32 x i64> poison)
  %5 = getelementptr inbounds [1024 x i64], ptr @C, i64 0, i64 %0
  %6 = getelementptr inbounds i64, ptr %5, i32 0
  store <32 x i64> %wide.masked.gather, ptr %6, align 8
  %index.next = add nuw i64 %index, 32
  %7 = icmp eq i64 %index.next, 1024
  br i1 %7, label %middle.block, label %vector.body, !llvm.loop !0

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 1024, 1024
  br i1 %cmp.n, label %for.cond.cleanup, label %vec.epilog.iter.check

vec.epilog.iter.check:                            ; preds = %middle.block
  br i1 true, label %vec.epilog.scalar.ph, label %vec.epilog.ph

vec.epilog.ph:                                    ; preds = %vector.main.loop.iter.check, %vec.epilog.iter.check
  %vec.epilog.resume.val = phi i64 [ 1024, %vec.epilog.iter.check ], [ 0, %vector.main.loop.iter.check ]
  br label %vec.epilog.vector.body

vec.epilog.vector.body:                           ; preds = %vec.epilog.vector.body, %vec.epilog.ph
  %index2 = phi i64 [ %vec.epilog.resume.val, %vec.epilog.ph ], [ %index.next5, %vec.epilog.vector.body ]
  %8 = add i64 %index2, 0
  %9 = getelementptr inbounds [1024 x i8], ptr @A, i64 0, i64 %8
  %10 = getelementptr inbounds i8, ptr %9, i32 0
  %wide.load3 = load <16 x i8>, ptr %10, align 1
  %11 = sext <16 x i8> %wide.load3 to <16 x i64>
  %12 = getelementptr inbounds [1024 x i64], ptr @B, i64 0, <16 x i64> %11
  %wide.masked.gather4 = call <16 x i64> @llvm.masked.gather.v16i64.v16p0(<16 x ptr> %12, i32 8, <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <16 x i64> poison)
  %13 = getelementptr inbounds [1024 x i64], ptr @C, i64 0, i64 %8
  %14 = getelementptr inbounds i64, ptr %13, i32 0
  store <16 x i64> %wide.masked.gather4, ptr %14, align 8
  %index.next5 = add nuw i64 %index2, 16
  %15 = icmp eq i64 %index.next5, 1024
  br i1 %15, label %vec.epilog.middle.block, label %vec.epilog.vector.body, !llvm.loop !2

vec.epilog.middle.block:                          ; preds = %vec.epilog.vector.body
  %cmp.n1 = icmp eq i64 1024, 1024
  br i1 %cmp.n1, label %for.cond.cleanup, label %vec.epilog.scalar.ph

vec.epilog.scalar.ph:                             ; preds = %iter.check, %vec.epilog.iter.check, %vec.epilog.middle.block
  %bc.resume.val = phi i64 [ 1024, %vec.epilog.middle.block ], [ 1024, %vec.epilog.iter.check ], [ 0, %iter.check ]
  br label %for.body

for.body:                                         ; preds = %for.body, %vec.epilog.scalar.ph
  %iv = phi i64 [ %bc.resume.val, %vec.epilog.scalar.ph ], [ %iv.next, %for.body ]
  %inA = getelementptr inbounds [1024 x i8], ptr @A, i64 0, i64 %iv
  %valA = load i8, ptr %inA, align 1
  %valA.ext = sext i8 %valA to i64
  %inB = getelementptr inbounds [1024 x i64], ptr @B, i64 0, i64 %valA.ext
  %valB = load i64, ptr %inB, align 8
  %out = getelementptr inbounds [1024 x i64], ptr @C, i64 0, i64 %iv
  store i64 %valB, ptr %out, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp = icmp ult i64 %iv.next, 1024
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !4

for.cond.cleanup:                                 ; preds = %vec.epilog.middle.block, %middle.block, %for.body
  ret void
}

declare <32 x i64> @llvm.masked.gather.v32i64.v32p0(<32 x ptr>, i32 immarg, <32 x i1>, <32 x i64>) #1

declare <16 x i64> @llvm.masked.gather.v16i64.v16p0(<16 x ptr>, i32 immarg, <16 x i1>, <16 x i64>) #1
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.isvectorized", i32 1}
!2 = distinct !{!2, !1, !3}
!3 = !{!"llvm.loop.unroll.runtime.disable"}
!4 = distinct !{!4, !3, !1}

; This tests the function that if prefer-no-gather can disable ScalarizeMaskedGather
define <4 x float> @gather_v4f32_ptr_v4i32(<4 x ptr> %ptr, <4 x i32> %trigger, <4 x float> %passthru) {
; GATHER-LABEL: gather_v4f32_ptr_v4i32:
; GATHER: vgatherqps
;
; NO-GATHER-LABEL: gather_v4f32_ptr_v4i32:
; NO-GATHER-NOT: vgatherqps
;
; GATHER-NO-SCATTER-LABEL: gather_v4f32_ptr_v4i32:
; GATHER-NO-SCATTER: vgatherqps
;
; NO-SCATTER-GATHER-LABEL: gather_v4f32_ptr_v4i32:
; NO-SCATTER-GATHER-NOT: vgatherqps
  %mask = icmp eq <4 x i32> %trigger, zeroinitializer
  %res = call <4 x float> @llvm.masked.gather.v4f32.v4p0(<4 x ptr> %ptr, i32 4, <4 x i1> %mask, <4 x float> %passthru)
  ret <4 x float> %res
}

declare <4 x float> @llvm.masked.gather.v4f32.v4p0(<4 x ptr>, i32, <4 x i1>, <4 x float>)

%struct.a = type { [4 x i32], [4 x i8], %struct.b, i32 }
%struct.b = type { i32, i32 }
@c = external dso_local global %struct.a, align 4

; This tests the function that if prefer-no-gather can disable ScalarizeMaskedGather
define <8 x i32> @gather_v8i32_v8i32(<8 x i32> %trigger) {
; GATHER-LABEL: gather_v8i32_v8i32:
; GATHER: vpgatherdd
;
; NO-GATHER-LABEL: gather_v8i32_v8i32:
; NO-GATHER-NOT: vpgatherdd
;
; NO-SCATTER-GATHER-LABEL: gather_v8i32_v8i32:
; NO-SCATTER-GATHER-NOT: vpgatherdd
  %1 = icmp eq <8 x i32> %trigger, zeroinitializer
  %2 = call <8 x i32> @llvm.masked.gather.v8i32.v8p0(<8 x ptr> getelementptr (%struct.a, <8 x ptr> <ptr @c, ptr @c, ptr @c, ptr @c, ptr @c, ptr @c, ptr @c, ptr @c>, <8 x i64> zeroinitializer, i32 0, <8 x i64> <i64 3, i64 3, i64 3, i64 3, i64 3, i64 3, i64 3, i64 3>), i32 4, <8 x i1> %1, <8 x i32> undef)
  %3 = call <8 x i32> @llvm.masked.gather.v8i32.v8p0(<8 x ptr> getelementptr (%struct.a, <8 x ptr> <ptr @c, ptr @c, ptr @c, ptr @c, ptr @c, ptr @c, ptr @c, ptr @c>, <8 x i64> zeroinitializer, i32 3), i32 4, <8 x i1> %1, <8 x i32> undef)
  %4 = add <8 x i32> %2, %3
  %5 = call <8 x i32> @llvm.masked.gather.v8i32.v8p0(<8 x ptr> getelementptr (%struct.a, <8 x ptr> <ptr @c, ptr @c, ptr @c, ptr @c, ptr @c, ptr @c, ptr @c, ptr @c>, <8 x i64> zeroinitializer, i32 3), i32 4, <8 x i1> %1, <8 x i32> undef)
  %6 = add <8 x i32> %4, %5
  ret <8 x i32> %6
}

declare <8 x i32> @llvm.masked.gather.v8i32.v8p0(<8 x ptr>, i32, <8 x i1>, <8 x i32>)

; scatter test cases 
define void @scatter_test1(ptr %base, <16 x i32> %ind, i16 %mask, <16 x i32>%val) {
; SCATTER-LABEL: scatter_test1:
; SCATTER: vpscatterdd
;
; SCATTER-NO-GATHER-LABEL: scatter_test1:
; SCATTER-NO-GATHER: vpscatterdd
;
; GATHER-NO-SCATTER-LABEL: scatter_test1:
; GATHER-NO-SCATTER-NOT: vpscatterdd
;
; NO-SCATTER-GATHER-LABEL: scatter_test1:
; NO-SCATTER-GATHER-NOT: vpscatterdd
  %broadcast.splatinsert = insertelement <16 x ptr> undef, ptr %base, i32 0
  %broadcast.splat = shufflevector <16 x ptr> %broadcast.splatinsert, <16 x ptr> undef, <16 x i32> zeroinitializer

  %gep.random = getelementptr i32, <16 x ptr> %broadcast.splat, <16 x i32> %ind
  %imask = bitcast i16 %mask to <16 x i1>
  call void @llvm.masked.scatter.v16i32.v16p0(<16 x i32>%val, <16 x ptr> %gep.random, i32 4, <16 x i1> %imask)
  call void @llvm.masked.scatter.v16i32.v16p0(<16 x i32>%val, <16 x ptr> %gep.random, i32 4, <16 x i1> %imask)
  ret void
}

declare void @llvm.masked.scatter.v8i32.v8p0(<8 x i32> , <8 x ptr> , i32 , <8 x i1> )
declare void @llvm.masked.scatter.v16i32.v16p0(<16 x i32> , <16 x ptr> , i32 , <16 x i1> )

define <8 x i32> @scatter_test2(<8 x i32>%a1, <8 x ptr> %ptr) {
; SCATTER-LABEL: scatter_test2:
; SCATTER: vpscatterqd
;
; SCATTER-NO-GATHER-LABEL: scatter_test2:
; SCATTER-NO-GATHER: vpscatterqd
;
; GATHER-NO-SCATTER-LABEL: scatter_test2:
; GATHER-NO-SCATTER-NOT: vpscatterqd
;
; NO-SCATTER-GATHER-LABEL: scatter_test2:
; NO-SCATTER-GATHER-NOT: vpscatterqd
  %a = call <8 x i32> @llvm.masked.gather.v8i32.v8p0(<8 x ptr> %ptr, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>, <8 x i32> undef)

  call void @llvm.masked.scatter.v8i32.v8p0(<8 x i32> %a1, <8 x ptr> %ptr, i32 4, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
  ret <8 x i32>%a
}
