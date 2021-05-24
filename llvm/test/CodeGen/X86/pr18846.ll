; RUN: llc -O3 -disable-peephole -mcpu=corei7-avx -mattr=+avx < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; pr18846 - needless avx spill/reload
; Test for unnecessary repeated spills due to eliminateRedundantSpills failing
; to recognise unaligned ymm load/stores to the stack.
; Bugpoint reduced testcase.

;CHECK-LABEL: _Z16opt_kernel_cachePfS_S_
;CHECK-NOT:   vmovups {{.*#+}} 32-byte Folded Spill
;CHECK-NOT:   vmovups {{.*#+}} 32-byte Folded Reload

; Function Attrs: uwtable
define void @_Z16opt_kernel_cachePfS_S_() #0 {
entry:
  br label %for.body29

for.body29:                                       ; preds = %for.body29, %entry
  br i1 undef, label %for.body29, label %for.body65

for.body65:                                       ; preds = %for.body29
  %0 = load float, float* undef, align 4, !tbaa !1
  %vecinit7.i4448 = insertelement <8 x float> undef, float %0, i32 7
  %1 = load float, float* null, align 4, !tbaa !1
  %vecinit7.i4304 = insertelement <8 x float> undef, float %1, i32 7
  %2 = load float, float* undef, align 4, !tbaa !1
  %vecinit7.i4196 = insertelement <8 x float> undef, float %2, i32 7
  %3 = or i64 0, 16
  %add.ptr111.sum4096 = add i64 %3, 0
  %4 = load <8 x float>, <8 x float>* null, align 16, !tbaa !5
  %add.ptr162 = getelementptr inbounds [65536 x float], [65536 x float]* null, i64 0, i64 %add.ptr111.sum4096
  %__v.i4158 = bitcast float* %add.ptr162 to <8 x float>*
  %5 = load <8 x float>, <8 x float>* %__v.i4158, align 16, !tbaa !5
  %add.ptr158.sum40975066 = or i64 %add.ptr111.sum4096, 8
  %add.ptr183 = getelementptr inbounds [65536 x float], [65536 x float]* null, i64 0, i64 %add.ptr158.sum40975066
  %__v.i4162 = bitcast float* %add.ptr183 to <8 x float>*
  %6 = load <8 x float>, <8 x float>* %__v.i4162, align 16, !tbaa !5
  %add.ptr200.sum40995067 = or i64 undef, 8
  %add.ptr225 = getelementptr inbounds [65536 x float], [65536 x float]* null, i64 0, i64 %add.ptr200.sum40995067
  %__v.i4167 = bitcast float* %add.ptr225 to <8 x float>*
  %7 = load <8 x float>, <8 x float>* %__v.i4167, align 4, !tbaa !5
  %8 = load <8 x float>, <8 x float>* undef, align 16, !tbaa !5
  %add.ptr242.sum41015068 = or i64 0, 8
  %add.ptr267 = getelementptr inbounds [65536 x float], [65536 x float]* null, i64 0, i64 %add.ptr242.sum41015068
  %__v.i4171 = bitcast float* %add.ptr267 to <8 x float>*
  %9 = load <8 x float>, <8 x float>* %__v.i4171, align 4, !tbaa !5
  %mul.i4690 = fmul <8 x float> %7, undef
  %add.i4665 = fadd <8 x float> undef, undef
  %mul.i4616 = fmul <8 x float> %8, undef
  %mul.i4598 = fmul <8 x float> undef, undef
  %add.i4597 = fadd <8 x float> undef, %mul.i4598
  %mul.i4594 = fmul <8 x float> %6, undef
  %add.i4593 = fadd <8 x float> undef, %mul.i4594
  %mul.i4578 = fmul <8 x float> %9, undef
  %add.i4577 = fadd <8 x float> %add.i4593, %mul.i4578
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> %add.i4577) #1
  %10 = load <8 x float>, <8 x float>* null, align 16, !tbaa !5
  %11 = load <8 x float>, <8 x float>* undef, align 16, !tbaa !5
  %mul.i4564 = fmul <8 x float> %4, undef
  %add.i4563 = fadd <8 x float> %10, %mul.i4564
  %mul.i4560 = fmul <8 x float> %5, undef
  %add.i4559 = fadd <8 x float> %11, %mul.i4560
  %add.i4547 = fadd <8 x float> %add.i4563, undef
  %mul.i4546 = fmul <8 x float> %7, undef
  %add.i4545 = fadd <8 x float> undef, %mul.i4546
  %mul.i4544 = fmul <8 x float> %8, undef
  %add.i4543 = fadd <8 x float> %add.i4559, %mul.i4544
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> %add.i4547) #1
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> %add.i4545) #1
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> %add.i4543) #1
  %add.i4455 = fadd <8 x float> undef, undef
  %mul.i4454 = fmul <8 x float> undef, undef
  %add.i4453 = fadd <8 x float> undef, %mul.i4454
  %mul.i4440 = fmul <8 x float> zeroinitializer, %vecinit7.i4448
  %add.i4439 = fadd <8 x float> %add.i4455, %mul.i4440
  %mul.i4438 = fmul <8 x float> %7, %vecinit7.i4448
  %add.i4437 = fadd <8 x float> %add.i4453, %mul.i4438
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> %add.i4439) #1
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> %add.i4437) #1
  %add.i4413 = fadd <8 x float> zeroinitializer, undef
  %mul.i4400 = fmul <8 x float> %8, undef
  %add.i4399 = fadd <8 x float> undef, %mul.i4400
  %add.i4397 = fadd <8 x float> %add.i4413, zeroinitializer
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> zeroinitializer) #1
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> %add.i4399) #1
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> %add.i4397) #1
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> undef) #1
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> undef) #1
  %mul.i4330 = fmul <8 x float> %7, undef
  %add.i4329 = fadd <8 x float> undef, %mul.i4330
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> %add.i4329) #1
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> undef) #1
  %mul.i4312 = fmul <8 x float> %4, undef
  %add.i4311 = fadd <8 x float> undef, %mul.i4312
  %mul.i4306 = fmul <8 x float> %6, undef
  %add.i4305 = fadd <8 x float> undef, %mul.i4306
  %add.i4295 = fadd <8 x float> %add.i4311, undef
  %mul.i4294 = fmul <8 x float> %7, %vecinit7.i4304
  %add.i4293 = fadd <8 x float> undef, %mul.i4294
  %mul.i4292 = fmul <8 x float> %8, %vecinit7.i4304
  %add.i4291 = fadd <8 x float> undef, %mul.i4292
  %mul.i4290 = fmul <8 x float> %9, %vecinit7.i4304
  %add.i4289 = fadd <8 x float> %add.i4305, %mul.i4290
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> %add.i4295) #1
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> %add.i4293) #1
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> %add.i4291) #1
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> %add.i4289) #1
  %12 = load <8 x float>, <8 x float>* undef, align 16, !tbaa !5
  %mul.i4274 = fmul <8 x float> undef, undef
  %add.i4273 = fadd <8 x float> %12, %mul.i4274
  %mul.i4258 = fmul <8 x float> %7, undef
  %add.i4257 = fadd <8 x float> %add.i4273, %mul.i4258
  %mul.i4254 = fmul <8 x float> %9, undef
  %add.i4253 = fadd <8 x float> undef, %mul.i4254
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> %add.i4257) #1
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> %add.i4253) #1
  %mul.i = fmul <8 x float> %9, %vecinit7.i4196
  %add.i = fadd <8 x float> undef, %mul.i
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> zeroinitializer) #1
  call void @llvm.x86.avx.storeu.ps.256(i8* undef, <8 x float> %add.i) #1
  unreachable
}

; Function Attrs: nounwind
declare void @llvm.x86.avx.storeu.ps.256(i8*, <8 x float>) #1

attributes #0 = { uwtable "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.5 "}
!1 = !{!2, !2, i64 0}
!2 = !{!"float", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!3, !3, i64 0}
