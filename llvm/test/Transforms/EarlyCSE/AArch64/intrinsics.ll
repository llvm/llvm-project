; RUN: opt < %s -S -mtriple=aarch64-none-linux-gnu -mattr=+neon -passes=early-cse -earlycse-debug-hash | FileCheck %s
; RUN: opt < %s -S -mtriple=aarch64-none-linux-gnu -mattr=+neon -aa-pipeline=basic-aa -passes='early-cse<memssa>' | FileCheck %s

define <4 x i32> @test_cse(ptr %a, [2 x <4 x i32>] %s.coerce, i32 %n) {
entry:
; Check that @llvm.aarch64.neon.ld2 is optimized away by Early CSE.
; CHECK-LABEL: @test_cse
; CHECK-NOT: call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0
  %s.coerce.fca.0.extract = extractvalue [2 x <4 x i32>] %s.coerce, 0
  %s.coerce.fca.1.extract = extractvalue [2 x <4 x i32>] %s.coerce, 1
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %res.0 = phi <4 x i32> [ undef, %entry ], [ %call, %for.body ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %0 = bitcast <4 x i32> %s.coerce.fca.0.extract to <16 x i8>
  %1 = bitcast <4 x i32> %s.coerce.fca.1.extract to <16 x i8>
  %2 = bitcast <16 x i8> %0 to <4 x i32>
  %3 = bitcast <16 x i8> %1 to <4 x i32>
  call void @llvm.aarch64.neon.st2.v4i32.p0(<4 x i32> %2, <4 x i32> %3, ptr %a)
  %vld2 = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0(ptr %a)
  %vld2.fca.0.extract = extractvalue { <4 x i32>, <4 x i32> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <4 x i32>, <4 x i32> } %vld2, 1
  %call = call <4 x i32> @vaddq_s32(<4 x i32> %vld2.fca.0.extract, <4 x i32> %vld2.fca.0.extract)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret <4 x i32> %res.0
}

define <4 x i32> @test_cse2(ptr %a, [2 x <4 x i32>] %s.coerce, i32 %n) {
entry:
; Check that the first @llvm.aarch64.neon.st2 is optimized away by Early CSE.
; CHECK-LABEL: @test_cse2
; CHECK-NOT: call void @llvm.aarch64.neon.st2.v4i32.p0(<4 x i32> %3, <4 x i32> %3, ptr %0)
; CHECK: call void @llvm.aarch64.neon.st2.v4i32.p0(<4 x i32> %s.coerce.fca.0.extract, <4 x i32> %s.coerce.fca.1.extract, ptr %a)
  %s.coerce.fca.0.extract = extractvalue [2 x <4 x i32>] %s.coerce, 0
  %s.coerce.fca.1.extract = extractvalue [2 x <4 x i32>] %s.coerce, 1
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %res.0 = phi <4 x i32> [ undef, %entry ], [ %call, %for.body ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %0 = bitcast <4 x i32> %s.coerce.fca.0.extract to <16 x i8>
  %1 = bitcast <4 x i32> %s.coerce.fca.1.extract to <16 x i8>
  %2 = bitcast <16 x i8> %0 to <4 x i32>
  %3 = bitcast <16 x i8> %1 to <4 x i32>
  call void @llvm.aarch64.neon.st2.v4i32.p0(<4 x i32> %2, <4 x i32> %2, ptr %a)
  call void @llvm.aarch64.neon.st2.v4i32.p0(<4 x i32> %2, <4 x i32> %3, ptr %a)
  %vld2 = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0(ptr %a)
  %vld2.fca.0.extract = extractvalue { <4 x i32>, <4 x i32> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <4 x i32>, <4 x i32> } %vld2, 1
  %call = call <4 x i32> @vaddq_s32(<4 x i32> %vld2.fca.0.extract, <4 x i32> %vld2.fca.0.extract)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret <4 x i32> %res.0
}

define <4 x i32> @test_cse3(ptr %a, [2 x <4 x i32>] %s.coerce, i32 %n) #0 {
entry:
; Check that the first @llvm.aarch64.neon.ld2 is optimized away by Early CSE.
; CHECK-LABEL: @test_cse3
; CHECK: call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0
; CHECK-NOT: call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0
  %s.coerce.fca.0.extract = extractvalue [2 x <4 x i32>] %s.coerce, 0
  %s.coerce.fca.1.extract = extractvalue [2 x <4 x i32>] %s.coerce, 1
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %res.0 = phi <4 x i32> [ undef, %entry ], [ %call, %for.body ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %vld2 = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0(ptr %a)
  %vld2.fca.0.extract = extractvalue { <4 x i32>, <4 x i32> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <4 x i32>, <4 x i32> } %vld2, 1
  %vld22 = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0(ptr %a)
  %vld22.fca.0.extract = extractvalue { <4 x i32>, <4 x i32> } %vld22, 0
  %vld22.fca.1.extract = extractvalue { <4 x i32>, <4 x i32> } %vld22, 1
  %call = call <4 x i32> @vaddq_s32(<4 x i32> %vld2.fca.0.extract, <4 x i32> %vld22.fca.0.extract)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret <4 x i32> %res.0
}


define <4 x i32> @test_nocse(ptr %a, ptr %b, [2 x <4 x i32>] %s.coerce, i32 %n) {
entry:
; Check that the store prevents @llvm.aarch64.neon.ld2 from being optimized
; away by Early CSE.
; CHECK-LABEL: @test_nocse
; CHECK: call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0
  %s.coerce.fca.0.extract = extractvalue [2 x <4 x i32>] %s.coerce, 0
  %s.coerce.fca.1.extract = extractvalue [2 x <4 x i32>] %s.coerce, 1
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %res.0 = phi <4 x i32> [ undef, %entry ], [ %call, %for.body ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %0 = bitcast <4 x i32> %s.coerce.fca.0.extract to <16 x i8>
  %1 = bitcast <4 x i32> %s.coerce.fca.1.extract to <16 x i8>
  %2 = bitcast <16 x i8> %0 to <4 x i32>
  %3 = bitcast <16 x i8> %1 to <4 x i32>
  call void @llvm.aarch64.neon.st2.v4i32.p0(<4 x i32> %2, <4 x i32> %3, ptr %a)
  store i32 0, ptr %b, align 4
  %vld2 = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0(ptr %a)
  %vld2.fca.0.extract = extractvalue { <4 x i32>, <4 x i32> } %vld2, 0
  %vld2.fca.1.extract = extractvalue { <4 x i32>, <4 x i32> } %vld2, 1
  %call = call <4 x i32> @vaddq_s32(<4 x i32> %vld2.fca.0.extract, <4 x i32> %vld2.fca.0.extract)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret <4 x i32> %res.0
}

define <4 x i32> @test_nocse2(ptr %a, [2 x <4 x i32>] %s.coerce, i32 %n) {
entry:
; Check that @llvm.aarch64.neon.ld3 is not optimized away by Early CSE due
; to mismatch between st2 and ld3.
; CHECK-LABEL: @test_nocse2
; CHECK: call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld3.v4i32.p0
  %s.coerce.fca.0.extract = extractvalue [2 x <4 x i32>] %s.coerce, 0
  %s.coerce.fca.1.extract = extractvalue [2 x <4 x i32>] %s.coerce, 1
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %res.0 = phi <4 x i32> [ undef, %entry ], [ %call, %for.body ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %0 = bitcast <4 x i32> %s.coerce.fca.0.extract to <16 x i8>
  %1 = bitcast <4 x i32> %s.coerce.fca.1.extract to <16 x i8>
  %2 = bitcast <16 x i8> %0 to <4 x i32>
  %3 = bitcast <16 x i8> %1 to <4 x i32>
  call void @llvm.aarch64.neon.st2.v4i32.p0(<4 x i32> %2, <4 x i32> %3, ptr %a)
  %vld3 = call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld3.v4i32.p0(ptr %a)
  %vld3.fca.0.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %vld3, 0
  %vld3.fca.2.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %vld3, 2
  %call = call <4 x i32> @vaddq_s32(<4 x i32> %vld3.fca.0.extract, <4 x i32> %vld3.fca.2.extract)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret <4 x i32> %res.0
}

define <4 x i32> @test_nocse3(ptr %a, [2 x <4 x i32>] %s.coerce, i32 %n) {
entry:
; Check that @llvm.aarch64.neon.st3 is not optimized away by Early CSE due to
; mismatch between st2 and st3.
; CHECK-LABEL: @test_nocse3
; CHECK: call void @llvm.aarch64.neon.st3.v4i32.p0
; CHECK: call void @llvm.aarch64.neon.st2.v4i32.p0
  %s.coerce.fca.0.extract = extractvalue [2 x <4 x i32>] %s.coerce, 0
  %s.coerce.fca.1.extract = extractvalue [2 x <4 x i32>] %s.coerce, 1
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %res.0 = phi <4 x i32> [ undef, %entry ], [ %call, %for.body ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %0 = bitcast <4 x i32> %s.coerce.fca.0.extract to <16 x i8>
  %1 = bitcast <4 x i32> %s.coerce.fca.1.extract to <16 x i8>
  %2 = bitcast <16 x i8> %0 to <4 x i32>
  %3 = bitcast <16 x i8> %1 to <4 x i32>
  call void @llvm.aarch64.neon.st3.v4i32.p0(<4 x i32> %3, <4 x i32> %2, <4 x i32> %2, ptr %a)
  call void @llvm.aarch64.neon.st2.v4i32.p0(<4 x i32> %2, <4 x i32> %2, ptr %a)
  %vld3 = call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld3.v4i32.p0(ptr %a)
  %vld3.fca.0.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %vld3, 0
  %vld3.fca.1.extract = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } %vld3, 1
  %call = call <4 x i32> @vaddq_s32(<4 x i32> %vld3.fca.0.extract, <4 x i32> %vld3.fca.0.extract)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret <4 x i32> %res.0
}

; Function Attrs: nounwind
declare void @llvm.aarch64.neon.st2.v4i32.p0(<4 x i32>, <4 x i32>, ptr nocapture)

; Function Attrs: nounwind
declare void @llvm.aarch64.neon.st3.v4i32.p0(<4 x i32>, <4 x i32>, <4 x i32>, ptr nocapture)

; Function Attrs: nounwind readonly
declare { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0(ptr)

; Function Attrs: nounwind readonly
declare { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld3.v4i32.p0(ptr)

define internal fastcc <4 x i32> @vaddq_s32(<4 x i32> %__p0, <4 x i32> %__p1) {
entry:
  %add = add <4 x i32> %__p0, %__p1
  ret <4 x i32> %add
}
