; RUN: opt -S -passes=loop-vectorize < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux"

; Function Attrs: nounwind
define zeroext i32 @test() #0 {
; CHECK-LABEL: @test
; CHECK-NOT: x i32>

entry:
  %a = alloca [1600 x i32], align 4
  %c = alloca [1600 x i32], align 4
  call void @llvm.lifetime.start(ptr %a) #3
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  call void @llvm.lifetime.start(ptr %c) #3
  %call = call signext i32 @bar(ptr %a, ptr %c) #3
  br label %for.body6

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv25 = phi i64 [ 0, %entry ], [ %indvars.iv.next26, %for.body ]
  %arrayidx = getelementptr inbounds [1600 x i32], ptr %a, i64 0, i64 %indvars.iv25
  %0 = trunc i64 %indvars.iv25 to i32
  store i32 %0, ptr %arrayidx, align 4
  %indvars.iv.next26 = add nuw nsw i64 %indvars.iv25, 1
  %exitcond27 = icmp eq i64 %indvars.iv.next26, 1600
  br i1 %exitcond27, label %for.cond.cleanup, label %for.body

for.cond.cleanup5:                                ; preds = %for.body6
  call void @llvm.lifetime.end(ptr nonnull %c) #3
  call void @llvm.lifetime.end(ptr %a) #3
  ret i32 %add

for.body6:                                        ; preds = %for.body6, %for.cond.cleanup
  %indvars.iv = phi i64 [ 0, %for.cond.cleanup ], [ %indvars.iv.next, %for.body6 ]
  %s.022 = phi i32 [ 0, %for.cond.cleanup ], [ %add, %for.body6 ]
  %arrayidx8 = getelementptr inbounds [1600 x i32], ptr %c, i64 0, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx8, align 4
  %add = add i32 %1, %s.022
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1600
  br i1 %exitcond, label %for.cond.cleanup5, label %for.body6
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start(ptr nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end(ptr nocapture) #1

declare signext i32 @bar(ptr, ptr) #2

attributes #0 = { nounwind "target-features"="-altivec,-bpermd,-crypto,-direct-move,-extdiv,-power8-vector,-vsx" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "target-features"="-altivec,-bpermd,-crypto,-direct-move,-extdiv,-power8-vector,-vsx" }
attributes #3 = { nounwind }

