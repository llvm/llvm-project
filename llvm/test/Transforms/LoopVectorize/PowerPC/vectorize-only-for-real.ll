; RUN: opt -S -loop-vectorize < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux"

; Function Attrs: nounwind
define zeroext i32 @test() #0 {
; CHECK-LABEL: @test
; CHECK-NOT: x i32>

entry:
  %a = alloca [1600 x i32], align 4
  %c = alloca [1600 x i32], align 4
  %0 = bitcast [1600 x i32]* %a to i8*
  call void @llvm.lifetime.start(i64 6400, i8* %0) #3
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %1 = bitcast [1600 x i32]* %c to i8*
  call void @llvm.lifetime.start(i64 6400, i8* %1) #3
  %arraydecay = getelementptr inbounds [1600 x i32], [1600 x i32]* %a, i64 0, i64 0
  %arraydecay1 = getelementptr inbounds [1600 x i32], [1600 x i32]* %c, i64 0, i64 0
  %call = call signext i32 @bar(i32* %arraydecay, i32* %arraydecay1) #3
  br label %for.body6

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv25 = phi i64 [ 0, %entry ], [ %indvars.iv.next26, %for.body ]
  %arrayidx = getelementptr inbounds [1600 x i32], [1600 x i32]* %a, i64 0, i64 %indvars.iv25
  %2 = trunc i64 %indvars.iv25 to i32
  store i32 %2, i32* %arrayidx, align 4
  %indvars.iv.next26 = add nuw nsw i64 %indvars.iv25, 1
  %exitcond27 = icmp eq i64 %indvars.iv.next26, 1600
  br i1 %exitcond27, label %for.cond.cleanup, label %for.body

for.cond.cleanup5:                                ; preds = %for.body6
  call void @llvm.lifetime.end(i64 6400, i8* nonnull %1) #3
  call void @llvm.lifetime.end(i64 6400, i8* %0) #3
  ret i32 %add

for.body6:                                        ; preds = %for.body6, %for.cond.cleanup
  %indvars.iv = phi i64 [ 0, %for.cond.cleanup ], [ %indvars.iv.next, %for.body6 ]
  %s.022 = phi i32 [ 0, %for.cond.cleanup ], [ %add, %for.body6 ]
  %arrayidx8 = getelementptr inbounds [1600 x i32], [1600 x i32]* %c, i64 0, i64 %indvars.iv
  %3 = load i32, i32* %arrayidx8, align 4
  %add = add i32 %3, %s.022
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1600
  br i1 %exitcond, label %for.cond.cleanup5, label %for.body6
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #1

declare signext i32 @bar(i32*, i32*) #2

attributes #0 = { nounwind "target-features"="-altivec,-bpermd,-crypto,-direct-move,-extdiv,-power8-vector,-vsx" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "target-features"="-altivec,-bpermd,-crypto,-direct-move,-extdiv,-power8-vector,-vsx" }
attributes #3 = { nounwind }

