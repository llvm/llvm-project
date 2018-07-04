; RUN: llc -O3 -mtriple arm64-apple-ios3 -aarch64-enable-gep-opt=false %s -o - | FileCheck %s
; <rdar://problem/13621857>

@block = common global i8* null, align 8

define i32 @fct(i32 %i1, i32 %i2) {
; CHECK: @fct
; Sign extension is used more than once, thus it should not be folded.
; CodeGenPrepare is not sharing sext across uses, thus this is folded because
; of that.
; _CHECK-NOT: , sxtw]
entry:
  %idxprom = sext i32 %i1 to i64
  %0 = load i8*, i8** @block, align 8
  %arrayidx = getelementptr inbounds i8, i8* %0, i64 %idxprom
  %1 = load i8, i8* %arrayidx, align 1
  %idxprom1 = sext i32 %i2 to i64
  %arrayidx2 = getelementptr inbounds i8, i8* %0, i64 %idxprom1
  %2 = load i8, i8* %arrayidx2, align 1
  %cmp = icmp eq i8 %1, %2
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %cmp7 = icmp ugt i8 %1, %2
  %conv8 = zext i1 %cmp7 to i32
  br label %return

if.end:                                           ; preds = %entry
  %inc = add nsw i32 %i1, 1
  %inc9 = add nsw i32 %i2, 1
  %idxprom10 = sext i32 %inc to i64
  %arrayidx11 = getelementptr inbounds i8, i8* %0, i64 %idxprom10
  %3 = load i8, i8* %arrayidx11, align 1
  %idxprom12 = sext i32 %inc9 to i64
  %arrayidx13 = getelementptr inbounds i8, i8* %0, i64 %idxprom12
  %4 = load i8, i8* %arrayidx13, align 1
  %cmp16 = icmp eq i8 %3, %4
  br i1 %cmp16, label %if.end23, label %if.then18

if.then18:                                        ; preds = %if.end
  %cmp21 = icmp ugt i8 %3, %4
  %conv22 = zext i1 %cmp21 to i32
  br label %return

if.end23:                                         ; preds = %if.end
  %inc24 = add nsw i32 %i1, 2
  %inc25 = add nsw i32 %i2, 2
  %idxprom26 = sext i32 %inc24 to i64
  %arrayidx27 = getelementptr inbounds i8, i8* %0, i64 %idxprom26
  %5 = load i8, i8* %arrayidx27, align 1
  %idxprom28 = sext i32 %inc25 to i64
  %arrayidx29 = getelementptr inbounds i8, i8* %0, i64 %idxprom28
  %6 = load i8, i8* %arrayidx29, align 1
  %cmp32 = icmp eq i8 %5, %6
  br i1 %cmp32, label %return, label %if.then34

if.then34:                                        ; preds = %if.end23
  %cmp37 = icmp ugt i8 %5, %6
  %conv38 = zext i1 %cmp37 to i32
  br label %return

return:                                           ; preds = %if.end23, %if.then34, %if.then18, %if.then
  %retval.0 = phi i32 [ %conv8, %if.then ], [ %conv22, %if.then18 ], [ %conv38, %if.then34 ], [ 1, %if.end23 ]
  ret i32 %retval.0
}

define i32 @fct1(i32 %i1, i32 %i2) optsize {
; CHECK: @fct1
; Addressing are folded when optimizing for code size.
; CHECK: , sxtw]
; CHECK: , sxtw]
entry:
  %idxprom = sext i32 %i1 to i64
  %0 = load i8*, i8** @block, align 8
  %arrayidx = getelementptr inbounds i8, i8* %0, i64 %idxprom
  %1 = load i8, i8* %arrayidx, align 1
  %idxprom1 = sext i32 %i2 to i64
  %arrayidx2 = getelementptr inbounds i8, i8* %0, i64 %idxprom1
  %2 = load i8, i8* %arrayidx2, align 1
  %cmp = icmp eq i8 %1, %2
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %cmp7 = icmp ugt i8 %1, %2
  %conv8 = zext i1 %cmp7 to i32
  br label %return

if.end:                                           ; preds = %entry
  %inc = add nsw i32 %i1, 1
  %inc9 = add nsw i32 %i2, 1
  %idxprom10 = sext i32 %inc to i64
  %arrayidx11 = getelementptr inbounds i8, i8* %0, i64 %idxprom10
  %3 = load i8, i8* %arrayidx11, align 1
  %idxprom12 = sext i32 %inc9 to i64
  %arrayidx13 = getelementptr inbounds i8, i8* %0, i64 %idxprom12
  %4 = load i8, i8* %arrayidx13, align 1
  %cmp16 = icmp eq i8 %3, %4
  br i1 %cmp16, label %if.end23, label %if.then18

if.then18:                                        ; preds = %if.end
  %cmp21 = icmp ugt i8 %3, %4
  %conv22 = zext i1 %cmp21 to i32
  br label %return

if.end23:                                         ; preds = %if.end
  %inc24 = add nsw i32 %i1, 2
  %inc25 = add nsw i32 %i2, 2
  %idxprom26 = sext i32 %inc24 to i64
  %arrayidx27 = getelementptr inbounds i8, i8* %0, i64 %idxprom26
  %5 = load i8, i8* %arrayidx27, align 1
  %idxprom28 = sext i32 %inc25 to i64
  %arrayidx29 = getelementptr inbounds i8, i8* %0, i64 %idxprom28
  %6 = load i8, i8* %arrayidx29, align 1
  %cmp32 = icmp eq i8 %5, %6
  br i1 %cmp32, label %return, label %if.then34

if.then34:                                        ; preds = %if.end23
  %cmp37 = icmp ugt i8 %5, %6
  %conv38 = zext i1 %cmp37 to i32
  br label %return

return:                                           ; preds = %if.end23, %if.then34, %if.then18, %if.then
  %retval.0 = phi i32 [ %conv8, %if.then ], [ %conv22, %if.then18 ], [ %conv38, %if.then34 ], [ 1, %if.end23 ]
  ret i32 %retval.0
}

; CHECK: @test
; CHECK-NOT: , uxtw #2]
define i32 @test(i32* %array, i8 zeroext %c, i32 %arg) {
entry:
  %conv = zext i8 %c to i32
  %add = sub i32 0, %arg
  %tobool = icmp eq i32 %conv, %add
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %idxprom = zext i8 %c to i64
  %arrayidx = getelementptr inbounds i32, i32* %array, i64 %idxprom
  %0 = load volatile i32, i32* %arrayidx, align 4
  %1 = load volatile i32, i32* %arrayidx, align 4
  %add3 = add nsw i32 %1, %0
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %res.0 = phi i32 [ %add3, %if.then ], [ 0, %entry ]
  ret i32 %res.0
}


; CHECK: @test2
; CHECK: , uxtw #2]
; CHECK: , uxtw #2]
define i32 @test2(i32* %array, i8 zeroext %c, i32 %arg) optsize {
entry:
  %conv = zext i8 %c to i32
  %add = sub i32 0, %arg
  %tobool = icmp eq i32 %conv, %add
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %idxprom = zext i8 %c to i64
  %arrayidx = getelementptr inbounds i32, i32* %array, i64 %idxprom
  %0 = load volatile i32, i32* %arrayidx, align 4
  %1 = load volatile i32, i32* %arrayidx, align 4
  %add3 = add nsw i32 %1, %0
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %res.0 = phi i32 [ %add3, %if.then ], [ 0, %entry ]
  ret i32 %res.0
}
