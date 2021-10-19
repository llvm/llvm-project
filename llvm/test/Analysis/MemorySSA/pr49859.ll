; RUN: opt -passes='loop-mssa(licm),print<memoryssa>' -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

@arr = global [12 x i8] [i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12], align 1

; CHECK-LABEL: @func()
define void @func()  {
entry:
  %res.addr.i = alloca i8, align 1
  %sum = alloca i8, align 1
  %n = alloca i8, align 1
  %i = alloca i8, align 1
  %cleanup.dest.slot = alloca i32, align 1
  call  void @llvm.lifetime.start.p0i8(i64 1, i8* %sum) #3
  store i8 0, i8* %sum, align 1
  call  void @llvm.lifetime.start.p0i8(i64 1, i8* %n) #3
  %call = call  i8 @idi(i8 10)
  store i8 %call, i8* %n, align 1
  call  void @llvm.lifetime.start.p0i8(i64 1, i8* %i) #3
  store i8 0, i8* %i, align 1
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i8, i8* %i, align 1
  %1 = load i8, i8* %n, align 1
  %cmp = icmp slt i8 %0, %1
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  store i32 2, i32* %cleanup.dest.slot, align 1
  br label %final.cleanup

for.body:                                         ; preds = %for.cond
  %2 = load i8, i8* %i, align 1
  %idxprom = sext i8 %2 to i32
  %arrayidx = getelementptr inbounds [12 x i8], [12 x i8]* @arr, i32 0, i32 %idxprom
  %3 = load i8, i8* %arrayidx, align 1
  %cmp1 = icmp eq i8 %3, 3
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  store i32 2, i32* %cleanup.dest.slot, align 1
  br label %final.cleanup

if.end:                                           ; preds = %for.body
  %4 = load i8, i8* %i, align 1
  %idxprom2 = sext i8 %4 to i32
  %arrayidx3 = getelementptr inbounds [12 x i8], [12 x i8]* @arr, i32 0, i32 %idxprom2
  %5 = load i8, i8* %arrayidx3, align 1
  %6 = load i8, i8* %sum, align 1
  %add = add nsw i8 %6, %5
  store i8 %add, i8* %sum, align 1
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %7 = load i8, i8* %i, align 1
  %inc = add nsw i8 %7, 1
  store i8 %inc, i8* %i, align 1
  br label %for.cond

; CHECK: final.cleanup:
; CHECK-NEXT: ; [[NO20:.*]] = MemoryPhi({if.then,[[NO9:.*]]},{for.cond.cleanup,[[NO8:.*]]})
; CHECK-NEXT: ; [[NO12:.*]] = MemoryDef([[NO20]])
; CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 1, i8* %i)
final.cleanup:                                          ; preds = %if.then, %for.cond.cleanup
  call  void @llvm.lifetime.end.p0i8(i64 1, i8* %i) #3
  br label %for.end

; CHECK: for.end:
; CHECK-NEXT: ; MemoryUse([[NO12]])
; CHECK-NEXT:  %3 = load i8, i8* %sum, align 1
for.end:                                          ; preds = %final.cleanup
  %8 = load i8, i8* %sum, align 1
  call  void @llvm.lifetime.start.p0i8(i64 1, i8* %res.addr.i)
  store i8 %8, i8* %res.addr.i, align 1
  %9 = load i8, i8* %res.addr.i, align 1
  call  void @foo(i8 %9) #3
  call  void @llvm.lifetime.end.p0i8(i64 1, i8* %res.addr.i)
  call  void @llvm.lifetime.end.p0i8(i64 1, i8* %n) #3
  call  void @llvm.lifetime.end.p0i8(i64 1, i8* %sum) #3
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)  #1

declare i8 @idi(i8)

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)  #1

; Function Attrs: nounwind
declare void @foo(i8)

attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #3 = { nounwind }
