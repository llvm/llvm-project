; RUN: opt -passes='module(function(mem2reg,mergereturn),ripple,function(dce))' -S %s | FileCheck %s --implicit-check-not="warning:"

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z3foomPf(i64 noundef %N, ptr noundef %a) #0 {
entry:
  %N.addr = alloca i64, align 8
  %a.addr = alloca ptr, align 8
  %v0 = alloca i64, align 8
  %i = alloca i64, align 8
  store i64 %N, ptr %N.addr, align 8
  store ptr %a, ptr %a.addr, align 8
  %0 = load i64, ptr %N.addr, align 8
  %BS = call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 8, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  %1 = call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  store i64 %1, ptr %v0, align 8
  %2 = load i64, ptr %v0, align 8
  %cmp = icmp ult i64 %2, 7
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i64 0, ptr %i, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %if.then
  %3 = load i64, ptr %i, align 8
  %4 = load i64, ptr %v0, align 8
  %add = add i64 %3, %4
  %5 = load i64, ptr %N.addr, align 8
  %cmp1 = icmp ult i64 %add, %5
  br i1 %cmp1, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %6 = load i64, ptr %v0, align 8
  %conv = uitofp i64 %6 to float
  %7 = load ptr, ptr %a.addr, align 8
  %8 = load i64, ptr %i, align 8
  %9 = load i64, ptr %v0, align 8
  %add2 = add i64 %8, %9
  %arrayidx = getelementptr inbounds nuw float, ptr %7, i64 %add2
  store float %conv, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %10 = load i64, ptr %i, align 8
  %add3 = add i64 %10, 8
  store i64 %add3, ptr %i, align 8
  br label %for.cond, !llvm.loop !6

for.end:                                          ; preds = %for.cond
  br label %if.end

if.end:                                           ; preds = %for.end, %entry
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare ptr @llvm.ripple.block.setshape.i64(i64 immarg, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare i64 @llvm.ripple.block.index.i64(ptr, i64 immarg) #2

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"Clang $LLVM_VERSION_MAJOR.$LLVM_VERSION_MINOR"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}