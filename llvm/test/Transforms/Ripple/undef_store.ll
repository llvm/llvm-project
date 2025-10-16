; RUN: opt -passes='module(function(mem2reg,mergereturn),ripple,function(dce))' -S < %s | FileCheck %s --implicit-check-not="warning:"
define dso_local void @foo(ptr noundef %aptr) #0 {
entry:
  %aptr.addr = alloca ptr, align 8
  %v0 = alloca i64, align 8
  %x = alloca float, align 4
  store ptr %aptr, ptr %aptr.addr, align 8, !tbaa !2
  %BS = call ptr @llvm.ripple.block.setshape.i64(i64 0, i64 8, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1)
  call void @llvm.lifetime.start.p0(i64 8, ptr %v0) #4
  %0 = call i64 @llvm.ripple.block.index.i64(ptr %BS, i64 0)
  store i64 %0, ptr %v0, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(i64 4, ptr %x) #4
  store float undef, ptr %x, align 4, !tbaa !8
  %1 = load float, ptr %x, align 4, !tbaa !8
  %2 = load ptr, ptr %aptr.addr, align 8, !tbaa !2
  %3 = load i64, ptr %v0, align 8, !tbaa !6
  %arrayidx = getelementptr inbounds float, ptr %2, i64 %3
; CHECK-LABEL:  void @foo(ptr
; CHECK-NEXT: entry:
; CHECK-NEXT: ret void
  store float %1, ptr %arrayidx, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0(i64 4, ptr %x) #4
  call void @llvm.lifetime.end.p0(i64 8, ptr %v0) #4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare ptr @llvm.ripple.block.setshape.i64(i64 immarg, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare i64 @llvm.ripple.block.index.i64(ptr, i64 immarg) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2

attributes #0 = { nounwind "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read) }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"Clang $LLVM_VERSION_MAJOR.$LLVM_VERSION_MINOR"}
!2 = !{!3, !3, i64 0}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long", !4, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"float", !4, i64 0}

