; RUN: opt -S -passes=infer-address-spaces %s | FileCheck %s

; IR from:
;   __global__ void kernel(int **X, int **Y, int x, int y) {
;     int a[10] = {0};
;     X[1] = a;
;     X[x][y] = Y[x][y];
;   }
;
; X[1] = a stores a local pointer into X, so pointer values loaded through X
; must not be inferred as global (even though the X argument itself is in param/global AS).
; Y is only read, so loads through readonly Y should infer global address space.
;
; %0 = X (writable), %1 = Y (readonly).

target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define dso_local ptx_kernel void @_Z6kernelPPiS0_ii(ptr noundef captures(none) initializes((8, 16)) %0, ptr noundef readonly captures(none) %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
; CHECK-LABEL: define dso_local ptx_kernel void @_Z6kernelPPiS0_ii(
;
; Read Y[x][y]: loads through readonly %1 infer global.
; CHECK: getelementptr inbounds [8 x i8], ptr addrspace(1) {{%.*}}, i64 {{%.*}}
; CHECK: [[Y_PTR1:%.*]] = load ptr, ptr addrspace(1) {{%.*}}, align 8
; CHECK: addrspacecast ptr [[Y_PTR1]] to ptr addrspace(1)
; CHECK: load i32, ptr addrspace(1) {{%.*}}, align 4
;
; Write X[x][y]: pointer loaded through writable %0 must stay generic.
; CHECK: [[X_PTR1:%.*]] = load ptr, ptr addrspace(1) {{%.*}}, align 8
; CHECK-NOT: addrspacecast ptr [[X_PTR1]] to ptr addrspace(1)
; CHECK: getelementptr inbounds [4 x i8], ptr [[X_PTR1]], i64 {{%.*}}
; CHECK: store i32 {{.*}}, ptr {{%.*}}, align 4
;
  %5 = alloca [10 x i32], align 4
  %6 = addrspacecast ptr %5 to ptr addrspace(5)
  %7 = addrspacecast ptr addrspace(5) %6 to ptr
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(40) %5, i8 0, i64 40, i1 false)
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store ptr %5, ptr %8, align 8, !tbaa !10
  %9 = sext i32 %2 to i64
  %10 = getelementptr inbounds [8 x i8], ptr %1, i64 %9
  %11 = load ptr, ptr %10, align 8, !tbaa !10
  %12 = sext i32 %3 to i64
  %13 = getelementptr inbounds [4 x i8], ptr %11, i64 %12
  %14 = load i32, ptr %13, align 4, !tbaa !5
  %15 = getelementptr inbounds [8 x i8], ptr %0, i64 %9
  %16 = load ptr, ptr %15, align 8, !tbaa !10
  %17 = getelementptr inbounds [4 x i8], ptr %16, i64 %12
  store i32 %14, ptr %17, align 4, !tbaa !5
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #3
  ret void
}

declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #2
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none, target_mem: none) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
attributes #3 = { nounwind }

!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"p1 int", !12, i64 0}
!12 = !{!"any pointer", !7, i64 0}
