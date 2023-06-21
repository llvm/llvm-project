; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%class.float3 = type { float, float, float }

; Function Attrs: nounwind
; CHECK-LABEL: some_kernel
define void @some_kernel(ptr nocapture %dst) #0 {
_ZL11compute_vecRK6float3jb.exit:
  %ret_vec.sroa.8.i = alloca float, align 4
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %2 = mul nsw i32 %1, %0
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %4 = add nsw i32 %2, %3
  %5 = zext i32 %4 to i64
  call void @llvm.lifetime.start.p0(i64 4, ptr %ret_vec.sroa.8.i)
  %6 = and i32 %4, 15
  %7 = icmp eq i32 %6, 0
  %8 = select i1 %7, float 0.000000e+00, float -1.000000e+00
  store float %8, ptr %ret_vec.sroa.8.i, align 4
; CHECK: max.f32 %f{{[0-9]+}}, %f{{[0-9]+}}, 0f00000000
  %9 = fcmp olt float %8, 0.000000e+00
  %ret_vec.sroa.8.i.val = load float, ptr %ret_vec.sroa.8.i, align 4
  %10 = select i1 %9, float 0.000000e+00, float %ret_vec.sroa.8.i.val
  call void @llvm.lifetime.end.p0(i64 4, ptr %ret_vec.sroa.8.i)
  %11 = getelementptr inbounds %class.float3, ptr %dst, i64 %5, i32 0
  store float 0.000000e+00, ptr %11, align 4
  %12 = getelementptr inbounds %class.float3, ptr %dst, i64 %5, i32 1
  store float %10, ptr %12, align 4
  %13 = getelementptr inbounds %class.float3, ptr %dst, i64 %5, i32 2
  store float 0.000000e+00, ptr %13, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #2

; Function Attrs: nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #2

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "stack-protector-buffer-size"="8" "no-signed-zeros-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!nvvm.annotations = !{!0}
!llvm.ident = !{!1}

!0 = !{ptr @some_kernel, !"kernel", i32 1}
!1 = !{!"clang version 3.5.1 (tags/RELEASE_351/final)"}
