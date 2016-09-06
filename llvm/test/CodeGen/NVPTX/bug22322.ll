; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%class.float3 = type { float, float, float }

; Function Attrs: nounwind
; CHECK-LABEL: some_kernel
define void @some_kernel(%class.float3* nocapture %dst) #0 {
_ZL11compute_vecRK6float3jb.exit:
  %ret_vec.sroa.8.i = alloca float, align 4
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %2 = mul nsw i32 %1, %0
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %4 = add nsw i32 %2, %3
  %5 = zext i32 %4 to i64
  %6 = bitcast float* %ret_vec.sroa.8.i to i8*
  call void @llvm.lifetime.start(i64 4, i8* %6)
  %7 = and i32 %4, 15
  %8 = icmp eq i32 %7, 0
  %9 = select i1 %8, float 0.000000e+00, float -1.000000e+00
  store float %9, float* %ret_vec.sroa.8.i, align 4
; CHECK: setp.lt.f32     %p{{[0-9]+}}, %f{{[0-9]+}}, 0f00000000
  %10 = fcmp olt float %9, 0.000000e+00
  %ret_vec.sroa.8.i.val = load float, float* %ret_vec.sroa.8.i, align 4
  %11 = select i1 %10, float 0.000000e+00, float %ret_vec.sroa.8.i.val
  call void @llvm.lifetime.end(i64 4, i8* %6)
  %12 = getelementptr inbounds %class.float3, %class.float3* %dst, i64 %5, i32 0
  store float 0.000000e+00, float* %12, align 4
  %13 = getelementptr inbounds %class.float3, %class.float3* %dst, i64 %5, i32 1
  store float %11, float* %13, align 4
  %14 = getelementptr inbounds %class.float3, %class.float3* %dst, i64 %5, i32 2
  store float 0.000000e+00, float* %14, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #2

; Function Attrs: nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #2

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!nvvm.annotations = !{!0}
!llvm.ident = !{!1}

!0 = !{void (%class.float3*)* @some_kernel, !"kernel", i32 1}
!1 = !{!"clang version 3.5.1 (tags/RELEASE_351/final)"}
