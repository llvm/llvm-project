; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV-NOT: SConvert

; Check no OpenCL convert builtin function generated
; CHECK-LLVM-NOT: call {{.*}} @_Z{{[0-9]*}}convert

; Function Attrs: nounwind
define spir_kernel void @math_kernel8(<8 x i32> addrspace(1)* nocapture %out, <8 x float> addrspace(1)* nocapture readonly %in1, <8 x float> addrspace(1)* nocapture readonly %in2) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !5 !kernel_arg_type_qual !4 {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0) #2
  %sext = shl i64 %call, 32
  %idxprom = ashr exact i64 %sext, 32
  %arrayidx = getelementptr inbounds <8 x float>, <8 x float> addrspace(1)* %in1, i64 %idxprom
  %0 = load <8 x float>, <8 x float> addrspace(1)* %arrayidx, align 32, !tbaa !9
  %arrayidx2 = getelementptr inbounds <8 x float>, <8 x float> addrspace(1)* %in2, i64 %idxprom
  %1 = load <8 x float>, <8 x float> addrspace(1)* %arrayidx2, align 32, !tbaa !9
  %call3 = tail call spir_func <8 x i32> @_Z7isequalDv8_fDv8_f(<8 x float> %0, <8 x float> %1) #2
  %arrayidx5 = getelementptr inbounds <8 x i32>, <8 x i32> addrspace(1)* %out, i64 %idxprom
  store <8 x i32> %call3, <8 x i32> addrspace(1)* %arrayidx5, align 32, !tbaa !9
  ret void
}

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z13get_global_idj(i32) #1

; Function Attrs: nounwind readnone
declare spir_func <8 x i32> @_Z7isequalDv8_fDv8_f(<8 x float>, <8 x float>) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!6}
!opencl.used.extensions = !{!7}
!opencl.used.optional.core.features = !{!7}
!opencl.compiler.options = !{!7}
!llvm.ident = !{!8}

!1 = !{i32 1, i32 1, i32 1}
!2 = !{!"none", !"none", !"none"}
!3 = !{!"int8*", !"float8*", !"float8*"}
!4 = !{!"", !"", !""}
!5 = !{!"int8*", !"float8*", !"float8*"}
!6 = !{i32 2, i32 0}
!7 = !{}
!8 = !{!"clang version 3.4 "}
!9 = !{!10, !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
