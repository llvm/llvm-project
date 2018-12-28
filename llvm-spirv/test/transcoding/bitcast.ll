; ModuleID = 'bitcast.bc'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s

; Check the bitcast is translated back to bitcast

; CHECK: bitcast

; Function Attrs: nounwind
define spir_kernel void @test_fn(<2 x i8> addrspace(1)* nocapture readonly %src, i16 addrspace(1)* nocapture %dst) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !5 !kernel_arg_type_qual !4 {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0) #2
  %sext = shl i64 %call, 32
  %idxprom = ashr exact i64 %sext, 32
  %arrayidx = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(1)* %src, i64 %idxprom
  %0 = load <2 x i8>, <2 x i8> addrspace(1)* %arrayidx, align 2, !tbaa !9
  %astype = bitcast <2 x i8> %0 to i16
  %arrayidx2 = getelementptr inbounds i16, i16 addrspace(1)* %dst, i64 %idxprom
  store i16 %astype, i16 addrspace(1)* %arrayidx2, align 2, !tbaa !12
  ret void
}

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z13get_global_idj(i32) #1

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

!1 = !{i32 1, i32 1}
!2 = !{!"none", !"none"}
!3 = !{!"char2*", !"short*"}
!4 = !{!"", !""}
!5 = !{!"char2*", !"short*"}
!6 = !{i32 2, i32 0}
!7 = !{}
!8 = !{!"clang version 3.4 "}
!9 = !{!10, !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!13, !13, i64 0}
!13 = !{!"short", !10, i64 0}
