; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; NOTE: access qualifier infomation is not preserved after round-trip conversion to LLVM
; CHECK-LLVM: call spir_func <4 x float> @_Z11read_imagef14ocl_image1d_rw11ocl_sampleri(%opencl.image1d_rw_t

; CHECK-SPIRV-DAG: 2 Capability ImageBasic
; CHECK-SPIRV-DAG: 2 Capability ImageReadWrite
; CHECK-SPIRV-DAG: 2 Capability LiteralSampler
; CHECK-SPIRV-DAG: 10 TypeImage [[TyImageID:[0-9]+]] 2 0 0 0 0 0 0 2 
; CHECK-SPIRV-DAG: 3 TypeSampledImage [[TySampledImageID:[0-9]+]] [[TyImageID]]

; CHECK-SPIRV-DAG: 5 SampledImage [[TySampledImageID]] [[ResID:[0-9]+]]
; CHECK-SPIRV: 7 ImageSampleExplicitLod {{[0-9]+}} {{[0-9]+}} [[ResID]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%opencl.image1d_rw_t = type opaque

; Function Attrs: nounwind
define spir_func void @sampFun(%opencl.image1d_rw_t addrspace(1)* %image) #0 {
entry:
  %image.addr = alloca %opencl.image1d_rw_t addrspace(1)*, align 4
  store %opencl.image1d_rw_t addrspace(1)* %image, %opencl.image1d_rw_t addrspace(1)** %image.addr, align 4
  %0 = load %opencl.image1d_rw_t addrspace(1)*, %opencl.image1d_rw_t addrspace(1)** %image.addr, align 4
  %call = call spir_func <4 x float> @_Z11read_imagef14ocl_image1d_rw11ocl_sampleri(%opencl.image1d_rw_t addrspace(1)* %0, i32 8, i32 2) #2
  ret void
}

; Function Attrs: nounwind readnone
declare spir_func <4 x float> @_Z11read_imagef14ocl_image1d_rw11ocl_sampleri(%opencl.image1d_rw_t addrspace(1)*, i32, i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}

!0 = !{i32 1, i32 2}
!1 = !{i32 2, i32 0}
!2 = !{}
