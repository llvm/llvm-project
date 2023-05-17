; RUN: llc -O0 -opaque-pointers=0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: %[[#IntTyID:]] = OpTypeInt
; CHECK-SPIRV-DAG: %[[#VoidTyID:]] = OpTypeVoid
; CHECK-SPIRV-DAG: %[[#ImageTyID:]] = OpTypeImage %[[#VoidTyID]] 2D 0 1 0 0 Unknown ReadOnly
; CHECK-SPIRV-DAG: %[[#VectorTyID:]] = OpTypeVector %[[#IntTyID]] [[#]]
; CHECK-SPIRV:     %[[#ImageArgID:]] = OpFunctionParameter %[[#ImageTyID]]
; CHECK-SPIRV:     %[[#]] = OpImageQuerySizeLod %[[#VectorTyID]] %[[#ImageArgID]]

%opencl.image2d_array_ro_t = type opaque

define spir_kernel void @sample_kernel(%opencl.image2d_array_ro_t addrspace(1)* %input) {
entry:
  %call = call spir_func i32 @_Z15get_image_width20ocl_image2d_array_ro(%opencl.image2d_array_ro_t addrspace(1)* %input)
  ret void
}

declare spir_func i32 @_Z15get_image_width20ocl_image2d_array_ro(%opencl.image2d_array_ro_t addrspace(1)*)
