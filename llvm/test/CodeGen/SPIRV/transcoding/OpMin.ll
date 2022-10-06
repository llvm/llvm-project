; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: %[[#SetInstID:]] = OpExtInstImport "OpenCL.std"
; CHECK-SPIRV: %[[#IntTypeID:]] = OpTypeInt 32 [[#]]
; CHECK-SPIRV: %[[#Int2TypeID:]] = OpTypeVector %[[#IntTypeID]] 2
; CHECK-SPIRV: %[[#CompositeID:]] = OpCompositeInsert %[[#Int2TypeID]] %[[#]] %[[#]] [[#]]
; CHECK-SPIRV: %[[#ShuffleID:]] = OpVectorShuffle %[[#Int2TypeID]] %[[#CompositeID]] %[[#]] [[#]] [[#]]
; CHECK-SPIRV: %[[#]] = OpExtInst %[[#Int2TypeID]] %[[#SetInstID]] s_min %[[#]] %[[#ShuffleID]]

define spir_kernel void @test() {
entry:
  %call = tail call spir_func <2 x i32> @_Z3minDv2_ii(<2 x i32> <i32 1, i32 10>, i32 5) #2
  ret void
}

declare spir_func <2 x i32> @_Z3minDv2_ii(<2 x i32>, i32)
