; RUN: mlir-translate -import-llvm %s | FileCheck %s

; CHECK-LABEL: llvm.mlir.global external @global() {addr_space = 0 : i32}
; CHECK-SAME:    !llvm.target<"spirv.DeviceEvent">
; CHECK-NEXT:      %0 = llvm.mlir.zero : !llvm.target<"spirv.DeviceEvent">
; CHECK-NEXT:      llvm.return %0 : !llvm.target<"spirv.DeviceEvent">
@global = global target("spirv.DeviceEvent") zeroinitializer

; CHECK-LABEL: llvm.func spir_kernelcc @func1(
define spir_kernel void @func1(
  ; CHECK-SAME: %arg0: !llvm.target<"spirv.Pipe", 0>
  target("spirv.Pipe", 0) %a,
  ; CHECK-SAME:    %arg1: !llvm.target<"spirv.Pipe", 1>
  target("spirv.Pipe", 1) %b,
  ; CHECK-SAME:    %arg2: !llvm.target<"spirv.Image", !llvm.void, 0, 0, 0, 0, 0, 0, 0>
  target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %c1,
  ; CHECK-SAME:    %arg3: !llvm.target<"spirv.Image", i32, 1, 0, 0, 0, 0, 0, 0>
  target("spirv.Image", i32, 1, 0, 0, 0, 0, 0, 0) %d1,
  ; CHECK-SAME:    %arg4: !llvm.target<"spirv.Image", i32, 2, 0, 0, 0, 0, 0, 0>
  target("spirv.Image", i32, 2, 0, 0, 0, 0, 0, 0) %e1,
  ; CHECK-SAME:    %arg5: !llvm.target<"spirv.Image", f16, 1, 0, 1, 0, 0, 0, 0>
  target("spirv.Image", half, 1, 0, 1, 0, 0, 0, 0) %f1,
  ; CHECK-SAME:    %arg6: !llvm.target<"spirv.Image", f32, 5, 0, 0, 0, 0, 0, 0>
  target("spirv.Image", float, 5, 0, 0, 0, 0, 0, 0) %g1,
  ; CHECK-SAME:    %arg7: !llvm.target<"spirv.Image", !llvm.void, 0, 0, 0, 0, 0, 0, 1>
  target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 1) %c2,
  ; CHECK-SAME:    %arg8: !llvm.target<"spirv.Image", !llvm.void, 1, 0, 0, 0, 0, 0, 2>)
  target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 2) %d3) {
entry:
  ret void
}

; CHECK-LABEL: llvm.func @func2()
; CHECK-SAME:      !llvm.target<"spirv.Event"> {  
define target("spirv.Event") @func2() {
  ; CHECK-NEXT:    %0 = llvm.mlir.constant(1 : i32) : i32
  ; CHECK-NEXT:    %1 = llvm.mlir.poison : !llvm.target<"spirv.Event">
  ; CHECK-NEXT:    %2 = llvm.alloca %0 x !llvm.target<"spirv.Event"> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %mem = alloca target("spirv.Event")
  ; CHECK-NEXT:    %3 = ptr.load %2 {alignment = 8 : i64} : !llvm.ptr -> !llvm.target<"spirv.Event">
  %val = load target("spirv.Event"), ptr %mem
  ; CHECK-NEXT:    llvm.return %1 : !llvm.target<"spirv.Event">
  ret target("spirv.Event") poison
}

; CHECK-LABEL: llvm.func @func3()
define void @func3() {
  ; CHECK-NEXT:    %0 = llvm.mlir.zero : !llvm.target<"spirv.DeviceEvent">
  ; CHECK-NEXT:    %1 = llvm.freeze %0 : !llvm.target<"spirv.DeviceEvent">
  %val = freeze target("spirv.DeviceEvent") zeroinitializer
  ; CHECK-NEXT:    llvm.return
  ret void
}
