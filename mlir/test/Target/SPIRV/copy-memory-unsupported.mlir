// RUN: mlir-translate --no-implicit-module --serialize-spirv --split-input-file --verify-diagnostics %s

spirv.module Physical64 OpenCL requires #spirv.vce<v1.4, [Kernel, Addresses, Linkage], []> {
  spirv.func @unsupported_scope(%dst: !spirv.ptr<i32, Function>, %src: !spirv.ptr<i32, Function>, %size: i32) "None" {
    // expected-error @+1 {{unsupported memory access mask: MakePointerAvailable|NonPrivatePointer}}
    spirv.CopyMemorySized %dst, %src, %size ["MakePointerAvailable|NonPrivatePointer"] : !spirv.ptr<i32, Function>, !spirv.ptr<i32, Function>, i32
    spirv.Return
  }
}

// -----

spirv.module Physical64 OpenCL requires #spirv.vce<v1.4, [Kernel, Addresses, Linkage], []> {
  spirv.func @unsupported_source_alias(%dst: !spirv.ptr<i32, Function>, %src: !spirv.ptr<i32, Function>) "None" {
    // expected-error @+1 {{unsupported memory access mask: AliasScopeINTELMask}}
    spirv.CopyMemory "Function" %dst, "Function" %src ["None"], ["AliasScopeINTELMask"] : i32
    spirv.Return
  }
}
