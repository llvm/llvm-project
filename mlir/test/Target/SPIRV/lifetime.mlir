// RUN: mlir-translate -no-implicit-module -test-spirv-roundtrip %s | FileCheck %s

// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

spirv.module Physical64 OpenCL requires #spirv.vce<v1.0, [Kernel, Addresses, Int8, Linkage], []> {
  spirv.func @lifetime() "None" {
    %0 = spirv.Variable : !spirv.ptr<f32, Function>
    // CHECK: spirv.LifetimeStart %{{.*}}, 0 : !spirv.ptr<f32, Function>
    spirv.LifetimeStart %0, 0 : !spirv.ptr<f32, Function>
    // CHECK: spirv.LifetimeStop %{{.*}}, 0 : !spirv.ptr<f32, Function>
    spirv.LifetimeStop %0, 0 : !spirv.ptr<f32, Function>

    %1 = spirv.Bitcast %0 : !spirv.ptr<f32, Function> to !spirv.ptr<i8, Function>
    // CHECK: spirv.LifetimeStart %{{.*}}, 0 : !spirv.ptr<i8, Function>
    spirv.LifetimeStart %1, 0 : !spirv.ptr<i8, Function>
    // CHECK: spirv.LifetimeStop %{{.*}}, 0 : !spirv.ptr<i8, Function>
    spirv.LifetimeStop %1, 0 : !spirv.ptr<i8, Function>
    spirv.Return
  }
}
