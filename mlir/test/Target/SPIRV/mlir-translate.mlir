// Check that `--spirv-save-validation-files-with-prefix` generates
// a correct number of files.

// REQUIRES: shell
// RUN: rm -rf %t
// RUN: mkdir %t && mlir-translate --serialize-spirv --no-implicit-module \
// RUN: --split-input-file --spirv-save-validation-files-with-prefix=%t/foo %s \
// RUN: && ls %t | wc -l | FileCheck %s
// RUN: rm -rf %t

// CHECK: 4

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
}

// -----

spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader, Linkage], []> {
}
