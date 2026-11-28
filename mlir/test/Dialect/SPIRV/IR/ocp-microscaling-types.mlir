// RUN: mlir-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func private @all_microscaling_types
// CHECK-SAME: f4E2M1FN
// CHECK-SAME: f6E2M3FN
// CHECK-SAME: f6E3M2FN
// CHECK-SAME: f8E8M0FNU
func.func private @all_microscaling_types(f4E2M1FN, f6E2M3FN, f6E3M2FN,
                                          f8E8M0FNU)

// CHECK-LABEL: func.func @valid
// CHECK: spirv.BitcastExtractEXT %{{.*}}, %{{.*}} : i32, i32 to f4E2M1FN
// CHECK: spirv.BitcastExtractEXT %{{.*}}, %{{.*}} : i32, i32 to f6E2M3FN
// CHECK: spirv.BitcastExtractEXT %{{.*}}, %{{.*}} : i32, i32 to f6E3M2FN
// CHECK: spirv.BitcastExtractEXT %{{.*}}, %{{.*}} : i32, i32 to f8E8M0FNU
func.func @valid(%base: i32, %offset: i32) -> (f4E2M1FN, f6E2M3FN, f6E3M2FN, f8E8M0FNU) {
  %0 = spirv.BitcastExtractEXT %base, %offset : i32, i32 to f4E2M1FN
  %1 = spirv.BitcastExtractEXT %base, %offset : i32, i32 to f6E2M3FN
  %2 = spirv.BitcastExtractEXT %base, %offset : i32, i32 to f6E3M2FN
  %3 = spirv.BitcastExtractEXT %base, %offset : i32, i32 to f8E8M0FNU
  return %0, %1, %2, %3 : f4E2M1FN, f6E2M3FN, f6E3M2FN, f8E8M0FNU
}

// CHECK-LABEL: func.func @valid_vector
// CHECK: spirv.BitcastExtractEXT %{{.*}}, %{{.*}} : vector<2xi32>, i32 to vector<2xf4E2M1FN>
// CHECK: spirv.BitcastExtractEXT %{{.*}}, %{{.*}} : vector<2xi32>, i32 to vector<2xf6E2M3FN>
// CHECK: spirv.BitcastExtractEXT %{{.*}}, %{{.*}} : vector<2xi32>, i32 to vector<2xf6E3M2FN>
// CHECK: spirv.BitcastExtractEXT %{{.*}}, %{{.*}} : vector<2xi32>, i32 to vector<2xf8E8M0FNU>
func.func @valid_vector(%base: vector<2xi32>, %offset: i32) -> (vector<2xf4E2M1FN>, vector<2xf6E2M3FN>, vector<2xf6E3M2FN>, vector<2xf8E8M0FNU>) {
  %0 = spirv.BitcastExtractEXT %base, %offset : vector<2xi32>, i32 to vector<2xf4E2M1FN>
  %1 = spirv.BitcastExtractEXT %base, %offset : vector<2xi32>, i32 to vector<2xf6E2M3FN>
  %2 = spirv.BitcastExtractEXT %base, %offset : vector<2xi32>, i32 to vector<2xf6E3M2FN>
  %3 = spirv.BitcastExtractEXT %base, %offset : vector<2xi32>, i32 to vector<2xf8E8M0FNU>
  return %0, %1, %2, %3 : vector<2xf4E2M1FN>, vector<2xf6E2M3FN>, vector<2xf6E3M2FN>, vector<2xf8E8M0FNU>
}

// -----

// CHECK-LABEL: func.func @allowed_storage
// CHECK-SAME: !spirv.ptr<!spirv.array<4 x f4E2M1FN>, Private>
// CHECK-SAME: !spirv.ptr<f6E2M3FN, Function>
// CHECK-SAME: !spirv.ptr<f6E3M2FN, Private>
// CHECK-SAME: !spirv.ptr<f8E8M0FNU, StorageBuffer>
func.func @allowed_storage(%private: !spirv.ptr<!spirv.array<4xf4E2M1FN>, Private>,
                           %fp6a: !spirv.ptr<f6E2M3FN, Function>,
                           %fp6b: !spirv.ptr<f6E3M2FN, Private>,
                           %scale: !spirv.ptr<f8E8M0FNU, StorageBuffer>) {
  return
}

// -----

// CHECK: spirv.VectorExtractDynamic %{{.*}} : vector<2xf4E2M1FN>, i32
func.func @allowed_composite(%vector: vector<2xf4E2M1FN>, %index: i32) -> f4E2M1FN {
  %0 = spirv.VectorExtractDynamic %vector[%index] : vector<2xf4E2M1FN>, i32
  return %0 : f4E2M1FN
}
