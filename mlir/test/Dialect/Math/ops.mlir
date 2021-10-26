// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

// CHECK-LABEL: func @atan(
// CHECK-SAME:             %[[F:.*]]: f32, %[[V:.*]]: vector<4xf32>, %[[T:.*]]: tensor<4x4x?xf32>)
func @atan(%f: f32, %v: vector<4xf32>, %t: tensor<4x4x?xf32>) {
  // CHECK: %{{.*}} = math.atan %[[F]] : f32
  %0 = math.atan %f : f32
  // CHECK: %{{.*}} = math.atan %[[V]] : vector<4xf32>
  %1 = math.atan %v : vector<4xf32>
  // CHECK: %{{.*}} = math.atan %[[T]] : tensor<4x4x?xf32>
  %2 = math.atan %t : tensor<4x4x?xf32>
  return
}


// CHECK-LABEL: func @atan2(
// CHECK-SAME:             %[[F:.*]]: f32, %[[V:.*]]: vector<4xf32>, %[[T:.*]]: tensor<4x4x?xf32>)
func @atan2(%f: f32, %v: vector<4xf32>, %t: tensor<4x4x?xf32>) {
  // CHECK: %{{.*}} = math.atan2 %[[F]], %[[F]] : f32
  %0 = math.atan2 %f, %f : f32
  // CHECK: %{{.*}} = math.atan2 %[[V]], %[[V]] : vector<4xf32>
  %1 = math.atan2 %v, %v : vector<4xf32>
  // CHECK: %{{.*}} = math.atan2 %[[T]], %[[T]] : tensor<4x4x?xf32>
  %2 = math.atan2 %t, %t : tensor<4x4x?xf32>
  return
}

// CHECK-LABEL: func @cos(
// CHECK-SAME:            %[[F:.*]]: f32, %[[V:.*]]: vector<4xf32>, %[[T:.*]]: tensor<4x4x?xf32>)
func @cos(%f: f32, %v: vector<4xf32>, %t: tensor<4x4x?xf32>) {
  // CHECK: %{{.*}} = math.cos %[[F]] : f32
  %0 = math.cos %f : f32
  // CHECK: %{{.*}} = math.cos %[[V]] : vector<4xf32>
  %1 = math.cos %v : vector<4xf32>
  // CHECK: %{{.*}} = math.cos %[[T]] : tensor<4x4x?xf32>
  %2 = math.cos %t : tensor<4x4x?xf32>
  return
}

// CHECK-LABEL: func @sin(
// CHECK-SAME:            %[[F:.*]]: f32, %[[V:.*]]: vector<4xf32>, %[[T:.*]]: tensor<4x4x?xf32>)
func @sin(%f: f32, %v: vector<4xf32>, %t: tensor<4x4x?xf32>) {
  // CHECK: %{{.*}} = math.sin %[[F]] : f32
  %0 = math.sin %f : f32
  // CHECK: %{{.*}} = math.sin %[[V]] : vector<4xf32>
  %1 = math.sin %v : vector<4xf32>
  // CHECK: %{{.*}} = math.sin %[[T]] : tensor<4x4x?xf32>
  %2 = math.sin %t : tensor<4x4x?xf32>
  return
}

// CHECK-LABEL: func @erf(
// CHECK-SAME:            %[[F:.*]]: f32, %[[V:.*]]: vector<4xf32>, %[[T:.*]]: tensor<4x4x?xf32>)
func @erf(%f: f32, %v: vector<4xf32>, %t: tensor<4x4x?xf32>) {
  // CHECK: %{{.*}} = math.erf %[[F]] : f32
  %0 = math.erf %f : f32
  // CHECK: %{{.*}} = math.erf %[[V]] : vector<4xf32>
  %1 = math.erf %v : vector<4xf32>
  // CHECK: %{{.*}} = math.erf %[[T]] : tensor<4x4x?xf32>
  %2 = math.erf %t : tensor<4x4x?xf32>
  return
}

// CHECK-LABEL: func @exp(
// CHECK-SAME:            %[[F:.*]]: f32, %[[V:.*]]: vector<4xf32>, %[[T:.*]]: tensor<4x4x?xf32>)
func @exp(%f: f32, %v: vector<4xf32>, %t: tensor<4x4x?xf32>) {
  // CHECK: %{{.*}} = math.exp %[[F]] : f32
  %0 = math.exp %f : f32
  // CHECK: %{{.*}} = math.exp %[[V]] : vector<4xf32>
  %1 = math.exp %v : vector<4xf32>
  // CHECK: %{{.*}} = math.exp %[[T]] : tensor<4x4x?xf32>
  %2 = math.exp %t : tensor<4x4x?xf32>
  return
}

// CHECK-LABEL: func @exp2(
// CHECK-SAME:             %[[F:.*]]: f32, %[[V:.*]]: vector<4xf32>, %[[T:.*]]: tensor<4x4x?xf32>)
func @exp2(%f: f32, %v: vector<4xf32>, %t: tensor<4x4x?xf32>) {
  // CHECK: %{{.*}} = math.exp2 %[[F]] : f32
  %0 = math.exp2 %f : f32
  // CHECK: %{{.*}} = math.exp2 %[[V]] : vector<4xf32>
  %1 = math.exp2 %v : vector<4xf32>
  // CHECK: %{{.*}} = math.exp2 %[[T]] : tensor<4x4x?xf32>
  %2 = math.exp2 %t : tensor<4x4x?xf32>
  return
}

// CHECK-LABEL: func @expm1(
// CHECK-SAME:            %[[F:.*]]: f32, %[[V:.*]]: vector<4xf32>, %[[T:.*]]: tensor<4x4x?xf32>)
func @expm1(%f: f32, %v: vector<4xf32>, %t: tensor<4x4x?xf32>) {
  // CHECK: %{{.*}} = math.expm1 %[[F]] : f32
  %0 = math.expm1 %f : f32
  // CHECK: %{{.*}} = math.expm1 %[[V]] : vector<4xf32>
  %1 = math.expm1 %v : vector<4xf32>
  // CHECK: %{{.*}} = math.expm1 %[[T]] : tensor<4x4x?xf32>
  %2 = math.expm1 %t : tensor<4x4x?xf32>
  return
}

// CHECK-LABEL: func @log(
// CHECK-SAME:            %[[F:.*]]: f32, %[[V:.*]]: vector<4xf32>, %[[T:.*]]: tensor<4x4x?xf32>)
func @log(%f: f32, %v: vector<4xf32>, %t: tensor<4x4x?xf32>) {
  // CHECK: %{{.*}} = math.log %[[F]] : f32
  %0 = math.log %f : f32
  // CHECK: %{{.*}} = math.log %[[V]] : vector<4xf32>
  %1 = math.log %v : vector<4xf32>
  // CHECK: %{{.*}} = math.log %[[T]] : tensor<4x4x?xf32>
  %2 = math.log %t : tensor<4x4x?xf32>
  return
}

// CHECK-LABEL: func @log10(
// CHECK-SAME:              %[[F:.*]]: f32, %[[V:.*]]: vector<4xf32>, %[[T:.*]]: tensor<4x4x?xf32>)
func @log10(%f: f32, %v: vector<4xf32>, %t: tensor<4x4x?xf32>) {
  // CHECK: %{{.*}} = math.log10 %[[F]] : f32
  %0 = math.log10 %f : f32
  // CHECK: %{{.*}} = math.log10 %[[V]] : vector<4xf32>
  %1 = math.log10 %v : vector<4xf32>
  // CHECK: %{{.*}} = math.log10 %[[T]] : tensor<4x4x?xf32>
  %2 = math.log10 %t : tensor<4x4x?xf32>
  return
}

// CHECK-LABEL: func @log1p(
// CHECK-SAME:              %[[F:.*]]: f32, %[[V:.*]]: vector<4xf32>, %[[T:.*]]: tensor<4x4x?xf32>)
func @log1p(%f: f32, %v: vector<4xf32>, %t: tensor<4x4x?xf32>) {
  // CHECK: %{{.*}} = math.log1p %[[F]] : f32
  %0 = math.log1p %f : f32
  // CHECK: %{{.*}} = math.log1p %[[V]] : vector<4xf32>
  %1 = math.log1p %v : vector<4xf32>
  // CHECK: %{{.*}} = math.log1p %[[T]] : tensor<4x4x?xf32>
  %2 = math.log1p %t : tensor<4x4x?xf32>
  return
}

// CHECK-LABEL: func @log2(
// CHECK-SAME:             %[[F:.*]]: f32, %[[V:.*]]: vector<4xf32>, %[[T:.*]]: tensor<4x4x?xf32>)
func @log2(%f: f32, %v: vector<4xf32>, %t: tensor<4x4x?xf32>) {
  // CHECK: %{{.*}} = math.log2 %[[F]] : f32
  %0 = math.log2 %f : f32
  // CHECK: %{{.*}} = math.log2 %[[V]] : vector<4xf32>
  %1 = math.log2 %v : vector<4xf32>
  // CHECK: %{{.*}} = math.log2 %[[T]] : tensor<4x4x?xf32>
  %2 = math.log2 %t : tensor<4x4x?xf32>
  return
}

// CHECK-LABEL: func @powf(
// CHECK-SAME:             %[[F:.*]]: f32, %[[V:.*]]: vector<4xf32>, %[[T:.*]]: tensor<4x4x?xf32>)
func @powf(%f: f32, %v: vector<4xf32>, %t: tensor<4x4x?xf32>) {
  // CHECK: %{{.*}} = math.powf %[[F]], %[[F]] : f32
  %0 = math.powf %f, %f : f32
  // CHECK: %{{.*}} = math.powf %[[V]], %[[V]] : vector<4xf32>
  %1 = math.powf %v, %v : vector<4xf32>
  // CHECK: %{{.*}} = math.powf %[[T]], %[[T]] : tensor<4x4x?xf32>
  %2 = math.powf %t, %t : tensor<4x4x?xf32>
  return
}

// CHECK-LABEL: func @rsqrt(
// CHECK-SAME:              %[[F:.*]]: f32, %[[V:.*]]: vector<4xf32>, %[[T:.*]]: tensor<4x4x?xf32>)
func @rsqrt(%f: f32, %v: vector<4xf32>, %t: tensor<4x4x?xf32>) {
  // CHECK: %{{.*}} = math.rsqrt %[[F]] : f32
  %0 = math.rsqrt %f : f32
  // CHECK: %{{.*}} = math.rsqrt %[[V]] : vector<4xf32>
  %1 = math.rsqrt %v : vector<4xf32>
  // CHECK: %{{.*}} = math.rsqrt %[[T]] : tensor<4x4x?xf32>
  %2 = math.rsqrt %t : tensor<4x4x?xf32>
  return
}


// CHECK-LABEL: func @sqrt(
// CHECK-SAME:             %[[F:.*]]: f32, %[[V:.*]]: vector<4xf32>, %[[T:.*]]: tensor<4x4x?xf32>)
func @sqrt(%f: f32, %v: vector<4xf32>, %t: tensor<4x4x?xf32>) {
  // CHECK: %{{.*}} = math.sqrt %[[F]] : f32
  %0 = math.sqrt %f : f32
  // CHECK: %{{.*}} = math.sqrt %[[V]] : vector<4xf32>
  %1 = math.sqrt %v : vector<4xf32>
  // CHECK: %{{.*}} = math.sqrt %[[T]] : tensor<4x4x?xf32>
  %2 = math.sqrt %t : tensor<4x4x?xf32>
  return
}

// CHECK-LABEL: func @tanh(
// CHECK-SAME:             %[[F:.*]]: f32, %[[V:.*]]: vector<4xf32>, %[[T:.*]]: tensor<4x4x?xf32>)
func @tanh(%f: f32, %v: vector<4xf32>, %t: tensor<4x4x?xf32>) {
  // CHECK: %{{.*}} = math.tanh %[[F]] : f32
  %0 = math.tanh %f : f32
  // CHECK: %{{.*}} = math.tanh %[[V]] : vector<4xf32>
  %1 = math.tanh %v : vector<4xf32>
  // CHECK: %{{.*}} = math.tanh %[[T]] : tensor<4x4x?xf32>
  %2 = math.tanh %t : tensor<4x4x?xf32>
  return
}
