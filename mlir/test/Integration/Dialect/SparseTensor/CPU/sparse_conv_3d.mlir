// DEFINE: %{option} = "enable-index-reduction=true enable-runtime-library=true"
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{option} = "enable-runtime-library=false enable-buffer-initialization=true enable-index-reduction=true"
// RUN: %{compile} | %{run}
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false enable-buffer-initialization=true vl=2 reassociate-fp-reductions=true enable-index-optimizations=true enable-index-reduction=true"
// RUN: %{compile} | %{run}

// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=4 enable-arm-sve=%ENABLE_VLA enable-index-reduction=true"
// REDEFINE: %{run} = %lli \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

#CCC = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed", "compressed" ]
}>

#CDC = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "dense", "compressed" ]
}>

#DDC = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed", "compressed" ]
}>

// Creates and returns 3-D buffer of size (%s1, %s2, %s3) filled with the value %f
func.func @alloc_3d_filled_f32(%s1 : index, %s2 : index, %s3 : index, %f : f32) -> tensor<?x?x?xf32> {
  %buf = bufferization.alloc_tensor(%s1, %s2, %s3) : tensor<?x?x?xf32>
  %ret = linalg.fill ins(%f : f32) outs(%buf : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %ret : tensor<?x?x?xf32>
}

func.func @conv_3d(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %ret = linalg.conv_3d
     ins (%arg0, %arg1: tensor<?x?x?xf32>, tensor<?x?x?xf32>)
    outs (%arg2: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %ret : tensor<?x?x?xf32>
}

func.func @conv_3d_CCC(%arg0: tensor<?x?x?xf32, #CCC>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32, #CCC> {
  %c6 = arith.constant 6 : index
  %s = bufferization.alloc_tensor(%c6, %c6, %c6) : tensor<?x?x?xf32, #CCC>
  %ret = linalg.conv_3d
     ins (%arg0, %arg1: tensor<?x?x?xf32, #CCC>, tensor<?x?x?xf32>)
    outs (%s: tensor<?x?x?xf32, #CCC>) -> tensor<?x?x?xf32, #CCC>
  return %ret : tensor<?x?x?xf32, #CCC>
}

func.func @conv_3d_CDC(%arg0: tensor<?x?x?xf32, #CDC>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32, #CDC> {
  %c6 = arith.constant 6 : index
  %s = bufferization.alloc_tensor(%c6, %c6, %c6) : tensor<?x?x?xf32, #CDC>
  %ret = linalg.conv_3d
     ins (%arg0, %arg1: tensor<?x?x?xf32, #CDC>, tensor<?x?x?xf32>)
    outs (%s: tensor<?x?x?xf32, #CDC>) -> tensor<?x?x?xf32, #CDC>
  return %ret : tensor<?x?x?xf32, #CDC>
}

func.func @conv_3d_DDC(%arg0: tensor<?x?x?xf32, #DDC>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32, #DDC> {
  %c6 = arith.constant 6 : index
  %s = bufferization.alloc_tensor(%c6, %c6, %c6) : tensor<?x?x?xf32, #DDC>
  %ret = linalg.conv_3d
     ins (%arg0, %arg1: tensor<?x?x?xf32, #DDC>, tensor<?x?x?xf32>)
    outs (%s: tensor<?x?x?xf32, #DDC>) -> tensor<?x?x?xf32, #DDC>
  return %ret : tensor<?x?x?xf32, #DDC>
}

func.func @entry() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %c8 = arith.constant 8 : index
  %f10 = arith.constant 10.00000e+00 : f32
  %val = arith.constant 2.00000e+00 : f32
  %zero = arith.constant 0.00000e+00 : f32

  %filter3D = call @alloc_3d_filled_f32(%c3, %c3, %c3, %val) : (index, index, index, f32) -> (tensor<?x?x?xf32>)
  %in3D_tmp = call @alloc_3d_filled_f32(%c8, %c8, %c8, %val) : (index, index, index, f32) -> (tensor<?x?x?xf32>)
  %in3D = tensor.insert %f10 into %in3D_tmp[%c0, %c3, %c0] : tensor<?x?x?xf32>
  %out3D = call @alloc_3d_filled_f32(%c6, %c6, %c6, %zero) : (index, index, index, f32) -> (tensor<?x?x?xf32>)

  %in3D_CCC = sparse_tensor.convert %in3D
    : tensor<?x?x?xf32> to tensor<?x?x?xf32, #CCC>
  %in3D_CDC = sparse_tensor.convert %in3D
    : tensor<?x?x?xf32> to tensor<?x?x?xf32, #CDC>
  %in3D_DDC = sparse_tensor.convert %in3D
    : tensor<?x?x?xf32> to tensor<?x?x?xf32, #DDC>

  %dense_ret = call @conv_3d(%in3D, %filter3D, %out3D) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> (tensor<?x?x?xf32>)
  %CCC_ret = call @conv_3d_CCC(%in3D_CCC, %filter3D) : (tensor<?x?x?xf32, #CCC>, tensor<?x?x?xf32>) -> (tensor<?x?x?xf32, #CCC>)
  %CDC_ret = call @conv_3d_CDC(%in3D_CDC, %filter3D) : (tensor<?x?x?xf32, #CDC>, tensor<?x?x?xf32>) -> (tensor<?x?x?xf32, #CDC>)
  %DDC_ret = call @conv_3d_DDC(%in3D_DDC, %filter3D) : (tensor<?x?x?xf32, #DDC>, tensor<?x?x?xf32>) -> (tensor<?x?x?xf32, #DDC>)

  //      CHECK:( ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 124, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 124, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 124, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ),
  // CHECK-SAME:  ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ),
  // CHECK-SAME:  ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ),
  // CHECK-SAME:  ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ),
  // CHECK-SAME:  ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ),
  // CHECK-SAME:  ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ) )
  %dense_v = vector.transfer_read %dense_ret[%c0, %c0, %c0], %zero
      : tensor<?x?x?xf32>, vector<6x6x6xf32>
  vector.print %dense_v : vector<6x6x6xf32>

  // CHECK-NEXT:( ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 124, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 124, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 124, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ),
  // CHECK-SAME:  ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ),
  // CHECK-SAME:  ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ),
  // CHECK-SAME:  ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ),
  // CHECK-SAME:  ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ),
  // CHECK-SAME:  ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ) )
  %1 = sparse_tensor.convert %CCC_ret
    : tensor<?x?x?xf32, #CCC> to tensor<?x?x?xf32>
  %v1 = vector.transfer_read %1[%c0, %c0, %c0], %zero
      : tensor<?x?x?xf32>, vector<6x6x6xf32>
  vector.print %v1 : vector<6x6x6xf32>

  // CHECK-NEXT:( ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 124, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 124, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 124, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ),
  // CHECK-SAME:  ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ),
  // CHECK-SAME:  ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ),
  // CHECK-SAME:  ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ),
  // CHECK-SAME:  ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ),
  // CHECK-SAME:  ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ) )
  %2 = sparse_tensor.convert %CCC_ret
    : tensor<?x?x?xf32, #CCC> to tensor<?x?x?xf32>
  %v2 = vector.transfer_read %2[%c0, %c0, %c0], %zero
      : tensor<?x?x?xf32>, vector<6x6x6xf32>
  vector.print %v2 : vector<6x6x6xf32>

  // CHECK-NEXT:( ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 124, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 124, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 124, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ),
  // CHECK-SAME:  ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ),
  // CHECK-SAME:  ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ),
  // CHECK-SAME:  ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ),
  // CHECK-SAME:  ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ),
  // CHECK-SAME:  ( ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ),
  // CHECK-SAME:    ( 108, 108, 108, 108, 108, 108 ) ) )
  %3 = sparse_tensor.convert %DDC_ret
    : tensor<?x?x?xf32, #DDC> to tensor<?x?x?xf32>
  %v3 = vector.transfer_read %3[%c0, %c0, %c0], %zero
      : tensor<?x?x?xf32>, vector<6x6x6xf32>
  vector.print %v2 : vector<6x6x6xf32>

  // Free the resources
  bufferization.dealloc_tensor %in3D : tensor<?x?x?xf32>
  bufferization.dealloc_tensor %filter3D : tensor<?x?x?xf32>
  bufferization.dealloc_tensor %out3D : tensor<?x?x?xf32>

  bufferization.dealloc_tensor %in3D_CDC : tensor<?x?x?xf32, #CDC>
  bufferization.dealloc_tensor %in3D_CCC : tensor<?x?x?xf32, #CCC>
  bufferization.dealloc_tensor %in3D_DDC : tensor<?x?x?xf32, #DDC>

  bufferization.dealloc_tensor %CCC_ret : tensor<?x?x?xf32, #CCC>
  bufferization.dealloc_tensor %CDC_ret : tensor<?x?x?xf32, #CDC>
  bufferization.dealloc_tensor %DDC_ret : tensor<?x?x?xf32, #DDC>
  return
}
