// DEFINE: %{option} = enable-runtime-library=true
// DEFINE: %{compile} = mlir-opt %s --part-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext,%mlir_lib_dir/libmlir_runner_utils%shlibext | \
// DEFINE: FileCheck %s
// RUN: %{compile} | %{run}

#SortedCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed-nu", "singleton" ]
}>
#partEncoding = #part_tensor.encoding<{
  partConst = 1,
  sparseAttributes = #SortedCOO
}>
module {
  func.func private @printMemref1dInd(%ptr : memref<?xindex>) attributes { llvm.emit_c_interface }
  func.func @entry() {
  	%m44 = arith.constant dense<
      [ [ 0.0, 0.0, 1.5, 1.0],
        [ 0.0, 3.5, 0.0, 0.0],
        [ 1.0, 5.0, 2.0, 0.0],
        [ 1.0, 0.5, 0.0, 0.0] ]> : tensor<4x4xf64>
	// need a part_tensor.convert
    %sm44dc = sparse_tensor.convert %m44 : tensor<4x4xf64> to tensor<4x4xf64, #SortedCOO>
    %partition_plan = part_tensor.get_partitions %sm44dc: tensor<4x4xf64, #SortedCOO> -> memref<?xindex>
    call @printMemref1dInd(%partition_plan) : (memref<?xindex>) -> ()
    // bufferization.dealloc_tensor %sm44dc  : tensor<4x4xf64, #SortedCOO>
	return
  }
}
