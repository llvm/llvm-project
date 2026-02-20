// RUN:   mlir-opt -mlprogram-pipeline-globals %s

func.func @call_and_store_after(%arg1: memref<f32>) {
  memref.load %arg1[] {name = "caller"} : memref<f32>
  test.call_and_store @callee(%arg1), %arg1 {name = "call", store_before_call = false} : (memref<f32>, memref<f32>) -> ()
  memref.load %arg1[] {name = "post"} : memref<f32>
  return
}