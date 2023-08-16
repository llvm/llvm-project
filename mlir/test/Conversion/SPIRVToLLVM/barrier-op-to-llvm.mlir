// RUN: mlir-opt -convert-spirv-to-llvm='use-opaque-pointers=1' %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.ControlBarrier
//===----------------------------------------------------------------------===//

// COM: enum class Scope : uint32_t {
// COM:   CrossDevice = 0,
// COM:   Device = 1,
// COM:   Workgroup = 2,
// COM:   Subgroup = 3,
// COM:   Invocation = 4,
// COM:   QueueFamily = 5,
// COM:   ShaderCallKHR = 6,
// COM: };
// COM: enum class MemorySemantics : uint32_t {
// COM:   None = 0,
// COM:   Acquire = 2,
// COM:   Release = 4,
// COM:   AcquireRelease = 8,
// COM:   SequentiallyConsistent = 16,
// COM:   UniformMemory = 64,
// COM:   SubgroupMemory = 128,
// COM:   WorkgroupMemory = 256,
// COM:   CrossWorkgroupMemory = 512,
// COM:   AtomicCounterMemory = 1024,
// COM:   ImageMemory = 2048,
// COM:   OutputMemory = 4096,
// COM:   MakeAvailable = 8192,
// COM:   MakeVisible = 16384,
// COM:   Volatile = 32768,
// COM: };

// CHECK: llvm.func @_Z22__spirv_ControlBarrierjjj(i32, i32, i32)
// CHECK-LABEL: @control_barrier
spirv.func @control_barrier() "None" {
  // COM: The constants below represent their corresponding scope or memory semantic.
  // CHECK-NEXT: %0 = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: %1 = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: %2 = llvm.mlir.constant(272 : i32) : i32
  // CHECK-NEXT: llvm.call @_Z22__spirv_ControlBarrierjjj(%0, %1, %2) : (i32, i32, i32) -> ()
  spirv.ControlBarrier <Workgroup>, <Workgroup>, <SequentiallyConsistent|WorkgroupMemory>
  spirv.Return
}
