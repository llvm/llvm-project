; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpCapability DeviceEnqueue
; CHECK-SPIRV: OpTypeQueue

%opencl.queue_t = type opaque

define spir_func void @enqueue_simple_block(%opencl.queue_t* addrspace(3)* nocapture %q) {
entry:
  ret void
}
