; RUN: llc -O0 -opaque-pointers=0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpCapability DeviceEnqueue
; CHECK-SPIRV: OpTypeQueue

%spirv.Queue = type opaque

define spir_func void @enqueue_simple_block(%spirv.Queue* addrspace(3)* nocapture %q) {
entry:
  ret void
}
