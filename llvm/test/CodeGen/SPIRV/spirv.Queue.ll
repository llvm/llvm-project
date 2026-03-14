; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpCapability DeviceEnqueue
; CHECK-SPIRV: OpTypeQueue

define spir_func void @enqueue_simple_block(target("spirv.Queue") %q) {
entry:
  ret void
}
