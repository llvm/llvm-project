; RUN: llc -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx84 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s
; RUN: llc -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx84 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s

; The memory operations and b128 pack/unpack moves are separate instructions.
; Only the load/store carries the memory operand.

; CHECK-LABEL: name: load128
; CHECK: [[VALUE:%[0-9]+]]:b128 = LD_i128
; CHECK-SAME: :: (load acquire (s128) from %ir.p
; CHECK: {{%[0-9]+}}:b64, {{%[0-9]+}}:b64 = I128toV2I64 [[VALUE]]
define i128 @load128(ptr addrspace(1) %p) {
  %v = load atomic i128, ptr addrspace(1) %p acquire, align 16
  ret i128 %v
}

; CHECK-LABEL: name: store128
; CHECK: [[VALUE:%[0-9]+]]:b128 = V2I64toI128
; CHECK: ST_i128 [[VALUE]],
; CHECK-SAME: :: (store release (s128) into %ir.p
define void @store128(ptr addrspace(1) %p, i128 %v) {
  store atomic i128 %v, ptr addrspace(1) %p release, align 16
  ret void
}
