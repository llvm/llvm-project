; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; TODO(#60133): Requires updates following opaque pointer migration.
; XFAIL: *

%struct.Node = type { ptr addrspace(1) }
; CHECK: %[[#]] = OpTypeOpaque "struct.Node.0"
%struct.Node.0 = type opaque

define spir_kernel void @create_linked_lists(ptr addrspace(1) nocapture %pNodes, ptr addrspace(1) nocapture %allocation_index, i32 %list_length) {
entry:
  ret void
}
