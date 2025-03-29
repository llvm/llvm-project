; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; TODO(#60133): Requires updates following opaque pointer migration.
; XFAIL: *

%struct.Node = type { %struct.Node.0 addrspace(1)* }
; CHECK: %[[#]] = OpTypeOpaque "struct.Node.0"
%struct.Node.0 = type opaque

define spir_kernel void @create_linked_lists(%struct.Node addrspace(1)* nocapture %pNodes, i32 addrspace(1)* nocapture %allocation_index, i32 %list_length) {
entry:
  ret void
}
