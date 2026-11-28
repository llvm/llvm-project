; RUN: opt -S -mtriple=spirv64-amd-amdhsa -passes=hipstdpar-select-accelerator-code \
; RUN: %s | FileCheck %s

; CHECK-NOT: define {{.*}} @should_be_removed
define void @should_be_removed() {
  ret void
}

; CHECK: define {{.*}} @directly_called
define void @directly_called() {
  ret void
}

; CHECK: define spir_kernel void @accelerator_execution_root
define spir_kernel void @accelerator_execution_root() {
  call void @directly_called()
  ret void
}
