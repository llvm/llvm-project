; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=hipstdpar-select-accelerator-code \
; RUN: %s | FileCheck %s

$_ZNK8CallableclEPi = comdat any
$_ZNK8CallableclEPf = comdat any
$_ZNK8Callable6mem_fnEPKi = comdat any
$_ZN8Callable13static_mem_fnEPKi = comdat any
; CHECK-NOT: $_ZNK8Callable37another_mem_fn_which_will_get_removedEPKf
$_ZNK8Callable37another_mem_fn_which_will_get_removedEPKf = comdat any
; CHECK-NOT: $_ZN8Callable44another_static_mem_fn_which_will_get_removedEPKf
$_ZN8Callable44another_static_mem_fn_which_will_get_removedEPKf = comdat any

%struct.Callable = type { [64 x i8] }

; CHECK-NOT: @should_be_removed
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr @should_be_removed], section "llvm.metadata"

define void @should_be_removed(ptr %p) {
  ret void
}

declare void @llvm.trap()

; CHECK: define {{.*}} @called_via_chain
define void @called_via_chain(ptr %p) {
  entry:
    %tobool.not = icmp eq ptr %p, null
    br i1 %tobool.not, label %if.then, label %if.end

  if.then:
    tail call void @llvm.trap()
    unreachable

  if.end:
    ret void
}

; CHECK: define {{.*}} @directly_called
define void @directly_called(ptr %p) {
  tail call void @called_via_chain(ptr %p)
  ret void
}

; CHECK: define {{.*}} amdgpu_kernel {{.*}} @accelerator_execution_root
define hidden amdgpu_kernel void @accelerator_execution_root(ptr %p) {
  tail call void @directly_called(ptr %p)
  ret void
}

; CHECK-NOT: @defined_elsewhere_should_be_removed
declare void @defined_elsewhere_should_be_removed(ptr)

; CHECK: declare {{.*}} @defined_elsewhere_directly_called
declare void @defined_elsewhere_directly_called(ptr)

; CHECK: define {{.*}} amdgpu_kernel {{.*}} @another_accelerator_execution_root
define hidden amdgpu_kernel void @another_accelerator_execution_root(ptr %p) {
  tail call void @defined_elsewhere_directly_called(ptr %p)
  ret void
}

; Also test passing a callable object (functor / lambda) to a kernel, which is
; the common pattern for customising algorithms.

; CHECK: define {{.*}} amdgpu_kernel {{.*}} @_Z22accelerator_execution_root_taking_callablePi8Callable
define hidden amdgpu_kernel void @_Z22accelerator_execution_root_taking_callablePi8Callable(ptr noundef %p, ptr addrspace(4) nocapture readonly byref(%struct.Callable) align 8 %callable) {
  %callable_in_generic = addrspacecast ptr addrspace(4) %callable to ptr
  call void @_ZNK8CallableclEPi(ptr noundef nonnull align 1 dereferenceable(64) %callable_in_generic, ptr noundef %p)

  ret void
}

; CHECK: define {{.*}} @_ZNK8CallableclEPi
define linkonce_odr dso_local void @_ZNK8CallableclEPi(ptr noundef nonnull align 1 dereferenceable(64) %this, ptr noundef %p) {
  call void @_ZNK8Callable6mem_fnEPKi(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %p)

  ret void
}

; CHECK: define {{.*}} @_ZNK8Callable6mem_fnEPKi
define linkonce_odr dso_local void @_ZNK8Callable6mem_fnEPKi(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %p) {
  call void @_ZN8Callable13static_mem_fnEPKi(ptr noundef %p)

  ret void
}

; CHECK: define {{.*}} @_ZN8Callable13static_mem_fnEPKi
define linkonce_odr dso_local void @_ZN8Callable13static_mem_fnEPKi(ptr noundef %p) {
  ret void
}

; CHECK-NOT: define {{.*}} @_Z26non_kernel_taking_callablePf8Callable
define dso_local void @_Z26non_kernel_taking_callablePf8Callable(ptr noundef %p, ptr noundef byval(%struct.Callable) align 8 %callable) {
  call void @_ZNK8CallableclEPf(ptr noundef nonnull align 1 dereferenceable(64) %callable, ptr noundef %p)

  ret void
}

; CHECK-NOT: define {{.*}} @_ZNK8CallableclEPf
define linkonce_odr dso_local void @_ZNK8CallableclEPf(ptr noundef nonnull align 1 dereferenceable(64) %this, ptr noundef %p) {
  call void @_ZNK8Callable37another_mem_fn_which_will_get_removedEPKf(ptr noundef nonnull align 1 dereferenceable(64) %this, ptr noundef %p)

  ret void
}

; CHECK-NOT: @_ZNK8Callable37another_mem_fn_which_will_get_removedEPKf
define linkonce_odr dso_local void @_ZNK8Callable37another_mem_fn_which_will_get_removedEPKf(ptr noundef nonnull align 1 dereferenceable(64) %this, ptr noundef %p) {
  call void @_ZN8Callable44another_static_mem_fn_which_will_get_removedEPKf(ptr noundef %p)

  ret void
}

; CHECK-NOT: @_ZN8Callable44another_static_mem_fn_which_will_get_removedEPKf
define linkonce_odr dso_local void @_ZN8Callable44another_static_mem_fn_which_will_get_removedEPKf(ptr noundef %p) {
  ret void
}