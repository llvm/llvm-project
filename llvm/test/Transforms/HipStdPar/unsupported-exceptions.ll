; RUN: split-file %s %t
; RUN: not opt -S -mtriple=amdgpu-amd-amdhsa -passes=hipstdpar-select-accelerator-code \
; RUN:   %t/throw.ll 2>&1 | FileCheck %s --check-prefix=THROW
; RUN: not opt -S -mtriple=amdgpu-amd-amdhsa -passes=hipstdpar-select-accelerator-code \
; RUN:   %t/rethrow.ll 2>&1 | FileCheck %s --check-prefix=RETHROW
; RUN: not opt -S -mtriple=amdgpu-amd-amdhsa -passes=hipstdpar-select-accelerator-code \
; RUN:   %t/invoke.ll 2>&1 | FileCheck %s --check-prefix=INVOKE
; RUN: not opt -S -mtriple=amdgpu-amd-amdhsa -passes=hipstdpar-select-accelerator-code \
; RUN:   %t/bad-cast.ll 2>&1 | FileCheck %s --check-prefix=BAD-CAST
; RUN: not opt -S -mtriple=amdgpu-amd-amdhsa -passes=hipstdpar-select-accelerator-code \
; RUN:   %t/bad-typeid.ll 2>&1 | FileCheck %s --check-prefix=BAD-TYPEID
; RUN: not opt -S -mtriple=amdgpu-amd-amdhsa -passes=hipstdpar-select-accelerator-code \
; RUN:   %t/bad-array-new.ll 2>&1 | FileCheck %s --check-prefix=BAD-ARRAY-NEW
; RUN: not opt -S -mtriple=amdgpu-amd-amdhsa -passes=hipstdpar-select-accelerator-code \
; RUN:   %t/rethrow-primary.ll 2>&1 | FileCheck %s --check-prefix=RETHROW-PRIMARY
; RUN: not opt -S -mtriple=amdgpu-amd-amdhsa -passes=hipstdpar-select-accelerator-code \
; RUN:   %t/call-unexpected.ll 2>&1 | FileCheck %s --check-prefix=CALL-UNEXPECTED
; RUN: not opt -S -mtriple=amdgpu-amd-amdhsa -passes=hipstdpar-select-accelerator-code \
; RUN:   %t/catch-only.ll 2>&1 | FileCheck %s --check-prefix=CATCH-ONLY
; RUN: not opt -S -mtriple=amdgpu-amd-amdhsa -passes=hipstdpar-select-accelerator-code \
; RUN:   %t/try-marker.ll 2>&1 | FileCheck %s --check-prefix=TRY-MARKER
; RUN: opt -S -mtriple=amdgpu-amd-amdhsa -passes=hipstdpar-select-accelerator-code \
; RUN:   %t/unreachable.ll | FileCheck %s --check-prefix=UNREACHABLE
; RUN: opt -S -mtriple=amdgpu-amd-amdhsa -passes=hipstdpar-select-accelerator-code \
; RUN:   %t/similar-names.ll | FileCheck %s --check-prefix=SIMILAR-NAMES

; THROW: error: {{.*}} in function throwing_helper void (): Accelerator does not support C++ exception handling.
; RETHROW: error: {{.*}} in function rethrowing_helper void (): Accelerator does not support C++ exception handling.
; INVOKE: error: {{.*}} in function kernel void (): Accelerator does not support C++ exception handling.
; BAD-CAST: error: {{.*}} in function kernel void (): Accelerator does not support C++ exception handling.
; BAD-TYPEID: error: {{.*}} in function kernel void (): Accelerator does not support C++ exception handling.
; BAD-ARRAY-NEW: error: {{.*}} in function kernel void (): Accelerator does not support C++ exception handling.
; RETHROW-PRIMARY: error: {{.*}} in function kernel void (): Accelerator does not support C++ exception handling.
; CALL-UNEXPECTED: error: {{.*}} in function kernel void (): Accelerator does not support C++ exception handling.
; CATCH-ONLY: error: {{.*}} in function kernel void (): Accelerator does not support C++ exception handling.
; TRY-MARKER: error: {{.*}} in function helper void (): Accelerator does not support C++ exception handling.

; UNREACHABLE-NOT: @host_only
; UNREACHABLE-NOT: @host_catch_only
; UNREACHABLE-NOT: @__cxa_
; UNREACHABLE-NOT: @__CXX_EXCEPTION__hipstdpar_unsupported
; UNREACHABLE: define amdgpu_kernel void @kernel()
; UNREACHABLE-NOT: @host_only
; UNREACHABLE-NOT: @host_catch_only
; UNREACHABLE-NOT: @__cxa_
; UNREACHABLE-NOT: @__CXX_EXCEPTION__hipstdpar_unsupported

; SIMILAR-NAMES: define amdgpu_kernel void @kernel()
; SIMILAR-NAMES: call void @__cxa_throwing()
; SIMILAR-NAMES: call void @my___cxa_rethrow()
; SIMILAR-NAMES: call ptr @__cxa_allocate_exception(i64 4)

;--- throw.ll
define void @throwing_helper() {
entry:
  call void @__cxa_throw(ptr null, ptr null, ptr null)
  unreachable
}

define amdgpu_kernel void @kernel() {
entry:
  call void @throwing_helper()
  ret void
}

declare void @__cxa_throw(ptr, ptr, ptr)

;--- rethrow.ll
define void @rethrowing_helper() {
entry:
  call void @__cxa_rethrow()
  unreachable
}

define amdgpu_kernel void @kernel() {
entry:
  call void @rethrowing_helper()
  ret void
}

declare void @__cxa_rethrow()

;--- invoke.ll
define amdgpu_kernel void @kernel() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @__cxa_throw(ptr null, ptr null, ptr null)
          to label %normal unwind label %cleanup

normal:
  unreachable

cleanup:
  %landing = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } %landing
}

declare void @__cxa_throw(ptr, ptr, ptr)
declare i32 @__gxx_personality_v0(...)

;--- bad-cast.ll
define amdgpu_kernel void @kernel() {
entry:
  call void @__cxa_bad_cast()
  unreachable
}

declare void @__cxa_bad_cast()

;--- bad-typeid.ll
define amdgpu_kernel void @kernel() {
entry:
  call void @__cxa_bad_typeid()
  unreachable
}

declare void @__cxa_bad_typeid()

;--- bad-array-new.ll
define amdgpu_kernel void @kernel() {
entry:
  call void @__cxa_throw_bad_array_new_length()
  unreachable
}

declare void @__cxa_throw_bad_array_new_length()

;--- rethrow-primary.ll
define amdgpu_kernel void @kernel() {
entry:
  call void @__cxa_rethrow_primary_exception(ptr null)
  unreachable
}

declare void @__cxa_rethrow_primary_exception(ptr)

;--- call-unexpected.ll
define amdgpu_kernel void @kernel() {
entry:
  call void @__cxa_call_unexpected(ptr null)
  unreachable
}

declare void @__cxa_call_unexpected(ptr)

;--- catch-only.ll
define amdgpu_kernel void @kernel() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @may_throw()
          to label %exit unwind label %catch

catch:
  %landing = landingpad { ptr, i32 }
          catch ptr null
  ret void

exit:
  ret void
}

declare void @may_throw()
declare i32 @__gxx_personality_v0(...)

;--- try-marker.ll
define void @helper() {
entry:
  call void @__CXX_EXCEPTION__hipstdpar_unsupported()
  call void @may_throw()
  ret void
}

define amdgpu_kernel void @kernel() {
entry:
  call void @helper()
  ret void
}

declare void @__CXX_EXCEPTION__hipstdpar_unsupported()
declare void @may_throw()

;--- unreachable.ll
define void @host_only() {
entry:
  call void @__CXX_EXCEPTION__hipstdpar_unsupported()
  call void @__cxa_throw(ptr null, ptr null, ptr null)
  call void @__cxa_rethrow()
  call void @__cxa_bad_cast()
  call void @__cxa_bad_typeid()
  call void @__cxa_throw_bad_array_new_length()
  call void @__cxa_rethrow_primary_exception(ptr null)
  call void @__cxa_call_unexpected(ptr null)
  unreachable
}

define void @host_catch_only() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @may_throw()
          to label %exit unwind label %catch

catch:
  %landing = landingpad { ptr, i32 }
          catch ptr null
  ret void

exit:
  ret void
}

define amdgpu_kernel void @kernel() {
entry:
  ret void
}

declare void @__cxa_throw(ptr, ptr, ptr)
declare void @__cxa_rethrow()
declare void @__cxa_bad_cast()
declare void @__cxa_bad_typeid()
declare void @__cxa_throw_bad_array_new_length()
declare void @__cxa_rethrow_primary_exception(ptr)
declare void @__cxa_call_unexpected(ptr)
declare void @__CXX_EXCEPTION__hipstdpar_unsupported()
declare void @may_throw()
declare i32 @__gxx_personality_v0(...)

;--- similar-names.ll
define amdgpu_kernel void @kernel() {
entry:
  call void @__cxa_throwing()
  call void @my___cxa_rethrow()
  %exception = call ptr @__cxa_allocate_exception(i64 4)
  ret void
}

declare void @__cxa_throwing()
declare void @my___cxa_rethrow()
declare ptr @__cxa_allocate_exception(i64)
