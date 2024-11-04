; RUN: opt -O3 -debug-only=amdgpu-attributor -S -o - %s 2>&1 | FileCheck %s --check-prefix=PRE-LINK
; RUN: opt -passes="lto<O3>" -debug-only=amdgpu-attributor -S -o - %s 2>&1 | FileCheck %s --check-prefix=POST-LINK

; REQUIRES: amdgpu-registered-target
; REQUIRES: asserts

target triple = "amdgcn-amd-amdhsa"

; PRE-LINK: Module {{.*}} is not assumed to be a closed world.
; POST-LINK: Module {{.*}} is assumed to be a closed world.
define hidden noundef i32 @_Z3foov() {
  ret i32 1
}
