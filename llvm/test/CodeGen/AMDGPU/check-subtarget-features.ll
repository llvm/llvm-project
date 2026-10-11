; RUN: not llc -global-isel=0 -mtriple=amdgpu11.00 -mattr=+wavefrontsize32,+wavefrontsize64 < %s 2>&1 | FileCheck %s -check-prefix=ERR -implicit-check-not=error:
; RUN: not llc -global-isel=1 -mtriple=amdgpu11.00 -mattr=+wavefrontsize32,+wavefrontsize64 < %s 2>&1 | FileCheck %s -check-prefix=ERR -implicit-check-not=error:
; RUN: not llc -global-isel=0 -mtriple=amdgpu13.10 -mattr=+local-memory-size-limit-65536,+local-memory-size-limit-131072 < %s 2>&1 | FileCheck %s -check-prefix=LDS-LIMIT-ERR -implicit-check-not=error:
; RUN: not llc -global-isel=1 -mtriple=amdgpu13.10 -mattr=+local-memory-size-limit-65536,+local-memory-size-limit-131072 < %s 2>&1 | FileCheck %s -check-prefix=LDS-LIMIT-ERR -implicit-check-not=error:
; RUN: not llc -global-isel=0 -mtriple=amdgpu12.00 -mattr=+local-memory-size-limit-131072 < %s 2>&1 | FileCheck %s -check-prefix=LDS-LIMIT-UNSUPPORTED-ERR -implicit-check-not=error:
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.00 -mattr=+local-memory-size-limit-131072 < %s 2>&1 | FileCheck %s -check-prefix=LDS-LIMIT-UNSUPPORTED-ERR -implicit-check-not=error:

; ERR: error: {{.*}} in function f void (): must specify exactly one of wavefrontsize32 and wavefrontsize64
; LDS-LIMIT-ERR: error: {{.*}} in function f void (): only one local-memory-size-limit feature can be used
; LDS-LIMIT-UNSUPPORTED-ERR: error: {{.*}} in function f void (): local-memory-size-limit is not supported on this target

define void @f() {
  ret void
}
