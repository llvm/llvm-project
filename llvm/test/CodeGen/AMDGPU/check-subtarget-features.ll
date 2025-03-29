; RUN: not llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,+wavefrontsize64 < %s 2>&1 | FileCheck %s -check-prefix=ERR -implicit-check-not=error:
; RUN: not llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,+wavefrontsize64 < %s 2>&1 | FileCheck %s -check-prefix=ERR -implicit-check-not=error:

; ERR: error: {{.*}} in function f void (): must specify exactly one of wavefrontsize32 and wavefrontsize64

define void @f() {
  ret void
}
