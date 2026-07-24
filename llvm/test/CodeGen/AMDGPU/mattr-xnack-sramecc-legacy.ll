; Test that -mattr=±xnack/±sramecc emit errors in codegen
; xnack/sramecc should be specified via module flags instead of subtarget features.
;
; RUN: not llc -mtriple=amdgpu9.0a-amd-amdhsa -mattr=+xnack,+sramecc < %s 2>&1 | FileCheck --check-prefix=BOTH-ERR %s
; RUN: not llc -mtriple=amdgpu9.0a-amd-amdhsa -mattr=+xnack < %s 2>&1 | FileCheck --check-prefix=XNACK-ERR %s
; RUN: not llc -mtriple=amdgpu9.0a-amd-amdhsa -mattr=-xnack < %s 2>&1 | FileCheck --check-prefix=XNACK-ERR %s
; RUN: not llc -mtriple=amdgpu9.0a-amd-amdhsa -mattr=xnack < %s 2>&1 | FileCheck --check-prefix=XNACK-ERR %s
; RUN: not llc -mtriple=amdgpu9.0a-amd-amdhsa -mattr=+sramecc < %s 2>&1 | FileCheck --check-prefix=SRAMECC-ERR %s
; RUN: not llc -mtriple=amdgpu9.0a-amd-amdhsa -mattr=-sramecc < %s 2>&1 | FileCheck --check-prefix=SRAMECC-ERR %s
; RUN: not llc -mtriple=amdgpu9.0a-amd-amdhsa -mattr=sramecc < %s 2>&1 | FileCheck --check-prefix=SRAMECC-ERR %s

; BOTH-ERR: error: xnack/sramecc should be specified via module flags. Use module flag 'amdgpu.xnack' instead of subtarget feature
; BOTH-ERR: error: xnack/sramecc should be specified via module flags. Use module flag 'amdgpu.sramecc' instead of subtarget feature

; XNACK-ERR: error: xnack/sramecc should be specified via module flags. Use module flag 'amdgpu.xnack' instead of subtarget feature
; XNACK-ERR-NOT: sramecc

; SRAMECC-ERR: error: xnack/sramecc should be specified via module flags. Use module flag 'amdgpu.sramecc' instead of subtarget feature
; SRAMECC-ERR-NOT: xnack

define void @kernel() {
  ret void
}
