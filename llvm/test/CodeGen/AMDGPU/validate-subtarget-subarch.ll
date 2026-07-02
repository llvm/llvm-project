; RUN: llc -mtriple=amdgpu6 -mcpu=gfx600 -filetype=null %s
; RUN: llc -mtriple=amdgpu6.00 -mcpu=gfx600 -filetype=null %s
; RUN: llc -mtriple=amdgpu9 -mcpu=gfx900 -filetype=null %s
; RUN: llc -mtriple=amdgpu9.0a -mcpu=gfx90a -filetype=null %s
; RUN: llc -mtriple=amdgpu9.08 -mcpu=gfx908 -filetype=null %s
; RUN: llc -mtriple=amdgpu9.4 -mcpu=gfx950 -filetype=null %s
; RUN: llc -mtriple=amdgpu11 -mcpu=gfx1100 -filetype=null %s
; RUN: llc -mtriple=amdgpu11.7 -mcpu=gfx1170 -filetype=null %s

; Test legacy missing subarch
; RUN: llc -mtriple=amdgcn -mcpu=gfx950 -filetype=null %s
; RUN: llc -mtriple=amdgpu -mcpu=gfx950 -filetype=null %s
; RUN: llc -mtriple=amdgpu -mcpu=gfx600 -filetype=null %s

; RUN: llc -mtriple=amdgpu9 -mcpu=invalid -filetype=null %s 2>&1 | FileCheck -check-prefix=INVALID-MCPU-VALID-SUBARCH %s

; RUN: sed 's/TARGET_CPU/invalid/g' < %s | llc -mtriple=amdgpu12.50-amd-amdhsa -filetype=null 2>&1 | FileCheck -check-prefix=INVALID-MCPU-VALID-SUBARCH %s
; RUN: sed 's/TARGET_CPU/invalid/g' < %s | llc -mtriple=amdgpu12.51-amd-amdhsa -filetype=null 2>&1 | FileCheck -check-prefix=INVALID-MCPU-VALID-SUBARCH %s
; RUN: sed 's/TARGET_CPU/gfx900/g'< %s | not llc -mtriple=amdgpu12345 -filetype=null 2>&1 | FileCheck -check-prefix=INVALID-SUBARCH-VALID-MCPU %s

; RUN: sed 's/TARGET_CPU/gfx601/g' < %s | not llc -mtriple=amdgpu6.00 -filetype=null 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: sed 's/TARGET_CPU/gfx600/g' < %s | not llc -mtriple=amdgpu6.01 -filetype=null 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: sed 's/TARGET_CPU/gfx600/g' < %s | not llc -mtriple=amdgpu7 -filetype=null 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: sed 's/TARGET_CPU/gfx810/g' < %s | not llc -mtriple=amdgpu8.03 -filetype=null 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: sed 's/TARGET_CPU/gfx810/g' < %s | not llc -mtriple=amdgpu8 -filetype=null 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: sed 's/TARGET_CPU/gfx942/g' < %s | not llc -mtriple=amdgpu9 -filetype=null 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: sed 's/TARGET_CPU/gfx900/g' < %s | not llc -mtriple=amdgpu9.4 -filetype=null 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: sed 's/TARGET_CPU/gfx1030/g' < %s | not llc -mtriple=amdgpu9.4 -filetype=null 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: sed 's/TARGET_CPU/gfx900/g' < %s | not llc -mtriple=amdgpu9.08 -filetype=null 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: sed 's/TARGET_CPU/gfx90a/g' < %s | not llc -mtriple=amdgpu9 -filetype=null 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: sed 's/TARGET_CPU/gfx908/g' < %s | not llc -mtriple=amdgpu9 -filetype=null 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: sed 's/TARGET_CPU/gfx1170/g' < %s | not llc -mtriple=amdgpu11 -filetype=null 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: sed 's/TARGET_CPU/gfx1100/g' < %s | not llc -mtriple=amdgpu11.7 -filetype=null 2>&1 | FileCheck -check-prefix=ERR %s

; Check that subtargets not covered by the subarch are rejected. This
; tests the error on subtarget construction, which is different from
; the error on TargetMachine construction.


; INVALID-MCPU: error: invalid subtarget 'invalid' for subarch amdgpu{{[0-9.]+}}
; INVALID-SUBARCH-VALID-MCPU: LLVM ERROR: unknown subarch amdgpu12345
; INVALID-MCPU-VALID-SUBARCH: 'invalid' is not a recognized processor for this target (ignoring processor)

; ERR: error: invalid subtarget 'gfx{{.+}}' for subarch amdgpu{{[0-9.]+}}
define amdgpu_kernel void @foo() "target-cpu"="TARGET_CPU" {
  ret void
}
