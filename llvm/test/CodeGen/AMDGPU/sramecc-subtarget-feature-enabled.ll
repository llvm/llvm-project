; RUN: llc -mtriple=amdgcn -mcpu=gfx700 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=WARN %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx908 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s

; REQUIRES: asserts

; WARN: warning: sramecc 'On' was requested for a processor that does not support it!
; ON: sramecc setting for subtarget: On
define void @sramecc-subtarget-feature-enabled() #0 {
  ret void
}

attributes #0 = { "target-features"="+sramecc" }
