; RUN: llc -march=ve -mattr=help 2>&1 > /dev/null | FileCheck %s

; CHECK: Available features for this target:
; CHECK:   vpu    - Enable the VPU.

