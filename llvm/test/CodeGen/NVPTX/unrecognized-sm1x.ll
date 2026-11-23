; RUN: llc < %s -mtriple=nvptx -mcpu=sm_10 -debug-only=nvptx-subtarget -o /dev/null 2>&1 | FileCheck %s --check-prefix=SM10
; RUN: llc < %s -mtriple=nvptx -mcpu=sm_11 -debug-only=nvptx-subtarget -o /dev/null 2>&1 | FileCheck %s --check-prefix=SM11
; RUN: llc < %s -mtriple=nvptx -mcpu=sm_12 -debug-only=nvptx-subtarget -o /dev/null 2>&1 | FileCheck %s --check-prefix=SM12
; RUN: llc < %s -mtriple=nvptx -mcpu=sm_13 -debug-only=nvptx-subtarget -o /dev/null 2>&1 | FileCheck %s --check-prefix=SM13
; REQUIRES: asserts

; SM10: 'sm_10' is not a recognized processor for this target (ignoring processor)
; SM11: 'sm_11' is not a recognized processor for this target (ignoring processor)
; SM12: 'sm_12' is not a recognized processor for this target (ignoring processor)
; SM13: 'sm_13' is not a recognized processor for this target (ignoring processor)
