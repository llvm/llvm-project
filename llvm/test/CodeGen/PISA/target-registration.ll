; Smoke test that the PISA target is registered with and selectable by llc.

; RUN: llc --version 2>&1 | FileCheck %s

; CHECK: Registered Targets:
; CHECK: pisa{{.*}}- PISA 64-bit
