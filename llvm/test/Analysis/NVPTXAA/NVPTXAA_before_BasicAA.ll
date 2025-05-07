; REQUIRES: asserts, nvptx-registered-target
; RUN: opt -aa-pipeline=default -passes='require<aa>' -debug-pass-manager -disable-output -S < %s 2>&1 | FileCheck %s
; RUN: llc --debug-only='aa' -o /dev/null %s 2>&1 | FileCheck %s -check-prefix=LEGACY

; In default AA pipeline, NVPTXAA should run before BasicAA to reduce compile time for NVPTX backend
target triple = "nvptx64-nvidia-cuda"

; CHECK: Running analysis: NVPTXAA on foo
; CHECK-NEXT: Running analysis: BasicAA on foo

; LEGACY: AAResults register Early ExternalAA: NVPTX Address space based Alias Analysis Wrapper
; LEGACY-NEXT: AAResults register BasicAA
define void @foo(){
entry:
  ret void
}

