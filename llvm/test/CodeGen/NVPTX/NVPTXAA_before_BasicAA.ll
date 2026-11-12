; REQUIRES: asserts
; RUN: opt -aa-pipeline=default -debug-pass-manager -passes='require<aa>' -disable-output -S < %s 2>&1 | FileCheck %s
; RUN: llc -enable-new-pm=true -debug-pass-manager -filetype=null %s 2>&1 | FileCheck %s
; RUN: llc -enable-new-pm=false --debug-only='aa' -filetype=null %s 2>&1 | FileCheck %s -check-prefix=LEGACY

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

