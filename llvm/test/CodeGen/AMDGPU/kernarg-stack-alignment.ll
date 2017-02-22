; RUN: llc -O0 -march=amdgcn -verify-machineinstrs < %s | FileCheck %s

; Test that the alignment of kernel arguments does not impact the
; alignment of the stack

; CHECK-LABEL: {{^}}no_args:
; CHECK: ScratchSize: 5{{$}}
define void @no_args() {
  %alloca = alloca i8
  store volatile i8 0, i8* %alloca
  ret void
}

; CHECK-LABEL: {{^}}force_align32:
; CHECK: ScratchSize: 5{{$}}
define void @force_align32(<8 x i32>) {
  %alloca = alloca i8
  store volatile i8 0, i8* %alloca
  ret void
}

; CHECK-LABEL: {{^}}force_align64:
; CHECK: ScratchSize: 5{{$}}
define void @force_align64(<16 x i32>) {
  %alloca = alloca i8
  store volatile i8 0, i8* %alloca
  ret void
}

; CHECK-LABEL: {{^}}force_align128:
; CHECK: ScratchSize: 5{{$}}
define void @force_align128(<32 x i32>) {
  %alloca = alloca i8
  store volatile i8 0, i8* %alloca
  ret void
}

; CHECK-LABEL: {{^}}force_align256:
; CHECK: ScratchSize: 5{{$}}
define void @force_align256(<64 x i32>) {
  %alloca = alloca i8
  store volatile i8 0, i8* %alloca
  ret void
}
