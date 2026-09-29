; RUN: llc -O0 -march aarch64 -global-isel -stop-after=irtranslator -o - %s | llc -x mir -march aarch64 -run-pass=none -o - | FileCheck %s

define i16 @unnamed_stack() {
entry:
  ; CHECK-NAME: unnamed_stack
  ; CHECK:      stack:
  ; CHECK-NEXT:   - { id: 0, name: '',
  ; CHECK:      %0:_(p0) = G_FRAME_INDEX %stack.0
  %0 = alloca i16
  %1 = load i16, ptr %0
  ret i16 %1
}

define i16 @named_stack() {
entry:
  ; CHECK-NAME: named_stack
  ; CHECK:      stack:
  ; CHECK-NEXT:   - { id: 0, name: ptr,
  ; CHECK:      %0:_(p0) = G_FRAME_INDEX %stack.0.ptr
  %ptr = alloca i16
  %0 = load i16, ptr %ptr
  ret i16 %0
}
