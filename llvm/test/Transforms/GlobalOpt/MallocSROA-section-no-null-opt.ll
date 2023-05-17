; RUN: opt -passes=globalopt -S < %s | FileCheck %s
; CHECK: @Y
; CHECK: section ".foo"

%struct.xyz = type { double, i32 }

@Y = internal global ptr null ,section ".foo"            ; <ptr> [#uses=2]
@numf2s = external global i32                     ; <ptr> [#uses=1]

define void @init_net() #0 {
; CHECK-LABEL: init_net(
; CHECK: load i32, ptr @numf2s
; CHECK: call ptr @malloc
; CHECK: store ptr {{.*}}, ptr @Y
entry:
  %0 = load i32, ptr @numf2s, align 4                 ; <i32> [#uses=1]
  %mallocsize2 = shl i32 %0, 4                    ; <i32> [#uses=1]
  %malloccall3 = tail call ptr @malloc(i32 %mallocsize2)  ; <ptr> [#uses=1]
  store ptr %malloccall3, ptr @Y, align 8
  ret void
}

define ptr @load_train() #0 {
; CHECK-LABEL: load_train(
; CHECK: load ptr, ptr @Y
entry:
  %0 = load ptr, ptr @Y, align 8             ; <ptr> [#uses=0]
  ret ptr %0
}

declare noalias ptr @malloc(i32)

attributes #0 = { null_pointer_is_valid }
