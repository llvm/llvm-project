; RUN: opt -passes=gvn -enable-gvn-memoryssa=true -S < %s | FileCheck %s
;
; Explicitly requesting both engines is a contradiction and is rejected.
; RUN: not opt -passes=gvn -enable-gvn-memdep=true -enable-gvn-memoryssa=true \
; RUN:   -S < %s 2>&1 | FileCheck %s --check-prefix=CONFLICT
; CONFLICT: -enable-gvn-memdep and -enable-gvn-memoryssa are mutually exclusive

define i32 @redundant_load(ptr %p) {
; CHECK-LABEL: @redundant_load(
; CHECK:         %a = load i32, ptr %p
; CHECK-NOT:     load i32
; CHECK:         %c = add i32 %a, %a
; CHECK:         ret i32 %c
  %a = load i32, ptr %p
  %b = load i32, ptr %p
  %c = add i32 %a, %b
  ret i32 %c
}
