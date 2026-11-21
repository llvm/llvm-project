; RUN: opt < %s -passes='require<globals-aa>,aa-eval' -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; Ensure @g is correctly marked as address taken when it is stored into itself.

@g = internal global ptr null

; CHECK-LABEL: self_addresstaken
; CHECK:       MayAlias:	ptr* %p, ptr* @g

define ptr @self_addresstaken() {
  store ptr @g, ptr @g, align 8
  %p = load ptr, ptr @g, align 8
  store ptr null, ptr %p, align 8
  %q = load ptr, ptr @g, align 8
  ret ptr %q
}
