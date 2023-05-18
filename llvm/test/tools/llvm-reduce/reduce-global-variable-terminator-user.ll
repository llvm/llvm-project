; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=global-variables --test FileCheck --test-arg --check-prefixes=INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=RESULT %s < %t

; The global variable reduction was trying to delete use instructions
; of globals for some reason, and breaking the basic blocks that had
; global uses in the terminator

; RESULT-NOT: @zed
@zed = global i32 0

; INTERESTING: @bar
; RESULT: @bar
@bar = global i32 1

; RESULT: define ptr @zed_user() {
; RESULT-NEXT: ret ptr null
define ptr @zed_user() {
  ret ptr @zed
}
