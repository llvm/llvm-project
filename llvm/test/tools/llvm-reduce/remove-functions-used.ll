; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=functions --test FileCheck --test-arg --check-prefix=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t.0
; RUN: FileCheck --implicit-check-not=define --check-prefix=CHECK-FINAL %s < %t.0

; Check a case where llvm.used is fully deleted
; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=functions --test FileCheck --test-arg --check-prefixes=CHECK-OTHER --test-arg %s --test-arg --input-file %s -o %t.1
; RUN: FileCheck --implicit-check-not=define --check-prefix=CHECK-REMOVED %s < %t.1

@llvm.used = appending global [2 x ptr] [ptr @kept_used, ptr @removed_used ]
@llvm.compiler.used = appending global [2 x ptr] [ptr @kept_compiler_used, ptr @removed_compiler_used ]


; CHECK-REMOVED-NOT: @llvm.used
; CHECK-REMOVED-NOT: @llvm.compiler.used

; CHECK-FINAL: @llvm.used = appending global [1 x ptr] [ptr @kept_used]
; CHECK-FINAL: @llvm.compiler.used = appending global [1 x ptr] [ptr @kept_compiler_used]


; CHECK-INTERESTINGNESS: define void @kept_used(
; CHECK-FINAL: define void @kept_used(
define void @kept_used() {
  ret void
}

define void @removed_used() {
  ret void
}

; CHECK-INTERESTINGNESS: define void @kept_compiler_used(
; CHECK-FINAL: define void @kept_compiler_used(
define void @kept_compiler_used() {
  ret void
}

define void @removed_compiler_used() {
  ret void
}

; CHECK-OTHER: define void @foo(
; CHECK-REMOVED: define void @foo(
define void @foo() {
  ret void
}

