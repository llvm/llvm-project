; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=global-variables --test FileCheck --test-arg --check-prefix=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t.0
; RUN: FileCheck --implicit-check-not=define --check-prefix=CHECK-FINAL %s < %t.0

; Check a case where llvm.used is fully deleted
; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=global-variables --test FileCheck --test-arg --check-prefixes=CHECK-OTHER --test-arg %s --test-arg --input-file %s -o %t.1
; RUN: FileCheck --implicit-check-not=define --check-prefix=CHECK-REMOVED %s < %t.1

; CHECK-INTERESTINGNESS: @kept_used = global
; CHECK-FINAL: @kept_used = global
@kept_used = global i32 0

@removed_used = global i32 1

; CHECK-INTERESTINGNESS: @kept_compiler_used = global
; CHECK-FINAL: @kept_compiler_used = global
@kept_compiler_used = global i32 2

@removed_compiler_used = global i32 3

; CHECK-OTHER: @foo = global
; CHECK-REMOVED: @foo = global
@foo = global i32 4

@llvm.used = appending global [2 x ptr] [ptr @kept_used, ptr @removed_used ]
@llvm.compiler.used = appending global [2 x ptr] [ptr @kept_compiler_used, ptr @removed_compiler_used ]


; CHECK-REMOVED-NOT: @llvm.used
; CHECK-REMOVED-NOT: @llvm.compiler.used

; CHECK-FINAL: @llvm.used = appending global [1 x ptr] [ptr @kept_used]
; CHECK-FINAL: @llvm.compiler.used = appending global [1 x ptr] [ptr @kept_compiler_used]
