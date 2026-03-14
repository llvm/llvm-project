; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=global-values --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=CHECK-FINAL %s --input-file=%t

; CHECK-INTERESTINGNESS: @g = {{.*}}global i32
; CHECK-FINAL: @g = global i32

@g = thread_local(initialexec) global i32 0
