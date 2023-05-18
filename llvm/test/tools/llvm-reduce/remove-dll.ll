; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=global-values --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=CHECK-FINAL %s --input-file=%t

; CHECK-INTERESTINGNESS: @g = external {{.*}}global i32
; CHECK-FINAL: @g = external global i32
; CHECK-INTERESTINGNESS: @h = external {{.*}}global i32
; CHECK-FINAL: @h = external global i32

@g = external dllimport global i32
@h = external dllexport global i32
