; RUN: llvm-reduce --delta-passes=global-objects --abort-on-invalid-reduction --test FileCheck --test-arg --check-prefixes=INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=FINAL %s --input-file=%t

; INTERESTINGNESS: @b = global i32
; FINAL: @b = global i32 0{{$}}

@b = global i32 0, align 4

; INTERESTINGNESS: define {{.*}} @f
; FINAL: define void @f() {
define void @f() align 4 {
  ret void
}

; INTERESTINGNESS: declare {{.*}} @h
; FINAL: declare void @h(){{$}}
declare void @h() align 4
