; RUN: llvm-reduce --delta-passes=global-objects --abort-on-invalid-reduction --test FileCheck --test-arg --check-prefixes=INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=FINAL %s --input-file=%t

; INTERESTINGNESS: @a = global i32
; FINAL: @a = global i32 0{{$}}

@a = global i32 0, section "hi"

; INTERESTINGNESS: define {{.*}} @f
; FINAL: define void @f() {
define void @f() section "hello" {
  ret void
}

; INTERESTINGNESS: declare {{.*}} @g
; FINAL: declare void @g(){{$}}
declare void @g() section "hello"
