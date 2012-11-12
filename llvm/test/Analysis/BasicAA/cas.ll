; RUN: opt < %s -basicaa -gvn -instcombine -S | FileCheck %s

@flag0 = internal global i32 zeroinitializer
@turn = internal global i32 zeroinitializer

; CHECK: ret i32 0

define i32 @main() {
  %a = load i32* @flag0
  %b = atomicrmw xchg i32* @turn, i32 1 monotonic
  %c = load i32* @flag0
  %d = sub i32 %a, %c
  ret i32 %d
}
