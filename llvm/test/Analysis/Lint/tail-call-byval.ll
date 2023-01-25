; RUN: opt < %s -passes=lint -disable-output 2>&1 | FileCheck %s

%s = type { i8 }

declare void @f1(ptr)

define void @f2() {
entry:
  %c = alloca %s
  tail call void @f1(ptr %c)
  ret void
}

; Lint should complain about the tail call passing the alloca'd value %c to f1.
; CHECK: Undefined behavior: Call with "tail" keyword references alloca
; CHECK-NEXT:  tail call void @f1(ptr %c)

declare void @f3(ptr byval(%s))

define void @f4() {
entry:
  %c = alloca %s
  tail call void @f3(ptr byval(%s) %c)
  ret void
}

; Lint should not complain about passing the alloca'd %c since it's passed
; byval, effectively copying the data to the stack instead of leaking the
; pointer itself.
; CHECK-NOT: Undefined behavior: Call with "tail" keyword references alloca
; CHECK-NOT:  tail call void @f3(ptr byval(%s) %c)


