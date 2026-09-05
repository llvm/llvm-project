; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void @f(i1 %cond, target("dx.RawBuffer", half, 1, 0) %A,
               target("dx.RawBuffer", half, 1, 0) %B) {
  %sel = select i1 %cond, target("dx.RawBuffer", half, 1, 0) %A,
                          target("dx.RawBuffer", half, 1, 0) %B
; CHECK: select values cannot have token type
  ret void
}
