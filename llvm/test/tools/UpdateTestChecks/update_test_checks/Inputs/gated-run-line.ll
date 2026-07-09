; RUN(some-feature): opt < %s -passes=instsimplify -S | FileCheck %s --check-prefix=GATED

define i32 @add(i32 %a, i32 %b) {
  %r = add i32 %a, %b
  ret i32 %r
}
