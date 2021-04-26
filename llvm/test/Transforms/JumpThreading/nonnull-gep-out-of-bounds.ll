; RUN: opt -jump-threading -S %s -o - | FileCheck %s

define i32 @f(i64* %a, i64 %i) {
entry:
  store i64 0, i64* %a, align 8
  %p = getelementptr i64, i64* %a, i64 %i
  %c = icmp eq i64* %p, null
  ; `%a` is non-null at the end of the block, because we store through it.
  ; However, `%p` is derived from `%a` via a GEP that is not `inbounds`, therefore we cannot judge `%p` is non-null as well
  ; and must retain the `icmp` instruction.
  ; CHECK: %c = icmp eq i64* %p, null
  br i1 %c, label %if.else, label %if.then
if.then:
  ret i32 0

if.else:
  ret i32 1
}
