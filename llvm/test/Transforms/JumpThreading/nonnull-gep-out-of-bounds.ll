; RUN: opt -passes=jump-threading -S %s -o - | FileCheck %s

define i32 @f(ptr %a, i64 %i) {
entry:
  store i64 0, ptr %a, align 8
  %p = getelementptr i64, ptr %a, i64 %i
  %c = icmp eq ptr %p, null
  ; `%a` is non-null at the end of the block, because we store through it.
  ; However, `%p` is derived from `%a` via a GEP that is not `inbounds`, therefore we cannot judge `%p` is non-null as well
  ; and must retain the `icmp` instruction.
  ; CHECK: %c = icmp eq ptr %p, null
  br i1 %c, label %if.else, label %if.then
if.then:
  ret i32 0

if.else:
  ret i32 1
}
