; RUN: opt -S -passes='select-function<fn=foo>' < %s | FileCheck --check-prefix=FOO %s
; RUN: opt -S -passes='select-function<fn=bar>' < %s | FileCheck --check-prefix=BAR %s
; RUN: opt -S -passes='select-function<fn=baz>' < %s | FileCheck --check-prefix=BAZ %s
; RUN: opt -S -passes='select-function<fn=foo;fn=baz>' < %s | FileCheck --check-prefix=FOO_BAZ %s

; @foo calls @shared. @bar calls @shared and @bar_helper.
; @baz is standalone. Each selection should keep exactly its own
; transitive closure and remove everything else.

define i32 @foo(i32 %x) {
  %r = call i32 @shared(i32 %x)
  ret i32 %r
}

define i32 @bar(i32 %x) {
  %a = call i32 @shared(i32 %x)
  %b = call i32 @bar_helper(i32 %a)
  ret i32 %b
}

define i32 @bar_helper(i32 %x) {
  %r = add i32 %x, 10
  ret i32 %r
}

define i32 @shared(i32 %x) {
  %r = mul i32 %x, 3
  ret i32 %r
}

define i32 @baz(i32 %x) {
  ret i32 %x
}

; FOO: define {{.*}} @foo(
; FOO: define {{.*}} @shared(
; FOO-NOT: @bar
; FOO-NOT: @bar_helper
; FOO-NOT: @baz

; BAR: define {{.*}} @bar(
; BAR: define {{.*}} @bar_helper(
; BAR: define {{.*}} @shared(
; BAR-NOT: @foo
; BAR-NOT: @baz

; BAZ: define {{.*}} @baz(
; BAZ-NOT: @foo
; BAZ-NOT: @bar
; BAZ-NOT: @bar_helper
; BAZ-NOT: @shared

; FOO_BAZ: define {{.*}} @foo(
; FOO_BAZ: define {{.*}} @shared(
; FOO_BAZ: define {{.*}} @baz(
; FOO_BAZ-NOT: @bar
; FOO_BAZ-NOT: @bar_helper
