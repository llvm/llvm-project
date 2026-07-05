; RUN: verify-uselistorder < %s

; Global referencing ifunc.
@ptr_foo = global ptr @foo_ifunc

; Alias for ifunc.
@alias_foo = alias void (), ptr @foo_ifunc

@foo_ifunc = ifunc void (), ptr @foo_resolver

define ptr @foo_resolver() {
entry:
  ret ptr null
}

; Function referencing ifunc.
define void @bar() {
entry:
  call void @foo_ifunc()
  ret void
}

; Global referencing function.
@ptr_bar = global ptr @bar

; Alias for function.
@alias_bar = alias void (), ptr @bar

@bar_ifunc = ifunc void (), ptr @bar_resolver

define ptr @bar_resolver() {
entry:
  ret ptr null
}

; Function referencing bar.
define void @bar2() {
entry:
  call void @bar()
  ret void
}
