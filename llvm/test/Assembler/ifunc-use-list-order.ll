; RUN: verify-uselistorder < %s

; Global referencing ifunc.
@ptr_foo = global void ()* @foo_ifunc

; Alias for ifunc.
@alias_foo = alias void (), void ()* @foo_ifunc

@foo_ifunc = ifunc void (), void ()* ()* @foo_resolver

define void ()* @foo_resolver() {
entry:
  ret void ()* null
}

; Function referencing ifunc.
define void @bar() {
entry:
  call void @foo_ifunc()
  ret void
}

; Global referencing function.
@ptr_bar = global void ()* @bar

; Alias for function.
@alias_bar = alias void (), void ()* @bar

@bar_ifunc = ifunc void (), void ()* ()* @bar_resolver

define void ()* @bar_resolver() {
entry:
  ret void ()* null
}

; Function referencing bar.
define void @bar2() {
entry:
  call void @bar()
  ret void
}
