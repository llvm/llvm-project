; RUN: opt -ppc-prep-ifunc-aix -mtriple=powerpc64-ibm-aix-xcoff %s -S | FileCheck %s --check-prefixes=CHECK,CHECK64
; RUN: opt -ppc-prep-ifunc-aix -mtriple=powerpc-ibm-aix-xcoff %s -S | FileCheck %s --check-prefixes=CHECK,CHECK32

; CHECK64: @__update_foo = private global { ptr, ptr } { ptr @foo, ptr @foo.resolver }, section "__ifunc_sec", align 8, !associated ![[#INIT_IFUNC:]]
; CHECK64: @__update_bar = private global { ptr, ptr } { ptr @bar, ptr @bar.resolver }, section "__ifunc_sec", align 8, !associated ![[#INIT_IFUNC]]
; CHECK32: @__update_foo = private global { ptr, ptr } { ptr @foo, ptr @foo.resolver }, section "__ifunc_sec", align 4, !associated ![[#INIT_IFUNC:]]
; CHECK32: @__update_bar = private global { ptr, ptr } { ptr @bar, ptr @bar.resolver }, section "__ifunc_sec", align 4, !associated ![[#INIT_IFUNC]]
; CHECK: @foo = ifunc i32 (...), ptr @foo.resolver, !associated ![[#UPDATE_FOO:]]
; CHECK: @bar = ifunc void (i32, i1), ptr @bar.resolver, !associated ![[#UPDATE_BAR:]]
; CHECK: declare void @__init_ifuncs()
; CHECK: ![[#INIT_IFUNC]] = !{ptr @__init_ifuncs}
; CHECK: ![[#UPDATE_FOO]] = !{ptr @__update_foo}
; CHECK: ![[#UPDATE_BAR]] = !{ptr @__update_bar}

@foo = ifunc i32 (...), ptr @foo.resolver
@bar = ifunc void (i32, i1), ptr @bar.resolver

define hidden signext i32 @my_foo() {
entry:
  ret i32 4
}

define internal ptr @foo.resolver() {
entry:
  ret ptr @my_foo
}

declare void @my_bar(i32, i1)

define ptr @bar.resolver() {
entry:
  ret ptr @my_bar
}

