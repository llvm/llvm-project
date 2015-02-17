; RUN: opt < %s -instsimplify -S | FileCheck %s

@zeroinit = constant {} zeroinitializer
@undef = constant {} undef

define i32 @crash_on_zeroinit() {
; CHECK-LABEL: @crash_on_zeroinit
; CHECK: ret i32 0
  %load = load i32* bitcast ({}* @zeroinit to i32*)
  ret i32 %load
}

define i32 @crash_on_undef() {
; CHECK-LABEL: @crash_on_undef
; CHECK: ret i32 undef
  %load = load i32* bitcast ({}* @undef to i32*)
  ret i32 %load
}

