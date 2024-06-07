; RUN: not llc --mtriple=loongarch64 -emulated-tls -mattr=+d \
; RUN:     -relocation-model=pic < %s 2>&1 | FileCheck %s

; CHECK: error: the emulated TLS is prohibited
; CHECK: error: the emulated TLS is prohibited
; CHECK: error: the emulated TLS is prohibited

@external_x = external thread_local global i32, align 8
@y = thread_local global i8 7, align 2
@internal_z = internal thread_local global i64 9, align 16

define ptr @get_external_x() nounwind {
entry:
  ret ptr @external_x
}

define ptr @get_y() nounwind {
entry:
  ret ptr @y
}

define ptr @get_internal_z() nounwind {
entry:
  ret ptr @internal_z
}
