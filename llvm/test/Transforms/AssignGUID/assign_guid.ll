; RUN: opt -S -passes=assign-guid %s | FileCheck %s

@G = global i32 0
; CHECK: @G = global i32 0, !unique_id !0
@G_EXT = external global i32

declare external void @f_ext()

@A = alias i32, ptr @G
@A_EXT = external alias i32, ptr @G

define void @f() {
; CHECK: define void @f() !unique_id !1 {
  ret void
}

; CHECK: !0 = !{i64 -6455552227143004193}
; CHECK: !1 = !{i64 -3706093650706652785}
