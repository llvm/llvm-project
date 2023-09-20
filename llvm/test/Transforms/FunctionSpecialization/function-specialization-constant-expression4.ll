; RUN: opt -passes="ipsccp<func-spec>" -force-specialization -S < %s | FileCheck %s

; Check that we don't crash and specialise on a function call with byval attribute.

; CHECK-NOT: wombat.specialized.{{[0-9]+}}

declare ptr @quux()
declare ptr @eggs()

define i32 @main() {
; CHECK:       bb:
; CHECK-NEXT:    tail call void @wombat(ptr undef, i64 undef, i64 undef, ptr byval(i32) @quux)
; CHECK-NEXT:    tail call void @wombat(ptr undef, i64 undef, i64 undef, ptr byval(i32) @eggs)
; CHECK-NEXT:    ret i32 undef
;
bb:
  tail call void @wombat(ptr undef, i64 undef, i64 undef, ptr byval(i32) @quux)
  tail call void @wombat(ptr undef, i64 undef, i64 undef, ptr byval(i32) @eggs)
  ret i32 undef
}

define internal void @wombat(ptr %arg, i64 %arg1, i64 %arg2, ptr byval(i32) %func) {
; CHECK:       bb2:
; CHECK-NEXT:    [[TMP:%.*]] = tail call ptr %func(ptr undef, ptr undef)
; CHECK-NEXT:    ret void
;
bb2:
  %tmp = tail call ptr %func(ptr undef, ptr undef)
  ret void
}
