; RUN: opt -function-specialization -force-function-specialization -S < %s | FileCheck %s

; Check that we don't crash and specialise on a function call with byval attribute.

; CHECK-NOT: wombat.{{[0-9]+}}

declare i32* @quux()
declare i32* @eggs()

define i32 @main() {
; CHECK:       bb:
; CHECK-NEXT:    tail call void @wombat(i8* undef, i64 undef, i64 undef, i32* byval(i32) bitcast (i32* ()* @quux to i32*))
; CHECK-NEXT:    tail call void @wombat(i8* undef, i64 undef, i64 undef, i32* byval(i32) bitcast (i32* ()* @eggs to i32*))
; CHECK-NEXT:    ret i32 undef
;
bb:
  tail call void @wombat(i8* undef, i64 undef, i64 undef, i32* byval(i32) bitcast (i32* ()* @quux to i32*))
  tail call void @wombat(i8* undef, i64 undef, i64 undef, i32* byval(i32) bitcast (i32* ()* @eggs to i32*))
  ret i32 undef
}

define internal void @wombat(i8* %arg, i64 %arg1, i64 %arg2, i32* byval(i32) %func) {
; CHECK:       bb2:
; CHECK-NEXT:    [[FUNPTR:%.*]] = bitcast i32* %func to i32* (i8*, i8*)*
; CHECK-NEXT:    [[TMP:%.*]] = tail call i32* [[FUNPTR]](i8* undef, i8* undef)
; CHECK-NEXT:    ret void
;
bb2:
  %mycall = bitcast i32* %func to i32* (i8*, i8*)*
  %tmp = tail call i32* %mycall(i8* undef, i8* undef)
  ret void
}
