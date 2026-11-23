; RUN: opt < %s -passes='require<globals-aa>,aa-eval' -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; Make sure memory effects of optnone functions in an SCC are correctly
; handled.

@g = internal global i32 0
@ctl = global i32 0

define internal void @A() noinline optnone {
entry:
  store i32 42, ptr @g
  %c = load i32, ptr @ctl
  %t = icmp ne i32 %c, 0
  br i1 %t, label %rec, label %done

rec:
  call void @B()
  br label %done

done:
  ret void
}

define internal void @B() noinline {
  call void @A()
  ret void
}

; CHECK-LABEL: Function: main
; CHECK: Both ModRef:  Ptr: i32* @g	<->  call void @A()
define i32 @main() {
  store i32 1, ptr @g
  call void @A()
  %v = load i32, ptr @g
  ret i32 %v
}
