; RUN: opt < %s -basicaa -functionattrs -rpo-functionattrs -S | FileCheck %s

; CHECK: Function Attrs
; CHECK-NOT: convergent
; CHECK-NEXT: define i32 @nonleaf()
define i32 @nonleaf() convergent {
  %a = call i32 @leaf()
  ret i32 %a
}

; CHECK: Function Attrs
; CHECK-NOT: convergent
; CHECK-NEXT: define i32 @leaf()
define i32 @leaf() convergent {
  ret i32 0
}

; CHECK: Function Attrs
; CHECK-SAME: convergent
; CHECK-NEXT: declare i32 @k()
declare i32 @k() convergent

; CHECK: Function Attrs
; CHECK-SAME: convergent
; CHECK-NEXT: define i32 @extern()
define i32 @extern() convergent {
  %a = call i32 @k()
  ret i32 %a
}

; CHECK: Function Attrs
; CHECK-SAME: convergent
; CHECK-NEXT: define i32 @call_extern()
define i32 @call_extern() convergent {
  %a = call i32 @extern()
  ret i32 %a
}

; CHECK: Function Attrs
; CHECK-SAME: convergent
; CHECK-NEXT: declare void @llvm.cuda.syncthreads()
declare void @llvm.cuda.syncthreads() convergent

; CHECK: Function Attrs
; CHECK-SAME: convergent
; CHECK-NEXT: define i32 @intrinsic()
define i32 @intrinsic() convergent {
  call void @llvm.cuda.syncthreads()
  ret i32 0
}

@xyz = global i32 ()* null
; CHECK: Function Attrs
; CHECK-SAME: convergent
; CHECK-NEXT: define i32 @functionptr()
define i32 @functionptr() convergent {
  %1 = load i32 ()*, i32 ()** @xyz
  %2 = call i32 %1()
  ret i32 %2
}

; CHECK: Function Attrs
; CHECK-NOT: convergent
; CHECK-NEXT: define i32 @recursive1()
define i32 @recursive1() convergent {
  %a = call i32 @recursive2()
  ret i32 %a
}

; CHECK: Function Attrs
; CHECK-NOT: convergent
; CHECK-NEXT: define i32 @recursive2()
define i32 @recursive2() convergent {
  %a = call i32 @recursive1()
  ret i32 %a
}

; CHECK: Function Attrs
; CHECK-SAME: convergent
; CHECK-NEXT: define i32 @noopt()
define i32 @noopt() convergent optnone noinline {
  %a = call i32 @noopt_friend()
  ret i32 0
}

; A function which is mutually-recursive with a convergent, optnone function
; shouldn't have its convergent attribute stripped.
; CHECK: Function Attrs
; CHECK-SAME: convergent
; CHECK-NEXT: define i32 @noopt_friend()
define i32 @noopt_friend() convergent {
  %a = call i32 @noopt()
  ret i32 0
}
