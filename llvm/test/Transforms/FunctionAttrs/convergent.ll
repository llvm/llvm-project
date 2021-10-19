; RUN: opt -function-attrs -S < %s | FileCheck %s
; RUN: opt -passes=function-attrs -S < %s | FileCheck %s

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
  %a = call i32 @k() convergent
  ret i32 %a
}

; Convergent should not be removed on the function here.  Although the call is
; not explicitly convergent, it picks up the convergent attr from the callee.
;
; CHECK: Function Attrs
; CHECK-SAME: convergent
; CHECK-NEXT: define i32 @extern_non_convergent_call()
define i32 @extern_non_convergent_call() convergent {
  %a = call i32 @k()
  ret i32 %a
}

; CHECK: Function Attrs
; CHECK-SAME: convergent
; CHECK-NEXT: define i32 @indirect_convergent_call(
define i32 @indirect_convergent_call(i32 ()* %f) convergent {
   %a = call i32 %f() convergent
   ret i32 %a
}
; Give indirect_non_convergent_call the norecurse attribute so we get a
; "Function Attrs" comment in the output.
;
; CHECK: Function Attrs
; CHECK-NOT: convergent
; CHECK-NEXT: define i32 @indirect_non_convergent_call(
define i32 @indirect_non_convergent_call(i32 ()* %f) convergent norecurse {
   %a = call i32 %f()
   ret i32 %a
}

; CHECK: Function Attrs
; CHECK-SAME: convergent
; CHECK-NEXT: declare void @llvm.nvvm.barrier0()
declare void @llvm.nvvm.barrier0() convergent

; CHECK: Function Attrs
; CHECK-SAME: convergent
; CHECK-NEXT: define i32 @intrinsic()
define i32 @intrinsic() convergent {
  ; Implicitly convergent, because the intrinsic is convergent.
  call void @llvm.nvvm.barrier0()
  ret i32 0
}

; CHECK: Function Attrs
; CHECK-NOT: convergent
; CHECK-NEXT: define i32 @recursive1()
define i32 @recursive1() convergent {
  %a = call i32 @recursive2() convergent
  ret i32 %a
}

; CHECK: Function Attrs
; CHECK-NOT: convergent
; CHECK-NEXT: define i32 @recursive2()
define i32 @recursive2() convergent {
  %a = call i32 @recursive1() convergent
  ret i32 %a
}

; CHECK: Function Attrs
; CHECK-SAME: convergent
; CHECK-NEXT: define i32 @noopt()
define i32 @noopt() convergent optnone noinline {
  %a = call i32 @noopt_friend() convergent
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
