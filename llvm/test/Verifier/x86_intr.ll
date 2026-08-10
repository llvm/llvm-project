; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: Calling convention parameter requires byval
; CHECK-NEXT: ptr @non_ptr_arg0
define x86_intrcc void @non_ptr_arg0(i32) {
  ret void
}

; CHECK: Calling convention parameter requires byval
; CHECK-NEXT: ptr @non_byval_ptr_arg0
define x86_intrcc void @non_byval_ptr_arg0(ptr) {
  ret void
}

; CHECK: Calling convention parameter requires byval
; CHECK-NEXT: ptr @non_ptr_arg0_decl
declare x86_intrcc void @non_ptr_arg0_decl(i32)

; CHECK: Calling convention parameter requires byval
; CHECK-NEXT: ptr @non_byval_ptr_arg0_decl
declare x86_intrcc void @non_byval_ptr_arg0_decl(ptr)
