; RUN: llc < %s -emulated-tls -relocation-model=pic -mtriple=i686-unknown-linux-gnu -fast-isel | FileCheck %s
; RUN: llc < %s -relocation-model=pic -mtriple=i686-unknown-linux-gnu -fast-isel \
; RUN: | FileCheck -check-prefix=NoEMU %s
; RUN: llc < %s -relocation-model=pic -mtriple=i686-linux-android29 -fast-isel \
; RUN: | FileCheck -check-prefix=NoEMU %s
; PR3654

; NoEMU-NOT: __emutls

@v = thread_local global i32 0
define i32 @f() nounwind {
entry:
          %t = load i32, ptr @v
          %s = add i32 %t, 1
          ret i32 %s
}

; CHECK-LABEL: f:
; CHECK:      movl __emutls_v.v@GOT(%ebx), %eax
; CHECK-NEXT: movl %eax, (%esp)
; CHECK-NEXT: calll __emutls_get_address@PLT
; CHECK-NEXT: movl (%eax), %eax

@alias = internal alias i32, ptr @v
define i32 @f_alias() nounwind {
entry:
          %t = load i32, ptr @v
          %s = add i32 %t, 1
          ret i32 %s
}

; CHECK-LABEL: f_alias:
; CHECK:      movl __emutls_v.v@GOT(%ebx), %eax
; CHECK-NEXT: movl %eax, (%esp)
; CHECK-NEXT: calll __emutls_get_address@PLT
; CHECK-NEXT: movl (%eax), %eax

; Use my_emutls_get_address like __emutls_get_address.
@my_emutls_v_xyz = external global ptr, align 4
declare ptr @my_emutls_get_address(ptr)

define i32 @my_get_xyz() {
entry:
  %call = call ptr @my_emutls_get_address(ptr @my_emutls_v_xyz)
  %0 = load i32, ptr %call, align 4
  ret i32 %0
}

; CHECK-LABEL: my_get_xyz:
; CHECK:      movl my_emutls_v_xyz@GOT(%ebx), %eax
; CHECK-NEXT: movl %eax, (%esp)
; CHECK-NEXT: calll my_emutls_get_address@PLT
; CHECK-NEXT: movl (%eax), %eax
