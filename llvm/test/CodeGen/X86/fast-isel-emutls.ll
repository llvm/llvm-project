; RUN: llc < %s -emulated-tls -march=x86 -relocation-model=pic -mtriple=i686-unknown-linux-gnu -fast-isel | FileCheck %s
; PR3654

@v = thread_local global i32 0
define i32 @f() nounwind {
entry:
          %t = load i32, i32* @v
          %s = add i32 %t, 1
          ret i32 %s
}

; CHECK-LABEL: f:
; CHECK:      movl __emutls_v.v@GOT(%ebx), %eax
; CHECK-NEXT: movl %eax, (%esp)
; CHECK-NEXT: calll __emutls_get_address@PLT
; CHECK-NEXT: movl (%eax), %eax

@alias = internal alias i32, i32* @v
define i32 @f_alias() nounwind {
entry:
          %t = load i32, i32* @v
          %s = add i32 %t, 1
          ret i32 %s
}

; CHECK-LABEL: f_alias:
; CHECK:      movl __emutls_v.v@GOT(%ebx), %eax
; CHECK-NEXT: movl %eax, (%esp)
; CHECK-NEXT: calll __emutls_get_address@PLT
; CHECK-NEXT: movl (%eax), %eax

; Use my_emutls_get_address like __emutls_get_address.
@my_emutls_v_xyz = external global i8*, align 4
declare i8* @my_emutls_get_address(i8*)

define i32 @my_get_xyz() {
entry:
  %call = call i8* @my_emutls_get_address(i8* bitcast (i8** @my_emutls_v_xyz to i8*))
  %0 = bitcast i8* %call to i32*
  %1 = load i32, i32* %0, align 4
  ret i32 %1
}

; CHECK-LABEL: my_get_xyz:
; CHECK:      movl my_emutls_v_xyz@GOT(%ebx), %eax
; CHECK-NEXT: movl %eax, (%esp)
; CHECK-NEXT: calll my_emutls_get_address@PLT
; CHECK-NEXT: movl (%eax), %eax
