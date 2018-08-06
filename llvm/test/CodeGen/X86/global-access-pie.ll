; RUN: llc < %s -emulated-tls -mcpu=generic -mtriple=x86_64-linux-gnu -relocation-model=pic \
; RUN:   | FileCheck -check-prefix=X64 %s
; RUN: llc < %s -emulated-tls -mcpu=generic -mtriple=i386-linux-gnu -relocation-model=pic \
; RUN:   | FileCheck -check-prefix=X32 %s

; RUN: llc < %s -mcpu=generic -mtriple=x86_64-linux-gnu -relocation-model=pic \
; RUN:   | FileCheck -check-prefix=X64 %s
; RUN: llc < %s -mcpu=generic -mtriple=i386-linux-gnu -relocation-model=pic \
; RUN:   | FileCheck -check-prefix=X32 %s

; External Linkage
@a = global i32 0, align 4

define i32 @my_access_global_a() #0 {
; X32-LABEL: my_access_global_a:
; X32:       addl $_GLOBAL_OFFSET_TABLE_{{.*}}, %eax
; X32-NEXT:  movl a@GOTOFF(%eax), %eax
; X64-LABEL: my_access_global_a:
; X64:       movl a(%rip), %eax

entry:
  %0 = load i32, i32* @a, align 4
  ret i32 %0
}

; WeakAny Linkage
@b = weak global i32 0, align 4

define i32 @my_access_global_b() #0 {
; X32-LABEL: my_access_global_b:
; X32:       addl $_GLOBAL_OFFSET_TABLE_{{.*}}, %eax
; X32-NEXT:  movl b@GOTOFF(%eax), %eax
; X64-LABEL: my_access_global_b:
; X64:       movl b(%rip), %eax

entry:
  %0 = load i32, i32* @b, align 4
  ret i32 %0
}

; Internal Linkage
@c = internal global i32 0, align 4

define i32 @my_access_global_c() #0 {
; X32-LABEL: my_access_global_c:
; X32:       addl $_GLOBAL_OFFSET_TABLE_{{.*}}, %eax
; X32-NEXT:  movl c@GOTOFF(%eax), %eax
; X64-LABEL: my_access_global_c:
; X64:       movl c(%rip), %eax

entry:
  %0 = load i32, i32* @c, align 4
  ret i32 %0
}

; External Linkage, only declaration.
@d = external global i32, align 4

define i32 @my_access_global_load_d() #0 {
; X32-LABEL: my_access_global_load_d:
; X32:       addl $_GLOBAL_OFFSET_TABLE_{{.*}}, %eax
; X32-NEXT:  movl d@GOT(%eax), %eax
; X32-NEXT:  movl (%eax), %eax
; X64-LABEL: my_access_global_load_d:
; X64:       movq d@GOTPCREL(%rip), %rax
; X64-NEXT:  movl (%rax), %eax

entry:
  %0 = load i32, i32* @d, align 4
  ret i32 %0
}

; External Linkage, only declaration, store a value.

define i32 @my_access_global_store_d() #0 {
; X32-LABEL: my_access_global_store_d:
; X32:       addl $_GLOBAL_OFFSET_TABLE_{{.*}}, %eax
; X32-NEXT:  movl d@GOT(%eax), %eax
; X32-NEXT:  movl $2, (%eax)
; X64-LABEL: my_access_global_store_d:
; X64:       movq d@GOTPCREL(%rip), %rax
; X64-NEXT:  movl $2, (%rax)

entry:
  store i32 2, i32* @d, align 4
  ret i32 0
}

; External Linkage, function pointer access.
declare i32 @access_fp(i32 ()*)
declare i32 @foo()

define i32 @my_access_fp_foo() #0 {
; X32-LABEL: my_access_fp_foo:
; X32:       addl $_GLOBAL_OFFSET_TABLE_{{.*}}, %ebx
; X32-NEXT:  movl	foo@GOT(%ebx), %eax
; X64-LABEL: my_access_fp_foo:
; X64:       movq foo@GOTPCREL(%rip), %rdi

entry:
  %call = call i32 @access_fp(i32 ()* @foo)
  ret i32 %call
}

; LinkOnceODR Linkage, function pointer access.

$bar = comdat any

define linkonce_odr i32 @bar() comdat {
entry:
  ret i32 0
}

define i32 @my_access_fp_bar() #0 {
; X32-LABEL: my_access_fp_bar:
; X32:       addl $_GLOBAL_OFFSET_TABLE_{{.*}}, %ebx
; X32-NEXT:  leal	bar@GOTOFF(%ebx), %eax
; X64-LABEL: my_access_fp_bar:
; X64:       leaq bar(%rip), %rdi

entry:
  %call = call i32 @access_fp(i32 ()* @bar)
  ret i32 %call
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"PIC Level", i32 1}
!1 = !{i32 1, !"PIE Level", i32 1}
