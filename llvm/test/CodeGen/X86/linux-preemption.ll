; RUN: llc -mtriple x86_64-pc-linux -relocation-model=static < %s | \
; RUN:   FileCheck --check-prefixes=COMMON,STATIC %s
; RUN: llc -mtriple x86_64-pc-linux -relocation-model=pic < %s | \
; RUN:   FileCheck --check-prefixes=COMMON,CHECK %s
; RUN: llc -mtriple x86_64-pc-linux -relocation-model=dynamic-no-pic < %s | \
; RUN:   FileCheck --check-prefixes=COMMON,CHECK %s

; 32 bits

; RUN: llc -mtriple i386-pc-linux \
; RUN:     -relocation-model=pic     < %s | FileCheck --check-prefix=CHECK32 %s

; globals

@strong_default_global = global i32 42
define i32* @get_strong_default_global() {
  ret i32* @strong_default_global
}
; CHECK: movq strong_default_global@GOTPCREL(%rip), %rax
; STATIC: movl $strong_default_global, %eax
; CHECK32: movl strong_default_global@GOT(%eax), %eax

@strong_hidden_global = hidden global i32 42
define i32* @get_hidden_default_global() {
  ret i32* @strong_hidden_global
}
; CHECK: leaq strong_hidden_global(%rip), %rax
; STATIC: movl $strong_hidden_global, %eax
; CHECK32: leal strong_hidden_global@GOTOFF(%eax), %eax

@weak_default_global = weak global i32 42
define i32* @get_weak_default_global() {
  ret i32* @weak_default_global
}
; CHECK: movq weak_default_global@GOTPCREL(%rip), %rax
; STATIC: movl $weak_default_global, %eax
; CHECK32: movl weak_default_global@GOT(%eax), %eax

@external_default_global = external global i32
define i32* @get_external_default_global() {
  ret i32* @external_default_global
}
; CHECK: movq external_default_global@GOTPCREL(%rip), %rax
; STATIC: movq external_default_global@GOTPCREL(%rip), %rax
; CHECK32: movl external_default_global@GOT(%eax), %eax

@strong_local_global = dso_local global i32 42
define i32* @get_strong_local_global() {
  ret i32* @strong_local_global
}
; CHECK: leaq .Lstrong_local_global$local(%rip), %rax
; STATIC: movl $strong_local_global, %eax
; CHECK32: leal .Lstrong_local_global$local@GOTOFF(%eax), %eax

@weak_local_global = weak dso_local global i32 42
define i32* @get_weak_local_global() {
  ret i32* @weak_local_global
}
; CHECK: leaq weak_local_global(%rip), %rax
; STATIC: movl $weak_local_global, %eax
; CHECK32: leal weak_local_global@GOTOFF(%eax), %eax

@external_local_global = external dso_local global i32
define i32* @get_external_local_global() {
  ret i32* @external_local_global
}
; CHECK: leaq external_local_global(%rip), %rax
; STATIC: movl $external_local_global, %eax
; CHECK32: leal external_local_global@GOTOFF(%eax), %eax


@strong_preemptable_global = dso_preemptable global i32 42
define i32* @get_strong_preemptable_global() {
  ret i32* @strong_preemptable_global
}
; CHECK: movq strong_preemptable_global@GOTPCREL(%rip), %rax
; STATIC: movl $strong_preemptable_global, %eax
; CHECK32: movl strong_preemptable_global@GOT(%eax), %eax

@weak_preemptable_global = weak dso_preemptable global i32 42
define i32* @get_weak_preemptable_global() {
  ret i32* @weak_preemptable_global
}
; CHECK: movq weak_preemptable_global@GOTPCREL(%rip), %rax
; STATIC: movl $weak_preemptable_global, %eax
; CHECK32: movl weak_preemptable_global@GOT(%eax), %eax

@external_preemptable_global = external dso_preemptable global i32
define i32* @get_external_preemptable_global() {
  ret i32* @external_preemptable_global
}
; CHECK: movq external_preemptable_global@GOTPCREL(%rip), %rax
; STATIC: movq external_preemptable_global@GOTPCREL(%rip), %rax
; CHECK32: movl external_preemptable_global@GOT(%eax), %eax

; aliases
@aliasee = global i32 42

@strong_default_alias = alias i32, i32* @aliasee
define i32* @get_strong_default_alias() {
  ret i32* @strong_default_alias
}
; CHECK: movq strong_default_alias@GOTPCREL(%rip), %rax
; STATIC: movl $strong_default_alias, %eax
; CHECK32: movl strong_default_alias@GOT(%eax), %eax

@strong_hidden_alias = hidden alias i32, i32* @aliasee
define i32* @get_strong_hidden_alias() {
  ret i32* @strong_hidden_alias
}
; CHECK: leaq strong_hidden_alias(%rip), %rax
; STATIC: movl $strong_hidden_alias, %eax
; CHECK32: leal strong_hidden_alias@GOTOFF(%eax), %eax

@weak_default_alias = weak alias i32, i32* @aliasee
define i32* @get_weak_default_alias() {
  ret i32* @weak_default_alias
}
; CHECK: movq weak_default_alias@GOTPCREL(%rip), %rax
; STATIC: movl $weak_default_alias, %eax
; CHECK32: movl weak_default_alias@GOT(%eax), %eax

@strong_local_alias = dso_local alias i32, i32* @aliasee
define i32* @get_strong_local_alias() {
  ret i32* @strong_local_alias
}
; CHECK: leaq .Lstrong_local_alias$local(%rip), %rax
; STATIC: movl $strong_local_alias, %eax
; CHECK32: leal .Lstrong_local_alias$local@GOTOFF(%eax), %eax

@weak_local_alias = weak dso_local alias i32, i32* @aliasee
define i32* @get_weak_local_alias() {
  ret i32* @weak_local_alias
}
; CHECK: leaq weak_local_alias(%rip), %rax
; STATIC: movl $weak_local_alias, %eax
; CHECK32: leal weak_local_alias@GOTOFF(%eax), %eax


@strong_preemptable_alias = dso_preemptable alias i32, i32* @aliasee
define i32* @get_strong_preemptable_alias() {
  ret i32* @strong_preemptable_alias
}
; CHECK: movq strong_preemptable_alias@GOTPCREL(%rip), %rax
; STATIC: movl $strong_preemptable_alias, %eax
; CHECK32: movl strong_preemptable_alias@GOT(%eax), %eax

@weak_preemptable_alias = weak dso_preemptable alias i32, i32* @aliasee
define i32* @get_weak_preemptable_alias() {
  ret i32* @weak_preemptable_alias
}
; CHECK: movq weak_preemptable_alias@GOTPCREL(%rip), %rax
; STATIC: movl $weak_preemptable_alias, %eax
; CHECK32: movl weak_preemptable_alias@GOT(%eax), %eax

; functions

define void @strong_default_function() {
  ret void
}
define void()* @get_strong_default_function() {
  ret void()* @strong_default_function
}
; CHECK: movq strong_default_function@GOTPCREL(%rip), %rax
; STATIC: movl $strong_default_function, %eax
; CHECK32: movl strong_default_function@GOT(%eax), %eax

define hidden void @strong_hidden_function() {
  ret void
}
define void()* @get_strong_hidden_function() {
  ret void()* @strong_hidden_function
}
; CHECK: leaq strong_hidden_function(%rip), %rax
; STATIC: movl $strong_hidden_function, %eax
; CHECK32: leal strong_hidden_function@GOTOFF(%eax), %eax

define weak void @weak_default_function() {
  ret void
}
define void()* @get_weak_default_function() {
  ret void()* @weak_default_function
}
; CHECK: movq weak_default_function@GOTPCREL(%rip), %rax
; STATIC: movl $weak_default_function, %eax
; CHECK32: movl weak_default_function@GOT(%eax), %eax

declare void @external_default_function()
define void()* @get_external_default_function() {
  ret void()* @external_default_function
}
; CHECK: movq external_default_function@GOTPCREL(%rip), %rax
; STATIC: movq external_default_function@GOTPCREL(%rip), %rax
; CHECK32: movl external_default_function@GOT(%eax), %eax

define dso_local void @strong_local_function() {
  ret void
}
define void()* @get_strong_local_function() {
  ret void()* @strong_local_function
}
; COMMON:     {{^}}strong_local_function:
; CHECK-NEXT: .Lstrong_local_function$local:
; CHECK: leaq .Lstrong_local_function$local(%rip), %rax
; STATIC: movl $strong_local_function, %eax
; CHECK32: leal .Lstrong_local_function$local@GOTOFF(%eax), %eax

define weak dso_local void @weak_local_function() {
  ret void
}
define void()* @get_weak_local_function() {
  ret void()* @weak_local_function
}
; CHECK: leaq weak_local_function(%rip), %rax
; STATIC: movl $weak_local_function, %eax
; CHECK32: leal weak_local_function@GOTOFF(%eax), %eax

declare dso_local void @external_local_function()
define void()* @get_external_local_function() {
  ret void()* @external_local_function
}
; CHECK: leaq external_local_function(%rip), %rax
; STATIC: movl $external_local_function, %eax
; CHECK32: leal external_local_function@GOTOFF(%eax), %eax


define dso_preemptable void @strong_preemptable_function() {
  ret void
}
define void()* @get_strong_preemptable_function() {
  ret void()* @strong_preemptable_function
}
; CHECK: movq strong_preemptable_function@GOTPCREL(%rip), %rax
; STATIC: movl $strong_preemptable_function, %eax
; CHECK32: movl strong_preemptable_function@GOT(%eax), %eax

define weak dso_preemptable void @weak_preemptable_function() {
  ret void
}
define void()* @get_weak_preemptable_function() {
  ret void()* @weak_preemptable_function
}
; CHECK: movq weak_preemptable_function@GOTPCREL(%rip), %rax
; STATIC: movl $weak_preemptable_function, %eax
; CHECK32: movl weak_preemptable_function@GOT(%eax), %eax

declare dso_preemptable void @external_preemptable_function()
define void()* @get_external_preemptable_function() {
  ret void()* @external_preemptable_function
}
; CHECK: movq external_preemptable_function@GOTPCREL(%rip), %rax
; STATIC: movq external_preemptable_function@GOTPCREL(%rip), %rax
; CHECK32: movl external_preemptable_function@GOT(%eax), %eax

!llvm.module.flags = !{!0}
!0 = !{i32 7, !"PIC Level", i32 2}

; COMMON:     {{^}}strong_local_global:
; CHECK-NEXT: .Lstrong_local_global$local:

; COMMON:      .globl strong_default_alias
; COMMON-NEXT: .set strong_default_alias, aliasee
; COMMON-NEXT: .globl strong_hidden_alias
; COMMON-NEXT: .hidden strong_hidden_alias
; COMMON-NEXT: .set strong_hidden_alias, aliasee
; COMMON-NEXT: .weak weak_default_alias
; COMMON-NEXT: .set weak_default_alias, aliasee
; COMMON-NEXT: .globl strong_local_alias
; COMMON-NEXT: .set strong_local_alias, aliasee
; CHECK-NEXT:  .set .Lstrong_local_alias$local, aliasee
; COMMON-NEXT: .weak weak_local_alias
; COMMON-NEXT: .set weak_local_alias, aliasee
; COMMON-NEXT: .globl strong_preemptable_alias
; COMMON-NEXT: .set strong_preemptable_alias, aliasee
; COMMON-NEXT: .weak weak_preemptable_alias
; COMMON-NEXT: .set weak_preemptable_alias, aliasee
