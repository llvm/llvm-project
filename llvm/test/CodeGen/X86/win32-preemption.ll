; RUN: llc -mtriple x86_64-pc-win32 \
; RUN:     -relocation-model=static  < %s | FileCheck --check-prefix=COFF_S %s
; RUN: llc -mtriple x86_64-pc-win32 \
; RUN:     -relocation-model=pic     < %s | FileCheck --check-prefix=COFF %s
; RUN: llc -mtriple x86_64-pc-win32 \
; RUN:  -relocation-model=dynamic-no-pic < %s | FileCheck --check-prefix=COFF %s


; 32 bits

; RUN: llc -mtriple i386-pc-win32 \
; RUN:    -relocation-model=static  < %s | FileCheck --check-prefix=COFF32 %s
; RUN: llc -mtriple i386-pc-win32 \
; RUN:     -relocation-model=pic     < %s | FileCheck --check-prefix=COFF32 %s
; RUN: llc -mtriple i386-pc-win32 \
; RUN:   -relocation-model=dynamic-no-pic < %s | \
; RUN:   FileCheck --check-prefix=COFF32 %s

; globals

@strong_default_global = global i32 42
define ptr @get_strong_default_global() {
  ret ptr @strong_default_global
}
; COFF: leaq strong_default_global(%rip), %rax
; COFF_S: movl $strong_default_global, %eax
; COFF32: movl $_strong_default_global, %eax

@weak_default_global = weak global i32 42
define ptr @get_weak_default_global() {
  ret ptr @weak_default_global
}
; COFF: leaq weak_default_global(%rip), %rax
; COFF_S: movl $weak_default_global, %eax
; COFF32: movl $_weak_default_global, %eax

@external_default_global = external global i32
define ptr @get_external_default_global() {
  ret ptr @external_default_global
}
; COFF: leaq external_default_global(%rip), %rax
; COFF_S: movl $external_default_global, %eax
; COFF32: movl $_external_default_global, %eax


@strong_local_global = dso_local global i32 42
define ptr @get_strong_local_global() {
  ret ptr @strong_local_global
}
; COFF: leaq strong_local_global(%rip), %rax
; COFF_S: movl $strong_local_global, %eax
; COFF32: movl $_strong_local_global, %eax

@weak_local_global = weak dso_local global i32 42
define ptr @get_weak_local_global() {
  ret ptr @weak_local_global
}
; COFF: leaq weak_local_global(%rip), %rax
; COFF_S: movl $weak_local_global, %eax
; COFF32: movl $_weak_local_global, %eax

@external_local_global = external dso_local global i32
define ptr @get_external_local_global() {
  ret ptr @external_local_global
}
; COFF: leaq external_local_global(%rip), %rax
; COFF_S: movl $external_local_global, %eax
; COFF32: movl $_external_local_global, %eax


@strong_preemptable_global = dso_preemptable global i32 42
define ptr @get_strong_preemptable_global() {
  ret ptr @strong_preemptable_global
}
; COFF: leaq strong_preemptable_global(%rip), %rax
; COFF_S: movl $strong_preemptable_global, %eax
; COFF32: movl $_strong_preemptable_global, %eax

@weak_preemptable_global = weak dso_preemptable global i32 42
define ptr @get_weak_preemptable_global() {
  ret ptr @weak_preemptable_global
}
; COFF: leaq weak_preemptable_global(%rip), %rax
; COFF_S: movl $weak_preemptable_global, %eax
; COFF32: movl $_weak_preemptable_global, %eax

@external_preemptable_global = external dso_preemptable global i32
define ptr @get_external_preemptable_global() {
  ret ptr @external_preemptable_global
}
; COFF: leaq external_preemptable_global(%rip), %rax
; COFF_S: movl $external_preemptable_global, %eax
; COFF32: movl $_external_preemptable_global, %eax


; aliases
@aliasee = global i32 42

@strong_default_alias = alias i32, ptr @aliasee
define ptr @get_strong_default_alias() {
  ret ptr @strong_default_alias
}
; COFF: leaq strong_default_alias(%rip), %rax
; COFF_S: movl $strong_default_alias, %eax
; COFF32: movl $_strong_default_alias, %eax

@weak_default_alias = weak alias i32, ptr @aliasee
define ptr @get_weak_default_alias() {
  ret ptr @weak_default_alias
}
; COFF: leaq weak_default_alias(%rip), %rax
; COFF_S: movl $weak_default_alias, %eax
; COFF32: movl $_weak_default_alias, %eax


@strong_local_alias = dso_local alias i32, ptr @aliasee
define ptr @get_strong_local_alias() {
  ret ptr @strong_local_alias
}
; COFF: leaq strong_local_alias(%rip), %rax
; COFF_S: movl $strong_local_alias, %eax
; COFF32: movl $_strong_local_alias, %eax

@weak_local_alias = weak dso_local alias i32, ptr @aliasee
define ptr @get_weak_local_alias() {
  ret ptr @weak_local_alias
}
; COFF: leaq weak_local_alias(%rip), %rax
; COFF_S: movl $weak_local_alias, %eax
; COFF32: movl $_weak_local_alias, %eax


@strong_preemptable_alias = dso_preemptable alias i32, ptr @aliasee
define ptr @get_strong_preemptable_alias() {
  ret ptr @strong_preemptable_alias
}
; COFF: leaq strong_preemptable_alias(%rip), %rax
; COFF_S: movl $strong_preemptable_alias, %eax
; COFF32: movl $_strong_preemptable_alias, %eax

@weak_preemptable_alias = weak dso_preemptable alias i32, ptr @aliasee
define ptr @get_weak_preemptable_alias() {
  ret ptr @weak_preemptable_alias
}
; COFF: leaq weak_preemptable_alias(%rip), %rax
; COFF_S: movl $weak_preemptable_alias, %eax
; COFF32: movl $_weak_preemptable_alias, %eax


; functions

define void @strong_default_function() {
  ret void
}
define ptr @get_strong_default_function() {
  ret ptr @strong_default_function
}
; COFF: leaq strong_default_function(%rip), %rax
; COFF_S: movl $strong_default_function, %eax
; COFF32: movl $_strong_default_function, %eax

define weak void @weak_default_function() {
  ret void
}
define ptr @get_weak_default_function() {
  ret ptr @weak_default_function
}
; COFF: leaq weak_default_function(%rip), %rax
; COFF_S: movl $weak_default_function, %eax
; COFF32: movl $_weak_default_function, %eax

declare void @external_default_function()
define ptr @get_external_default_function() {
  ret ptr @external_default_function
}
; COFF: leaq external_default_function(%rip), %rax
; COFF_S: movl $external_default_function, %eax
; COFF32: movl $_external_default_function, %eax


define dso_local void @strong_local_function() {
  ret void
}
define ptr @get_strong_local_function() {
  ret ptr @strong_local_function
}
; COFF: leaq strong_local_function(%rip), %rax
; COFF_S: movl $strong_local_function, %eax
; COFF32: movl $_strong_local_function, %eax

define weak dso_local void @weak_local_function() {
  ret void
}
define ptr @get_weak_local_function() {
  ret ptr @weak_local_function
}
; COFF: leaq weak_local_function(%rip), %rax
; COFF_S: movl $weak_local_function, %eax
; COFF32: movl $_weak_local_function, %eax

declare dso_local void @external_local_function()
define ptr @get_external_local_function() {
  ret ptr @external_local_function
}
; COFF: leaq external_local_function(%rip), %rax
; COFF_S: movl $external_local_function, %eax
; COFF32: movl $_external_local_function, %eax


define dso_preemptable void @strong_preemptable_function() {
  ret void
}
define ptr @get_strong_preemptable_function() {
  ret ptr @strong_preemptable_function
}
; COFF: leaq strong_preemptable_function(%rip), %rax
; COFF_S: movl $strong_preemptable_function, %eax
; COFF32: movl $_strong_preemptable_function, %eax

define weak dso_preemptable void @weak_preemptable_function() {
  ret void
}
define ptr @get_weak_preemptable_function() {
  ret ptr @weak_preemptable_function
}
; COFF: leaq weak_preemptable_function(%rip), %rax
; COFF_S: movl $weak_preemptable_function, %eax
; COFF32: movl $_weak_preemptable_function, %eax

declare dso_preemptable void @external_preemptable_function()
define ptr @get_external_preemptable_function() {
  ret ptr @external_preemptable_function
}
; COFF: leaq external_preemptable_function(%rip), %rax
; COFF_S: movl $external_preemptable_function, %eax
; COFF32: movl $_external_preemptable_function, %eax
