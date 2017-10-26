; RUN: llc -mtriple x86_64-apple-darwin \
; RUN:     -relocation-model=static          < %s | FileCheck %s
; RUN: llc -mtriple x86_64-apple-darwin \
; RUN:     -relocation-model=pic             < %s | FileCheck %s
; RUN: llc -mtriple x86_64-apple-darwin \
; RUN:     -relocation-model=dynamic-no-pic  < %s | FileCheck %s

; 32 bits

; RUN: llc -mtriple i386-apple-darwin \
; RUN:    -relocation-model=static < %s | FileCheck --check-prefix=DARWIN32_S %s
; RUN: llc -mtriple i386-apple-darwin \
; RUN:     -relocation-model=pic     < %s | FileCheck --check-prefix=DARWIN32 %s
; RUN: llc -mtriple i386-apple-darwin \
; RUN:   -relocation-model=dynamic-no-pic < %s | \
; RUN:   FileCheck --check-prefix=DARWIN32_DNP %s

; globals

@strong_default_global = global i32 42
define i32* @get_strong_default_global() {
  ret i32* @strong_default_global
}
; CHECK: leaq _strong_default_global(%rip), %rax
; DARWIN32: leal _strong_default_global-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_strong_default_global, %eax
; DARWIN32_DNP: movl $_strong_default_global, %eax

@weak_default_global = weak global i32 42
define i32* @get_weak_default_global() {
  ret i32* @weak_default_global
}
; CHECK: movq _weak_default_global@GOTPCREL(%rip), %rax
; DARWIN32: movl L_weak_default_global$non_lazy_ptr-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_weak_default_global, %eax
; DARWIN32_DNP: movl L_weak_default_global$non_lazy_ptr, %eax

@external_default_global = external global i32
define i32* @get_external_default_global() {
  ret i32* @external_default_global
}
; CHECK: movq _external_default_global@GOTPCREL(%rip), %rax
; DARWIN32: movl L_external_default_global$non_lazy_ptr-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_external_default_global, %eax
; DARWIN32_DNP: movl L_external_default_global$non_lazy_ptr, %eax

@strong_local_global = dso_local global i32 42
define i32* @get_strong_local_global() {
  ret i32* @strong_local_global
}
; CHECK: leaq _strong_local_global(%rip), %rax
; DARWIN32: leal _strong_local_global-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_strong_local_global, %eax
; DARWIN32_DNP: movl $_strong_local_global, %eax

@weak_local_global = weak dso_local global i32 42
define i32* @get_weak_local_global() {
  ret i32* @weak_local_global
}
; CHECK: leaq _weak_local_global(%rip), %rax
; DARWIN32: leal _weak_local_global-L{{.}}$pb(%eax), %eax
; DARWIN32_S: movl $_weak_local_global, %eax
; DARWIN32_DNP: movl $_weak_local_global, %eax

@external_local_global = external dso_local global i32
define i32* @get_external_local_global() {
  ret i32* @external_local_global
}
; CHECK: leaq _external_local_global(%rip), %rax
; DARWIN32: movl L_external_local_global$non_lazy_ptr-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_external_local_global, %eax
; DARWIN32_DNP: movl $_external_local_global, %eax

@strong_preemptable_global = dso_preemptable global i32 42
define i32* @get_strong_preemptable_global() {
  ret i32* @strong_preemptable_global
}
; CHECK: leaq _strong_preemptable_global(%rip), %rax
; DARWIN32: leal _strong_preemptable_global-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_strong_preemptable_global, %eax
; DARWIN32_DNP: movl $_strong_preemptable_global, %eax

@weak_preemptable_global = weak dso_preemptable global i32 42
define i32* @get_weak_preemptable_global() {
  ret i32* @weak_preemptable_global
}
; CHECK: movq _weak_preemptable_global@GOTPCREL(%rip), %rax
; DARWIN32: movl L_weak_preemptable_global$non_lazy_ptr-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_weak_preemptable_global, %eax
; DARWIN32_DNP: movl L_weak_preemptable_global$non_lazy_ptr, %eax

@external_preemptable_global = external dso_preemptable global i32
define i32* @get_external_preemptable_global() {
  ret i32* @external_preemptable_global
}
; CHECK: movq _external_preemptable_global@GOTPCREL(%rip), %rax
; DARWIN32: movl L_external_preemptable_global$non_lazy_ptr-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_external_preemptable_global, %eax
; DARWIN32_DNP: movl L_external_preemptable_global$non_lazy_ptr, %eax

; aliases
@aliasee = global i32 42

@strong_default_alias = alias i32, i32* @aliasee
define i32* @get_strong_default_alias() {
  ret i32* @strong_default_alias
}
; CHECK: leaq _strong_default_alias(%rip), %rax
; DARWIN32: leal _strong_default_alias-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_strong_default_alias, %eax
; DARWIN32_DNP: movl $_strong_default_alias, %eax

@weak_default_alias = weak alias i32, i32* @aliasee
define i32* @get_weak_default_alias() {
  ret i32* @weak_default_alias
}
; CHECK: movq _weak_default_alias@GOTPCREL(%rip), %rax
; DARWIN32: movl L_weak_default_alias$non_lazy_ptr-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_weak_default_alias, %eax
; DARWIN32_DNP: movl L_weak_default_alias$non_lazy_ptr, %eax

@strong_local_alias = dso_local alias i32, i32* @aliasee
define i32* @get_strong_local_alias() {
  ret i32* @strong_local_alias
}
; CHECK: leaq _strong_local_alias(%rip), %rax
; DARWIN32: leal _strong_local_alias-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_strong_local_alias, %eax
; DARWIN32_DNP: movl $_strong_local_alias, %eax

@weak_local_alias = weak dso_local alias i32, i32* @aliasee
define i32* @get_weak_local_alias() {
  ret i32* @weak_local_alias
}
; CHECK: leaq _weak_local_alias(%rip), %rax
; DARWIN32: leal _weak_local_alias-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_weak_local_alias, %eax
; DARWIN32_DNP: movl $_weak_local_alias, %eax

@strong_preemptable_alias = dso_preemptable alias i32, i32* @aliasee
define i32* @get_strong_preemptable_alias() {
  ret i32* @strong_preemptable_alias
}
; CHECK: leaq _strong_preemptable_alias(%rip), %rax
; DARWIN32: leal _strong_preemptable_alias-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_strong_preemptable_alias, %eax
; DARWIN32_DNP: movl $_strong_preemptable_alias, %eax

@weak_preemptable_alias = weak dso_preemptable alias i32, i32* @aliasee
define i32* @get_weak_preemptable_alias() {
  ret i32* @weak_preemptable_alias
}
; CHECK: movq _weak_preemptable_alias@GOTPCREL(%rip), %rax
; DARWIN32: movl L_weak_preemptable_alias$non_lazy_ptr-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_weak_preemptable_alias, %eax
; DARWIN32_DNP: movl L_weak_preemptable_alias$non_lazy_ptr, %eax

; functions

define void @strong_default_function() {
  ret void
}
define void()* @get_strong_default_function() {
  ret void()* @strong_default_function
}
; CHECK: leaq _strong_default_function(%rip), %rax
; DARWIN32: leal _strong_default_function-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_strong_default_function, %eax
; DARWIN32_DNP: movl $_strong_default_function, %eax

define weak void @weak_default_function() {
  ret void
}
define void()* @get_weak_default_function() {
  ret void()* @weak_default_function
}
; CHECK: movq _weak_default_function@GOTPCREL(%rip), %rax
; DARWIN32: movl L_weak_default_function$non_lazy_ptr-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_weak_default_function, %eax
; DARWIN32_DNP: movl L_weak_default_function$non_lazy_ptr, %eax

declare void @external_default_function()
define void()* @get_external_default_function() {
  ret void()* @external_default_function
}
; CHECK: movq _external_default_function@GOTPCREL(%rip), %rax
; DARWIN32: movl L_external_default_function$non_lazy_ptr-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_external_default_function, %eax
; DARWIN32_DNP: movl L_external_default_function$non_lazy_ptr, %eax

define dso_local void @strong_local_function() {
  ret void
}
define void()* @get_strong_local_function() {
  ret void()* @strong_local_function
}
; CHECK: leaq _strong_local_function(%rip), %rax
; DARWIN32: leal _strong_local_function-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_strong_local_function, %eax
; DARWIN32_DNP: movl $_strong_local_function, %eax

define weak dso_local void @weak_local_function() {
  ret void
}
define void()* @get_weak_local_function() {
  ret void()* @weak_local_function
}
; CHECK: leaq _weak_local_function(%rip), %rax
; DARWIN32: leal _weak_local_function-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_weak_local_function, %eax
; DARWIN32_DNP: movl $_weak_local_function, %eax

declare dso_local void @external_local_function()
define void()* @get_external_local_function() {
  ret void()* @external_local_function
}
; CHECK: leaq _external_local_function(%rip), %rax
; DARWIN32: movl L_external_local_function$non_lazy_ptr-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_external_local_function, %eax
; DARWIN32_DNP: movl $_external_local_function, %eax

define dso_preemptable void @strong_preemptable_function() {
  ret void
}
define void()* @get_strong_preemptable_function() {
  ret void()* @strong_preemptable_function
}
; CHECK: leaq _strong_preemptable_function(%rip), %rax
; DARWIN32: leal _strong_preemptable_function-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_strong_preemptable_function, %eax
; DARWIN32_DNP: movl $_strong_preemptable_function, %eax

define weak dso_preemptable void @weak_preemptable_function() {
  ret void
}
define void()* @get_weak_preemptable_function() {
  ret void()* @weak_preemptable_function
}
; CHECK: movq _weak_preemptable_function@GOTPCREL(%rip), %rax
; DARWIN32: movl L_weak_preemptable_function$non_lazy_ptr-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_weak_preemptable_function, %eax
; DARWIN32_DNP: movl L_weak_preemptable_function$non_lazy_ptr, %eax

declare dso_preemptable void @external_preemptable_function()
define void()* @get_external_preemptable_function() {
  ret void()* @external_preemptable_function
}
; CHECK: movq _external_preemptable_function@GOTPCREL(%rip), %rax
; DARWIN32: movl L_external_preemptable_function$non_lazy_ptr-L{{.*}}$pb(%eax), %eax
; DARWIN32_S: movl $_external_preemptable_function, %eax
; DARWIN32_DNP: movl L_external_preemptable_function$non_lazy_ptr, %eax
