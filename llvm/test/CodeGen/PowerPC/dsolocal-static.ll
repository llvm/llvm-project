; RUN: llc -mtriple=ppc64le -relocation-model=static < %s | FileCheck %s

@default = global i32 55
define dso_local ptr @get_default_global() {
; CHECK-LABEL: get_default_global:
; CHECK:         addis 3, 2, .LC{{.*}}@toc@ha
; CHECK-NEXT:    ld 3, .LC{{.*}}@toc@l(3)
; CHECK-NEXT:    blr
  ret ptr @default
}

@local_global = dso_local global i32 55
define dso_local ptr @get_local_global() {
; CHECK-LABEL: get_local_global:
; CHECK:         addis 3, 2, local_global@toc@ha
; CHECK-NEXT:    addi 3, 3, local_global@toc@l
; CHECK-NEXT:    blr
  ret ptr @local_global
}

@preemptable_global = dso_preemptable global i32 42
define dso_local ptr @get_preemptable_global() {
; CHECK-LABEL: get_preemptable_global:
; CHECK:         addis 3, 2, .LC{{.*}}@toc@ha
; CHECK-NEXT:    ld 3, .LC{{.*}}@toc@l(3)
; CHECK-NEXT:    blr
  ret ptr @preemptable_global
}


@external_default_global = external global i32
define dso_local ptr @get_external_default_global() {
; CHECK-LABEL: get_external_default_global:
; CHECK:         addis 3, 2, .LC{{.*}}@toc@ha
; CHECK-NEXT:    ld 3, .LC{{.*}}@toc@l(3)
; CHECK-NEXT:    blr
  ret ptr @external_default_global
}

@external_local_global = external dso_local global i32
define dso_local ptr @get_external_local_global() {
; CHECK-LABEL: get_external_local_global:
; CHECK:         addis 3, 2, external_local_global@toc@ha
; CHECK-NEXT:    addi 3, 3, external_local_global@toc@l
; CHECK-NEXT:    blr
  ret ptr @external_local_global
}

@external_preemptable_global = external dso_preemptable global i32
define dso_local ptr @get_external_preemptable_global() {
; CHECK-LABEL: get_external_preemptable_global:
; CHECK:         addis 3, 2, .LC{{.*}}@toc@ha
; CHECK-NEXT:    ld 3, .LC{{.*}}@toc@l(3)
; CHECK-NEXT:    blr
  ret ptr @external_preemptable_global
}


; functions
define signext i32 @default_function(i32 %i) {
  ret i32 %i
}
define dso_local signext i32 @default_function_caller(i32 %i) {
; CHECK-LABEL: default_function_caller:
; CHECK:         bl default_function
; CHECK-NEXT:    nop
  %call = notail call signext i32 @default_function(i32 signext %i)
  ret i32 %call
}

define dso_local signext i32 @local_function(i32 %i) {
  ret i32 %i
}
define dso_local signext i32 @local_function_caller(i32 %i) {
; CHECK-LABEL: local_function_caller:
; CHECK:         bl local_function
; CHECK-NOT:     nop
; CHECK:         blr
  %call = notail call signext i32 @local_function(i32 signext %i)
  ret i32 %call
}

define dso_preemptable signext i32 @preemptable_function(i32 %i) {
  ret i32 %i
}
define dso_local signext i32 @preemptable_function_caller(i32 %i) {
; CHECK-LABEL: preemptable_function_caller:
; CHECK:         bl preemptable_function
; CHECK-NEXT:    nop
  %call = notail call signext i32 @preemptable_function(i32 signext %i)
  ret i32 %call
}


declare i32 @external_default_function(i32 %i)
define dso_local i32 @external_default_function_caller(i32 %i) {
; CHECK-LABEL: external_default_function_caller:
; CHECK:         bl external_default_function
; CHECK-NEXT:    nop
; CHECK:         blr
  %call = notail call signext i32 @external_default_function(i32 signext %i)
  ret i32 %call
}

declare dso_local i32 @external_local_function(i32 %i)
define dso_local i32 @external_local_function_caller(i32 %i) {
; CHECK-LABEL: external_local_function_caller:
; CHECK:         bl external_local_function
; CHECK-NEXT:    nop
  %call = notail call signext i32 @external_local_function(i32 signext %i)
  ret i32 %call
}

declare dso_preemptable i32 @external_preemptable_function(i32 %i)
define dso_local i32 @external_preemptable_function_caller(i32 %i) {
; CHECK-LABEL: external_preemptable_function_caller:
; CHECK:         bl external_preemptable_function
; CHECK-NEXT:    nop
  %call = notail call signext i32 @external_preemptable_function(i32 signext %i)
  ret i32 %call
}
