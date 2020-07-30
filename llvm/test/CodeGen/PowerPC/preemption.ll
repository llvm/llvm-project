; RUN: llc -mtriple powerpc64le-unknown-gnu-linux  -relocation-model=pic \
; RUN: < %s |  FileCheck %s
; RUN: llc -mtriple powerpc64le-unknown-gnu-linux -relocation-model=static \
; RUN: < %s |  FileCheck --check-prefix=STATIC %s
; RUN: llc -mtriple powerpc64le-unknown-gnu-linux -relocation-model=pic \
; RUN: < %s |  FileCheck %s

; globals

@strong_default = global i32 55
define i32* @get_strong_default() #0 {
  ret i32* @strong_default

; STATIC-LABEL: @get_strong_default
; STATIC: addis 3, 2, strong_default@toc@ha
; STATIC: addi 3, 3, strong_default@toc@l
; STATIC: blr

; CHECK-LABEL: @get_strong_default
; CHECK: addis 3, 2, .LC0@toc@ha
; CHECK: ld 3, .LC0@toc@l(3)
; CHECK: blr
}

@weak_default = weak global i32 55
define i32* @get_weak_default() #0 {
  ret i32* @weak_default

; STATIC-LABEL: @get_weak_default
; STATIC: addis 3, 2, weak_default@toc@ha
; STATIC: addi 3, 3, weak_default@toc@l
; STATIC: blr

; CHECK-LABEL: @get_weak_default
; CHECK: addis 3, 2, .LC1@toc@ha
; CHECK: ld 3, .LC1@toc@l(3)
; CHECK: blr
}

@external_default_global = external global i32
define i32* @get_external_default_global() {
  ret i32* @external_default_global

; STATIC-LABEL: @get_external_default_global
; STATIC: addis 3, 2, .LC0@toc@ha
; STATIC: ld 3, .LC0@toc@l(3)
; STATIC: blr

; CHECK-LABEL: @get_external_default_global
; CHECK: addis 3, 2, .LC2@toc@ha
; CHECK: ld 3, .LC2@toc@l(3)
; CHECK: blr
}


@strong_local_global = dso_local global i32 55
define i32* @get_strong_local_global() {
  ret i32* @strong_local_global

; STATIC-LABEL: @get_strong_local_global
; STATIC:       addis 3, 2, strong_local_global@toc@ha
; STATIC:       addi 3, 3, strong_local_global@toc@l
; STATIC:       blr

; CHECK-LABEL: @get_strong_local_global
; CHECK:       addis 3, 2, strong_local_global@toc@ha
; CHECK:       addi 3, 3, strong_local_global@toc@l
; CHECK:       blr
}

@weak_local_global = weak dso_local global i32 42
define i32* @get_weak_local_global() {
  ret i32* @weak_local_global

; STATIC-LABEL: @get_weak_local_global
; STATIC:       addis 3, 2, weak_local_global@toc@ha
; STATIC:       addi 3, 3, weak_local_global@toc@l
; STATIC:       blr

; CHECK-LABEL: @get_weak_local_global
; CHECK:       addis 3, 2, weak_local_global@toc@ha
; CHECK:       addi 3, 3, weak_local_global@toc@l
; CHECK:       blr
}

@external_local_global = external dso_local global i32
define i32* @get_external_local_global() {
  ret i32* @external_local_global
; STATIC-LABEL: @get_external_local_global
; STATIC:       addis 3, 2, external_local_global@toc@ha
; STATIC:       addi 3, 3, external_local_global@toc@l
; STATIC:       blr

; CHECK-LABEL: @get_external_local_global
; CHECK:       addis 3, 2, external_local_global@toc@ha
; CHECK:       addi 3, 3, external_local_global@toc@l
; CHECK:       blr
}

@strong_preemptable_global = dso_preemptable global i32 42
define i32* @get_strong_preemptable_global() {
  ret i32* @strong_preemptable_global

; STATIC-LABEL: @get_strong_preemptable_global
; STATIC: addis 3, 2, strong_preemptable_global@toc@ha
; STATIC: addi 3, 3, strong_preemptable_global@toc@l
; STATIC: blr

; CHECK-LABEL: @get_strong_preemptable_global
; CHECK: addis 3, 2, .LC3@toc@ha
; CHECK: ld 3, .LC3@toc@l(3)
; CHECK: blr
}

@weak_preemptable_global = weak dso_preemptable global i32 42
define i32* @get_weak_preemptable_global() {
  ret i32* @weak_preemptable_global

; STATIC-LABEL: @get_weak_preemptable_global
; STATIC: addis 3, 2, weak_preemptable_global@toc@ha
; STATIC: addi 3, 3, weak_preemptable_global@toc@l
; STATIC: blr

; CHECK-LABEL: @get_weak_preemptable_global
; CHECK: addis 3, 2, .LC4@toc@ha
; CHECK: ld 3, .LC4@toc@l(3)
; CHECK: blr
}

@external_preemptable_global = external dso_preemptable global i32
define i32* @get_external_preemptable_global() {
  ret i32* @external_preemptable_global

; STATIC-LABEL: @get_external_preemptable_global
; STATIC: addis 3, 2, .LC1@toc@ha
; STATIC: ld 3, .LC1@toc@l(3)
; STATIC: blr

; CHECK-LABEL: @get_external_preemptable_global
; CHECK: addis 3, 2, .LC5@toc@ha
; CHECK: ld 3, .LC5@toc@l(3)
; CHECK: blr
}

; functions
define signext i32 @strong_default_function(i32 %i) {
  ret i32 %i
}
define signext i32 @strong_default_function_caller(i32 %i) {
  %call = notail call signext i32 @strong_default_function(i32 signext %i)
  ret i32 %call

; STATIC-LABEL: @strong_default_function_caller
; STATIC:       bl strong_default_function
; STATIC-NOT:   nop
; STATIC:       blr

; CHECK-LABEL:  @strong_default_function_caller
; CHECK:        bl strong_default_function
; CHECK-NEXT:   nop
; CHECK:        blr
}

define weak signext i32 @weak_default_function(i32 %i) {
  ret i32 %i
}
define signext i32 @weak_default_function_caller(i32 %i) {
  %call = notail call signext i32 @weak_default_function(i32 signext %i)
  ret i32 %call

; STATIC-LABEL: @weak_default_function_caller
; STATIC:       bl weak_default_function
; STATIC-NOT:   nop
; STATIC:       blr

; CHECK-LABEL:  @weak_default_function_caller
; CHECK:        bl weak_default_function
; CHECK-NEXT:   nop
; CHECK:        blr
}


declare i32 @external_default_function(i32 %i)
define i32 @external_default_function_caller(i32 %i) {
  %call = notail call signext i32  @external_default_function(i32 signext %i)
  ret i32 %call

; STATIC-LABEL: @external_default_function_caller
; STATIC:       bl external_default_function
; STATIC-NEXT:  nop
; STATIC:       blr

; CHECK-LABEL:  @external_default_function_caller
; CHECK:        bl external_default_function
; CHECK-NEXT:   nop
; CHECK:        blr
}

define dso_local signext i32 @strong_local_function(i32 %i) {
  ret i32 %i
}
define signext i32 @strong_local_function_caller(i32 %i) {
  %call = notail call signext i32 @strong_local_function(i32 signext %i)
  ret i32 %call

; STATIC-LABEL: @strong_local_function_caller
; STATIC:       bl strong_local_function
; STATIC-NOT:   nop
; STATIC:       blr

; CHECK-LABEL:  @strong_local_function_caller
; CHECK:        bl strong_local_function
; CHECK-NOT:    nop
; CHECK:        blr
}

define weak dso_local signext i32 @weak_local_function(i32 %i) {
  ret i32 %i
}
define signext i32 @weak_local_function_caller(i32 %i) {
  %call = notail call signext i32 @weak_local_function(i32 signext %i)
  ret i32 %call

; STATIC-LABEL: @weak_local_function_caller
; STATIC:       bl weak_local_function
; STATIC-NOT:   nop
; STATIC:       blr

; CHECK-LABEL:  @weak_local_function_caller
; CHECK:        bl weak_local_function
; CHECK-NOT:    nop
; CHECK:        blr
}

declare dso_local i32 @external_local_function(i32 %i)
define i32 @external_local_function_caller(i32 %i) {
  %call = notail call signext i32  @external_local_function(i32 signext %i)
  ret i32 %call

; STATIC-LABEL: @external_local_function_caller
; STATIC:       bl external_local_function
; STATIC-NOT:  nop
; STATIC:       blr

; CHECK-LABEL:  @external_local_function_caller
; CHECK:        bl external_local_function
; CHECK-NOT:    nop
; CHECK:        blr
}

define dso_preemptable signext i32 @strong_preemptable_function(i32 %i) {
  ret i32 %i
}
define signext i32 @strong_preemptable_function_caller(i32 %i) {
  %call = notail call signext i32 @strong_preemptable_function(i32 signext %i)
  ret i32 %call

; STATIC-LABEL: @strong_preemptable_function_caller
; STATIC:       bl strong_preemptable_function
; STATIC-NOT:   nop
; STATIC:       blr

; CHECK-LABEL:  @strong_preemptable_function_caller
; CHECK:        bl strong_preemptable_function
; CHECK-NEXT:   nop
; CHECK:        blr
}

define weak dso_preemptable signext i32 @weak_preemptable_function(i32 %i) {
  ret i32 %i
}
define signext i32 @weak_preemptable_function_caller(i32 %i) {
  %call = notail call signext i32 @weak_preemptable_function(i32 signext %i)
  ret i32 %call

; STATIC-LABEL: @weak_preemptable_function_caller
; STATIC:       bl weak_preemptable_function
; STATIC-NOT:   nop
; STATIC:       blr

; CHECK-LABEL:  @weak_preemptable_function_caller
; CHECK:        bl weak_preemptable_function
; CHECK-NEXT:   nop
; CHECK:        blr
}

declare dso_preemptable i32 @external_preemptable_function(i32 %i)
define i32 @external_preemptable_function_caller(i32 %i) {
  %call = notail call signext i32  @external_preemptable_function(i32 signext %i)
  ret i32 %call

; STATIC-LABEL: @external_preemptable_function_caller
; STATIC:       bl external_preemptable_function
; STATIC-NEXT:   nop
; STATIC:       blr

; CHECK-LABEL:  @external_preemptable_function_caller
; CHECK:        bl external_preemptable_function
; CHECK-NEXT:    nop
; CHECK:        blr
}

