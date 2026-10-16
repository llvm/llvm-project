; RUN: opt -passes=internalize-linkonce-odr -enable-linkonce-odr-internalization=likely-module-local -S < %s | FileCheck %s --check-prefixes=CHECK,MAIN
; RUN: opt -passes=internalize-linkonce-odr -enable-linkonce-odr-internalization=all -S < %s | FileCheck %s --check-prefixes=CHECK,ALL

$only_calls = comdat any
$shared = comdat any

; Only direct calls: made internal in place, removed from its comdat.
; CHECK: define internal i32 @only_calls(i32 %x) #0 {
define linkonce_odr hidden i32 @only_calls(i32 %x) #0 comdat {
  %r = add i32 %x, 1
  ret i32 %r
}

; Address taken: the original stays for that use, all direct calls (including
; the recursive ones in the original and in the clone) go to an internal clone.
; CHECK: define linkonce_odr i32 @address_taken(i32 %x) #0 {
; CHECK-NEXT: call i32 @address_taken.internal(
define linkonce_odr i32 @address_taken(i32 %x) #0 {
  %r = call i32 @address_taken(i32 %x)
  ret i32 %r
}

; Shares its comdat with a global: cannot leave the comdat, so it is cloned.
; CHECK: define linkonce_odr i32 @in_shared_comdat(i32 %x) #0 comdat($shared) {
@shared_var = linkonce_odr global i32 0, comdat($shared)
define linkonce_odr i32 @in_shared_comdat(i32 %x) #0 comdat($shared) {
  ret i32 %x
}

; Not marked likely module-local: only with =all.
; MAIN: define linkonce_odr i32 @from_header(i32 %x) {
; ALL: define internal i32 @from_header(i32 %x) {
define linkonce_odr i32 @from_header(i32 %x) {
  ret i32 %x
}

; Not linkonce_odr: untouched.
; CHECK: define weak_odr i32 @explicit_instantiation(i32 %x) #0 {
define weak_odr i32 @explicit_instantiation(i32 %x) #0 {
  ret i32 %x
}

; Address taken and not cloneable (inline asm may define labels): untouched.
; CHECK: define linkonce_odr void @has_asm() #0 {
; CHECK-NOT: @has_asm.internal
define linkonce_odr void @has_asm() #0 {
  call void asm sideeffect "nop", ""()
  ret void
}

; No calls: nothing to do.
; CHECK: define linkonce_odr void @never_called() #0 {
define linkonce_odr void @never_called() #0 {
  ret void
}

@fptrs = global [3 x ptr] [ptr @address_taken, ptr @has_asm, ptr @never_called]

; CHECK-LABEL: define i32 @caller(
; CHECK: call i32 @only_calls(
; CHECK: call i32 @address_taken.internal(
; CHECK: call i32 @in_shared_comdat.internal(
; MAIN: call i32 @from_header(
; ALL: call i32 @from_header(
; CHECK: call i32 @explicit_instantiation(
; CHECK: call void @has_asm(
define i32 @caller(i32 %x) {
  %a = call i32 @only_calls(i32 %x)
  %b = call i32 @address_taken(i32 %a)
  %c = call i32 @in_shared_comdat(i32 %b)
  %d = call i32 @from_header(i32 %c)
  %e = call i32 @explicit_instantiation(i32 %d)
  call void @has_asm()
  ret i32 %e
}

; Clones are appended to the module.
; CHECK: define internal i32 @address_taken.internal(i32 %x) unnamed_addr #0 {
; CHECK-NEXT: call i32 @address_taken.internal(
; CHECK: define internal i32 @in_shared_comdat.internal(i32 %x) unnamed_addr #0 {

attributes #0 = { "frontend-hint-likely-module-local" }
