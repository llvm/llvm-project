;; Test comdat functions.
; RUN: opt < %s -mtriple=x86_64-linux -passes=pgo-instr-gen,instrprof -S | FileCheck %s --check-prefixes=ELF
; RUN: opt < %s -mtriple=x86_64-windows -passes=pgo-instr-gen,instrprof -S | FileCheck %s --check-prefixes=COFF

$linkonceodr = comdat any
$weakodr = comdat any
$weak = comdat any
$linkonce = comdat any

;; profc/profd have hash suffixes. This definition doesn't have value profiling,
;; so definitions with the same name in other modules must have the same CFG and
;; cannot have value profiling, either. profd can be made private for ELF.
; ELF:   @__profc_linkonceodr.[[#]] = linkonce_odr hidden global {{.*}} comdat, align 8
; ELF:   @__profd_linkonceodr.[[#]] = private global {{.*}} comdat($__profc_linkonceodr.[[#]]), align 8
; COFF:  @__profc_linkonceodr.[[#]] = linkonce_odr hidden global {{.*}} comdat, align 8
; COFF:  @__profd_linkonceodr.[[#]] = linkonce_odr hidden global {{.*}} comdat, align 8
define linkonce_odr void @linkonceodr() comdat {
  ret void
}

;; weakodr in a comdat is not renamed. There is no guarantee that definitions in
;; other modules don't have value profiling. profd should be conservatively
;; non-private to prevent a caller from referencing a non-prevailing profd,
;; causing a linker error.
; ELF:   @__profc_weakodr = weak_odr hidden global {{.*}} comdat, align 8
; ELF:   @__profd_weakodr = weak_odr hidden global {{.*}} comdat($__profc_weakodr), align 8
; COFF:  @__profc_weakodr = weak_odr hidden global {{.*}} comdat, align 8
; COFF:  @__profd_weakodr = weak_odr hidden global {{.*}} comdat, align 8
define weak_odr void @weakodr() comdat {
  ret void
}

;; weak in a comdat is not renamed. There is no guarantee that definitions in
;; other modules don't have value profiling. profd should be conservatively
;; non-private to prevent a caller from referencing a non-prevailing profd,
;; causing a linker error.
; ELF:   @__profc_weak = weak hidden global {{.*}} comdat, align 8
; ELF:   @__profd_weak = weak hidden global {{.*}} comdat($__profc_weak), align 8
; COFF:  @__profc_weak = weak hidden global {{.*}} comdat, align 8
; COFF:  @__profd_weak = weak hidden global {{.*}} comdat, align 8
define weak void @weak() comdat {
  ret void
}

;; profc/profd have hash suffixes. This definition doesn't have value profiling,
;; so definitions with the same name in other modules must have the same CFG and
;; cannot have value profiling, either. profd can be made private for ELF.
; ELF:   @__profc_linkonce.[[#]] = linkonce hidden global {{.*}} comdat, align 8
; ELF:   @__profd_linkonce.[[#]] = private global {{.*}} comdat($__profc_linkonce.[[#]]), align 8
; COFF:  @__profc_linkonce.[[#]] = linkonce hidden global {{.*}} comdat, align 8
; COFF:  @__profd_linkonce.[[#]] = linkonce hidden global {{.*}} comdat, align 8
define linkonce void @linkonce() comdat {
  ret void
}

; Check that comdat aliases are hidden for all linkage types
; ELF:   @linkonceodr.local = linkonce_odr hidden alias void (), ptr @linkonceodr
; ELF:   @weakodr.local = weak_odr hidden alias void (), ptr @weakodr
; ELF:   @weak.local = weak hidden alias void (), ptr @weak
; ELF:   @linkonce.local = linkonce hidden alias void (), ptr @linkonce
