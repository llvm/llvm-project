; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: cat common.ll authflag.ll > auth.ll
; RUN: cat common.ll authflag-bb.ll > auth-bb.ll

; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+pauth -relocation-model=pic \
; RUN:   -verify-machineinstrs < auth.ll | FileCheck --check-prefixes=CHECK,CHECKA %s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+pauth -relocation-model=pic \
; RUN:   -filetype=obj < auth.ll | llvm-readelf -r -s - | FileCheck --check-prefix=CHECK-OBJ %s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+pauth -relocation-model=pic \
; RUN:   -global-isel=1 -verify-machineinstrs < auth.ll 2>&1 | FileCheck --check-prefixes=CHECK,CHECKA %s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+pauth -relocation-model=pic \
; RUN:   -global-isel=1 -filetype=obj < auth.ll | llvm-readelf -r -s - | FileCheck --check-prefix=CHECK-OBJ %s

; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+pauth -relocation-model=pic \
; RUN:   -verify-machineinstrs < auth-bb.ll | FileCheck --check-prefixes=CHECK,CHECKB %s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+pauth -relocation-model=pic \
; RUN:   -filetype=obj < auth-bb.ll | llvm-readelf -r -s - | FileCheck --check-prefix=CHECK-OBJ %s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+pauth -relocation-model=pic \
; RUN:   -global-isel=1 -verify-machineinstrs < auth-bb.ll 2>&1 | FileCheck --check-prefixes=CHECK,CHECKB %s
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+pauth -relocation-model=pic \
; RUN:   -global-isel=1 -filetype=obj < auth-bb.ll | llvm-readelf -r -s - | FileCheck --check-prefix=CHECK-OBJ %s

;--- common.ll
@general_dynamic_var = external thread_local global i32

define i32 @test_generaldynamic() {
; CHECK-LABEL: test_generaldynamic:

  %val = load i32, ptr @general_dynamic_var
  ret i32 %val

; CHECK: adrp x[[TLSDESC_HI:[0-9]+]], :tlsdesc_auth:general_dynamic_var
; CHECK-NEXT: ldr x16, [x[[TLSDESC_HI]], :tlsdesc_auth_lo12:general_dynamic_var]
; CHECK-NEXT: add x0, x[[TLSDESC_HI]], :tlsdesc_auth_lo12:general_dynamic_var
; CHECK-NEXT: .tlsauthdesccall general_dynamic_var
; CHECKA-NEXT: blraa x16, x0
; CHECKB-NEXT: blrab x16, x0
; CHECK-NEXT: mrs x[[TPIDR:[0-9]+]], TPIDR_EL0
; CHECK-NEXT: ldr w0, [x[[TPIDR]], x0]

; CHECK-OBJ: R_AARCH64_AUTH_TLSDESC_ADR_PAGE21
; CHECK-OBJ: R_AARCH64_AUTH_TLSDESC_LD64_LO12
; CHECK-OBJ: R_AARCH64_AUTH_TLSDESC_ADD_LO12
; CHECK-OBJ: R_AARCH64_AUTH_TLSDESC_CALL
}

define ptr @test_generaldynamic_addr() {
; CHECK-LABEL: test_generaldynamic_addr:

  ret ptr @general_dynamic_var

; CHECK: adrp x[[TLSDESC_HI:[0-9]+]], :tlsdesc_auth:general_dynamic_var
; CHECK-NEXT: ldr x16, [x[[TLSDESC_HI]], :tlsdesc_auth_lo12:general_dynamic_var]
; CHECK-NEXT: add x0, x[[TLSDESC_HI]], :tlsdesc_auth_lo12:general_dynamic_var
; CHECK-NEXT: .tlsauthdesccall general_dynamic_var
; CHECKA-NEXT: blraa x16, x0
; CHECKB-NEXT: blrab x16, x0
; CHECK-NEXT: mrs [[TP:x[0-9]+]], TPIDR_EL0
; CHECK-NEXT: add x0, [[TP]], x0

; CHECK-OBJ: R_AARCH64_AUTH_TLSDESC_ADR_PAGE21
; CHECK-OBJ: R_AARCH64_AUTH_TLSDESC_LD64_LO12
; CHECK-OBJ: R_AARCH64_AUTH_TLSDESC_ADD_LO12
; CHECK-OBJ: R_AARCH64_AUTH_TLSDESC_CALL
}

;; Note: with signed TLSDESC, general dynamic model is always used,
;; even when local dynamic is requested.

@local_dynamic_var = external thread_local(localdynamic) global i32

define i32 @test_localdynamic() {
; CHECK-LABEL: test_localdynamic:

  %val = load i32, ptr @local_dynamic_var
  ret i32 %val

; CHECK: adrp x[[TLSDESC_HI:[0-9]+]], :tlsdesc_auth:local_dynamic_var
; CHECK-NEXT: ldr x16, [x[[TLSDESC_HI]], :tlsdesc_auth_lo12:local_dynamic_var]
; CHECK-NEXT: add x0, x[[TLSDESC_HI]], :tlsdesc_auth_lo12:local_dynamic_var
; CHECK-NEXT: .tlsauthdesccall local_dynamic_var
; CHECKA-NEXT: blraa x16, x0
; CHECKB-NEXT: blrab x16, x0
; CHECK-NEXT: mrs x[[TPIDR:[0-9]+]], TPIDR_EL0
; CHECK-NEXT: ldr w0, [x[[TPIDR]], x0]

; CHECK-OBJ: R_AARCH64_AUTH_TLSDESC_ADR_PAGE21
; CHECK-OBJ: R_AARCH64_AUTH_TLSDESC_LD64_LO12
; CHECK-OBJ: R_AARCH64_AUTH_TLSDESC_ADD_LO12
; CHECK-OBJ: R_AARCH64_AUTH_TLSDESC_CALL
}

define ptr @test_localdynamic_addr() {
; CHECK-LABEL: test_localdynamic_addr:

  ret ptr @local_dynamic_var

; CHECK: adrp x[[TLSDESC_HI:[0-9]+]], :tlsdesc_auth:local_dynamic_var
; CHECK-NEXT: ldr x16, [x[[TLSDESC_HI]], :tlsdesc_auth_lo12:local_dynamic_var]
; CHECK-NEXT: add x0, x[[TLSDESC_HI]], :tlsdesc_auth_lo12:local_dynamic_var
; CHECK-NEXT: .tlsauthdesccall local_dynamic_var
; CHECKA-NEXT: blraa x16, x0
; CHECKB-NEXT: blrab x16, x0
; CHECK-NEXT: mrs x[[TPIDR:[0-9]+]], TPIDR_EL0
; CHECK-NEXT: add x0, x[[TPIDR]], x0

; CHECK-OBJ: R_AARCH64_AUTH_TLSDESC_ADR_PAGE21
; CHECK-OBJ: R_AARCH64_AUTH_TLSDESC_LD64_LO12
; CHECK-OBJ: R_AARCH64_AUTH_TLSDESC_ADD_LO12
; CHECK-OBJ: R_AARCH64_AUTH_TLSDESC_CALL
}

@extern_weak_var = extern_weak thread_local global i32

define i32 @test_extern_weak() {
; CHECK-LABEL: test_extern_weak:

  %val = load i32, ptr @extern_weak_var
  ret i32 %val

; CHECK: adrp x[[TLSDESC_HI:[0-9]+]], :tlsdesc_auth:extern_weak_var
; CHECK-NEXT: ldr x16, [x[[TLSDESC_HI]], :tlsdesc_auth_lo12:extern_weak_var]
; CHECK-NEXT: add x0, x[[TLSDESC_HI]], :tlsdesc_auth_lo12:extern_weak_var
; CHECK-NEXT: .tlsauthdesccall extern_weak_var
; CHECKA-NEXT: blraa x16, x0
; CHECKB-NEXT: blrab x16, x0
; CHECK-NEXT: mrs x[[TPIDR:[0-9]+]], TPIDR_EL0
; CHECK-NEXT: ldr w0, [x[[TPIDR]], x0]

; CHECK-OBJ: R_AARCH64_AUTH_TLSDESC_ADR_PAGE21
; CHECK-OBJ: R_AARCH64_AUTH_TLSDESC_LD64_LO12
; CHECK-OBJ: R_AARCH64_AUTH_TLSDESC_ADD_LO12
; CHECK-OBJ: R_AARCH64_AUTH_TLSDESC_CALL
; CHECK-OBJ: 0000000000000000     0 TLS     WEAK   DEFAULT   UND extern_weak_var
}

;--- authflag.ll
!llvm.module.flags = !{!0}
!0 = !{i32 8, !"ptrauth-elf-got", i32 1}

;--- authflag-bb.ll
!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"ptrauth-elf-got", i32 1}
!1 = !{i32 1, !"ptrauth-keys", !"ibdbibdb"}
