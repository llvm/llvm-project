; RUN: llc -mtriple armv7a--none-eabi < %s -enable-ipra | FileCheck %s

; A linkone_odr function (the same applies to available_externally, linkonce,
; weak, common, extern_weak and weak_odr) could be replaced with a
; differently-compiled version of the same source at link time, which might use
; different registers, so we can't do IPRA on it.
define linkonce_odr void @leaf_linkonce_odr() {
entry:
  ret void
}
define void @test_linkonce_odr() {
; CHECK-LABEL: test_linkonce_odr:
entry:
; CHECK: ASM1: r3
; CHECK: mov   [[TEMP:r[0-9]+]], r3
; CHECK: bl    leaf_linkonce_odr
; CHECK: mov   r3, [[TEMP]]
; CHECK: ASM2: r3
  %0 = tail call i32 asm sideeffect "// ASM1: $0", "={r3},0"(i32 undef)
  tail call void @leaf_linkonce_odr()
  %1 = tail call i32 asm sideeffect "// ASM2: $0", "={r3},0"(i32 %0)
  ret void
}

; This function has external linkage (the same applies to private, internal and
; appending), so the version we see here is guaranteed to be the version
; selected by the linker, so we can do IPRA.
define external void @leaf_external() {
entry:
  ret void
}
define void @test_external() {
; CHECK-LABEL: test_external:
entry:
; CHECK: ASM1: r3
; CHECK-NOT:   r3
; CHECK: bl    leaf_external
; CHECK-NOT:   r3
; CHECK: ASM2: r3
  %0 = tail call i32 asm sideeffect "// ASM1: $0", "={r3},0"(i32 undef)
  tail call void @leaf_external()
  %1 = tail call i32 asm sideeffect "// ASM2: $0", "={r3},0"(i32 %0)
  ret void
}
