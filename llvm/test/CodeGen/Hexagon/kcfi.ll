; RUN: llc -mtriple=hexagon -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=hexagon -verify-machineinstrs -stop-after=finalize-isel < %s \
; RUN:   | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=hexagon -verify-machineinstrs -stop-after=kcfi < %s \
; RUN:   | FileCheck %s --check-prefix=KCFI

; Verify KCFI type hash is emitted before the function.
; ASM:       .word 12345678
; ASM-LABEL: f1:

define void @f1(ptr noundef %x) !kcfi_type !1 {
; Load and type-hash materialization are combined in one packet.
; ASM:       r{{[0-9]+}} = memw(r0+#-4)
; ASM-NEXT:  r{{[0-9]+}} = ##12345678
; ASM-NEXT:  }
; ASM-NEXT:  {
; ASM-NEXT:    p0 = cmp.eq(r{{[0-9]+}},r{{[0-9]+}})
; ASM-NEXT:    if (p0.new) jump:t
; ASM-NEXT:  }
; ASM:       r{{[0-9]+}}:{{[0-9]+}} = memd(##3134984174)

; After ISel, the call should carry a cfi-type.
; ISEL-LABEL: name: f1
; ISEL:       J2_callr %0,{{.*}} cfi-type 12345678

; After the KCFI pass, the check and call are bundled.
; KCFI-LABEL: name: f1
; KCFI:       BUNDLE{{.*}} {
; KCFI-NEXT:    KCFI_CHECK $r0, 12345678
; KCFI-NEXT:    J2_callr killed $r0
; KCFI-NEXT:  }

  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

; Test with a second call using a different type hash.
define void @f2(ptr noundef %x) !kcfi_type !2 {
; ASM-LABEL: f2:
; ASM:       r{{[0-9]+}} = memw(r0+#-4)
; ASM-NEXT:  r{{[0-9]+}} = ##1234
; ASM-NEXT:  }
; ASM-NEXT:  {
; ASM-NEXT:    p0 = cmp.eq(r{{[0-9]+}},r{{[0-9]+}})
; ASM-NEXT:    if (p0.new) jump:t
; ASM-NEXT:  }
; ASM:       r{{[0-9]+}}:{{[0-9]+}} = memd(##3134984174)

  call void %x() [ "kcfi"(i32 1234) ]
  ret void
}

; Test with patchable-function-entry (nops placed after the label,
; so the KCFI offset is still -4).
define void @f3(ptr noundef %x) #0 {
; ASM-LABEL: f3:
; ASM:       nop
; ASM:       nop
; ASM:       r{{[0-9]+}} = memw(r0+#-4)
; ASM-NEXT:  r{{[0-9]+}} = ##12345678
; ASM-NEXT:  }
; ASM-NEXT:  {
; ASM-NEXT:    p0 = cmp.eq(r{{[0-9]+}},r{{[0-9]+}})
; ASM-NEXT:    if (p0.new) jump:t
; ASM-NEXT:  }
; ASM:       r{{[0-9]+}}:{{[0-9]+}} = memd(##3134984174)

  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

;; Test patchable-function-prefix: nops are placed before the function entry
;; (after the type hash), so the KCFI load offset is adjusted from -4 to
;; -(PrefixNops*4 + 4).
define void @f4_prefix(ptr noundef %x) #1 !kcfi_type !1 {
; ASM-LABEL: f4_prefix:
; ASM:       r{{[0-9]+}} = memw(r0+#-12)
; ASM-NEXT:  r{{[0-9]+}} = ##12345678
; ASM-NEXT:  }
; ASM-NEXT:  {
; ASM-NEXT:    p0 = cmp.eq(r{{[0-9]+}},r{{[0-9]+}})
; ASM-NEXT:    if (p0.new) jump:t
; ASM-NEXT:  }
; ASM:       r{{[0-9]+}}:{{[0-9]+}} = memd(##3134984174)

  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

;; Test patchable-function-prefix with 3 nops: offset = -(3*4+4) = -16.
define void @f5_prefix3(ptr noundef %x) #2 !kcfi_type !1 {
; ASM-LABEL: f5_prefix3:
; ASM:       r{{[0-9]+}} = memw(r0+#-16)
; ASM-NEXT:  r{{[0-9]+}} = ##12345678
; ASM-NEXT:  }
; ASM-NEXT:  {
; ASM-NEXT:    p0 = cmp.eq(r{{[0-9]+}},r{{[0-9]+}})
; ASM-NEXT:    if (p0.new) jump:t
; ASM-NEXT:  }
; ASM:       r{{[0-9]+}}:{{[0-9]+}} = memd(##3134984174)

  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

;; Test scratch register conflict: call target is R6. The default scratch
;; registers are R6/R7, so when the target occupies R6, the load scratch
;; must use R8 instead.
define void @f6_target_r6() {
; ASM-LABEL: f6_target_r6:
; ASM:       r8 = memw(r6+#-4)
; ASM-NEXT:  r7 = ##12345678
; ASM-NEXT:  }
; ASM-NEXT:  {
; ASM-NEXT:    p0 = cmp.eq(r8,r7)
; ASM-NEXT:    if (p0.new) jump:t
; ASM-NEXT:  }
; ASM:       r{{[0-9]+}}:{{[0-9]+}} = memd(##3134984174)

; KCFI-LABEL: name: f6_target_r6
; KCFI:       BUNDLE{{.*}} {
; KCFI-NEXT:    KCFI_CHECK $r6, 12345678
; KCFI-NEXT:    J2_callr{{.*}}killed $r6
; KCFI-NEXT:  }

  %target = call ptr asm sideeffect "", "={r6}"()
  call void %target() [ "kcfi"(i32 12345678) ]
  ret void
}

;; Test scratch register conflict: call target is R7. The type-hash scratch
;; must use R8 instead.
define void @f7_target_r7() {
; ASM-LABEL: f7_target_r7:
; ASM:       r6 = memw(r7+#-4)
; ASM-NEXT:  r8 = ##12345678
; ASM-NEXT:  }
; ASM-NEXT:  {
; ASM-NEXT:    p0 = cmp.eq(r6,r8)
; ASM-NEXT:    if (p0.new) jump:t
; ASM-NEXT:  }
; ASM:       r{{[0-9]+}}:{{[0-9]+}} = memd(##3134984174)

; KCFI-LABEL: name: f7_target_r7
; KCFI:       BUNDLE{{.*}} {
; KCFI-NEXT:    KCFI_CHECK $r7, 12345678
; KCFI-NEXT:    J2_callr{{.*}}killed $r7
; KCFI-NEXT:  }

  %target = call ptr asm sideeffect "", "={r7}"()
  call void %target() [ "kcfi"(i32 12345678) ]
  ret void
}

;; Test noreturn indirect call with KCFI (uses PS_callr_nr opcode).
define void @f8_noreturn(ptr noundef %x) {
; ASM-LABEL: f8_noreturn:
; ASM:       r{{[0-9]+}} = memw(r0+#-4)
; ASM-NEXT:  r{{[0-9]+}} = ##12345678
; ASM-NEXT:  }
; ASM-NEXT:  {
; ASM-NEXT:    p0 = cmp.eq(r{{[0-9]+}},r{{[0-9]+}})
; ASM-NEXT:    if (p0.new) jump:t
; ASM-NEXT:  }
; ASM:       r{{[0-9]+}}:{{[0-9]+}} = memd(##3134984174)

; ISEL-LABEL: name: f8_noreturn
; ISEL:       PS_callr_nr %0,{{.*}} cfi-type 12345678

; KCFI-LABEL: name: f8_noreturn
; KCFI:       BUNDLE{{.*}} {
; KCFI-NEXT:    KCFI_CHECK $r0, 12345678
; KCFI-NEXT:    PS_callr_nr killed $r0
; KCFI-NEXT:  }

  call void %x() #3 [ "kcfi"(i32 12345678) ]
  unreachable
}

; Verify the .kcfi_traps section is emitted.
; ASM:       .section .kcfi_traps

attributes #0 = { "patchable-function-entry"="2" }
attributes #1 = { "patchable-function-prefix"="2" }
attributes #2 = { "patchable-function-prefix"="3" }
attributes #3 = { noreturn }

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 12345678}
!2 = !{i32 1234}
