; RUN: llc %s -o - -mtriple=powerpc | FileCheck --check-prefix=REL %s
; RUN: llc %s -o - -mtriple=powerpc -relocation-model=pic | FileCheck --check-prefix=PLTREL %s
; RUN: llc %s -o - -mtriple=powerpc64 | FileCheck --check-prefix=REL %s
; RUN: llc %s -o - -mtriple=powerpc64 -relocation-model=pic | FileCheck --check-prefix=REL %s
; RUN: llc %s -o - -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 \
; RUN:   -verify-machineinstrs | FileCheck --check-prefix=LEP8 %s
; RUN: llc %s -o - -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr10 \
; RUN:   -verify-machineinstrs | FileCheck --check-prefix=LEP10 %s

@ifunc1 = dso_local ifunc void(), ptr @resolver
@ifunc2 = ifunc void(), ptr @resolver

define ptr @resolver() { ret ptr null }

define void @foo() #0 {
  ; REL-LABEL:    foo
  ; REL:          bl ifunc1{{$}}
  ; REL:          bl ifunc2{{$}}
  ; PLTREL-LABEL: foo
  ; PLTREL:       bl ifunc1@PLT+32768
  ; PLTREL:       bl ifunc2@PLT+32768
  ; LEP8-LABEL:   foo
  ; LEP8:         bl ifunc1
  ; LEP8-NEXT:    nop
  ; LEP8-NEXT:    bl ifunc2
  ; LEP8-NEXT:    nop
  ; LEP8:         blr
  ; LEP10-LABEL:  foo
  ; LEP10:        bl ifunc1@notoc
  ; LEP10-NEXT:   bl ifunc2@notoc
  ; LEP10-NOT:    nop
  ; LEP10:        blr
  call void @ifunc1()
  call void @ifunc2()
  ret void
}

;; Use Secure PLT ABI for PPC32.
attributes #0 = { "target-features"="+secure-plt" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"PIC Level", i32 2}
