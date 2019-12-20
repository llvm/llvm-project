; RUN: llc %s -o - -mtriple=powerpc | FileCheck --check-prefix=REL %s
; RUN: llc %s -o - -mtriple=powerpc -relocation-model=pic | FileCheck --check-prefix=PLTREL %s
; RUN: llc %s -o - -mtriple=powerpc64 | FileCheck --check-prefix=REL %s
; RUN: llc %s -o - -mtriple=powerpc64 -relocation-model=pic | FileCheck --check-prefix=REL %s

@ifunc1 = dso_local ifunc void(), i8*()* @resolver
@ifunc2 = ifunc void(), i8*()* @resolver

define i8* @resolver() { ret i8* null }

define void @foo() #0 {
  ; REL: bl ifunc1{{$}}
  ; REL: bl ifunc2{{$}}
  ; PLTREL: bl ifunc1@PLT+32768
  ; PLTREL: bl ifunc2@PLT+32768
  call void @ifunc1()
  call void @ifunc2()
  ret void
}

;; Use Secure PLT ABI for PPC32.
attributes #0 = { "target-features"="+secure-plt" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"PIC Level", i32 2}
