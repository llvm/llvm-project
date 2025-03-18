; RUN: llc -mtriple x86_64-linux-gnu -data-sections %s -o - | FileCheck %s --check-prefix=ELF
; RUN: llc -mtriple x86_64-linux-gnu -unique-section-names=0 -data-sections %s -o - | FileCheck %s --check-prefix=ELF-NOUNIQ

; RUN: llc -mtriple x86_64-windows-msvc -data-sections %s -o - | FileCheck %s --check-prefix=COFF-MSVC

; ELF: .section .data.hot.foo,
; ELF: .section .data.bar,
; ELF: .section .bss.unlikely.baz,
; ELF: .section .bss.quz,

; ELF-NOUNIQ: .section    .data.hot.,"aw",@progbits,unique,1
; ELF-NOUNIQ: .section    .data,"aw",@progbits,unique,2
; ELF-NOUNIQ: .section    .bss.unlikely.,"aw",@nobits,unique,3
; ELF-NOUNIQ: .section    .bss,"aw",@nobits,unique,4

; COFF-MSVC: .section .data,"dw",one_only,foo
; COFF-MSVC: .section .data,"dw",one_only,bar
; COFF-MSVC: .section .bss,"bw",one_only,baz
; COFF-MSVC: .section .bss,"bw",one_only,quz

@foo = global i32 1, !section_prefix !0
@bar = global i32 2
@baz = global i32 0, !section_prefix !1
@quz = global i32 0

!0 = !{!"section_prefix", !"hot"}
!1 = !{!"section_prefix", !"unlikely"}
