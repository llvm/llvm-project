; RUN: llc -mtriple x86_64-linux-gnu -function-sections %s -o - | FileCheck %s --check-prefix=ELF
; RUN: llc -mtriple x86_64-linux-gnu -unique-section-names=0 -function-sections %s -o - | FileCheck %s --check-prefix=ELF-NOUNIQ
; RUN: llc -mtriple x86_64-windows-msvc -function-sections %s -o - | FileCheck %s --check-prefix=COFF-MSVC
; RUN: llc -mtriple x86_64-windows-gnu -function-sections %s -o - | FileCheck %s --check-prefix=COFF-GNU

define void @foo1(i1 zeroext %0) nounwind !section_prefix !0 {
;; Check hot section name
; ELF:        .section  .text.hot.foo1,"ax",@progbits
; ELF-NOUNIQ: .section  .text.hot.,"ax",@progbits,unique,1
; COFF-MSVC:  .section  .text$hot,"xr",one_only,foo1
; COFF-GNU:   .section  .text$hot$foo1,"xr",one_only,foo1
  ret void
}

define void @foo2(i1 zeroext %0) nounwind !section_prefix !1 {
;; Check unlikely section name
; ELF:        .section  .text.unlikely.foo2,"ax",@progbits
; ELF-NOUNIQ: .section  .text.unlikely.,"ax",@progbits,unique,2
; COFF-MSVC:  .section  .text$unlikely,"xr",one_only,foo2
; COFF-GNU:   .section  .text$unlikely$foo2,"xr",one_only,foo2
  ret void
}

!0 = !{!"function_section_prefix", !"hot"}
!1 = !{!"function_section_prefix", !"unlikely"}

