;; Place a global object in the llvm.used list in a unique section with the SHF_GNU_RETAIN flag.
; RUN: llc -mtriple=x86_64 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64 -data-sections=1 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64 -no-integrated-as -binutils-version=2.36 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64 -no-integrated-as -binutils-version=2.35 < %s | FileCheck %s --check-prefix=OLDGAS
;; Solaris uses the equivalent SHF_SUNW_NODISCARD flag, also represented as "R".
; RUN: llc -mtriple=x86_64-solaris < %s | FileCheck %s

; RUN: llc -mtriple=x86_64 -data-sections=1 -unique-section-names=0 < %s | FileCheck %s --check-prefix=NOUNIQUE

@llvm.used = appending global [10 x ptr] [
  ptr @fa, ptr @fb, ptr @fc,
  ptr @ga, ptr @gb, ptr @gc, ptr @gd, ptr @ge,
  ptr @aa, ptr @ab ], section "llvm.metadata"

; CHECK:    .section .text.fa,"axR",@progbits{{$}}
; OLDGAS-NOT: .section .text
; NOUNIQUE: .section .text,"axR",@progbits,unique,1
define dso_local void @fa() {
entry:
  ret void
}

; CHECK:    .section .text.fb,"axR",@progbits{{$}}
; NOUNIQUE: .section .text,"axR",@progbits,unique,2
define internal void @fb() {
entry:
  ret void
}

;; Explicit section.
; CHECK:    .section ccc,"axR",@progbits,unique,1
; OLDGAS:   .section ccc,"ax",@progbits,unique,1
; NOUNIQUE: .section ccc,"axR",@progbits,unique,3
define dso_local void @fc() section "ccc" {
entry:
  ret void
}

; CHECK:    .section .bss.ga,"awR",@nobits{{$}}
; OLDGAS:   .bss{{$}}
; NOUNIQUE: .section .bss,"awR",@nobits,unique,4
@ga = global i32 0

; CHECK:    .section .data.gb,"awR",@progbits{{$}}
; OLDGAS:   .data{{$}}
; NOUNIQUE: .section .data,"awR",@progbits,unique,5
@gb = internal global i32 2

; CHECK:    .section .rodata.gc,"aR",@progbits{{$}}
; OLDGAS:   .section .rodata,"a",@progbits{{$}}
; NOUNIQUE: .section .rodata,"aR",@progbits,unique,6
@gc = constant i32 3

;; Explicit section.
; CHECK:    .section ddd,"awR",@progbits,unique,2
; OLDGAS:   .section ddd,"aw",@progbits,unique,2
; NOUNIQUE: .section ddd,"awR",@progbits,unique,7
@gd = global i32 1, section "ddd"

;; Used together with !associated.
; CHECK:    .section .data.ge,"awoR",@progbits,gc
; OLDGAS:   .section .data.ge,"awo",@progbits,gc
; NOUNIQUE: .section .data,"awoR",@progbits,gc,unique,8
@ge = global i32 1, !associated !0

;; Aliases in llvm.used are ignored.
; CHECK:    .section fff,"aw",@progbits{{$}}
; OLDGAS:   .section fff,"aw",@progbits{{$}}
; NOUNIQUE: .section fff,"aw",@progbits{{$}}
@gf = global i32 1, section "fff"

@aa = alias i32, ptr @gf
@ab = internal alias i32, ptr @gf

!0 = !{ptr @gc}
