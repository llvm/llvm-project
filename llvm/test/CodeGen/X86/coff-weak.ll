; RUN: llc -mtriple=i686-windows-msvc -o - %s | FileCheck -check-prefix=X86 %s
; RUN: llc -mtriple=x86_64-windows-msvc -o - %s | FileCheck -check-prefix=X64 %s

; X86: .weak _foo
; X64: .weak foo
define weak void @foo() {
  ret void
}

; X86: .weak _bar
; X64: .weak bar
define weak_odr void @bar() {
  ret void
}

; X86-NOT: .weak _bar_comdat
; X64-NOT: .weak bar_comdat
$bar_comdat = comdat any

define weak_odr void @bar_comdat() comdat {
  ret void
}

; X86: .weak _baz
; X64: .weak baz
define linkonce void @baz() {
  ret void
}

; X86-NOT: .weak _baz_comdat
; X64-NOT: .weak baz_comdat
$baz_comdat = comdat any

define linkonce void @baz_comdat() comdat {
  ret void
}

; X86: .weak _quux
; X64: .weak quux
define linkonce_odr void @quux() {
  ret void
}

; X86-NOT: .weak _quux_comdat
; X64-NOT: .weak quux_comdat
$quux_comdat = comdat any

define linkonce_odr void @quux_comdat() comdat {
  ret void
}
