; RUN: llc -mtriple i386-pc-win32 < %s \
; RUN:    | FileCheck --check-prefixes=CHECK,CHECK-MSVC %s
; RUN: llc -mtriple i386-pc-mingw32 < %s \
; RUN:    | FileCheck --check-prefixes=CHECK,CHECK-MINGW %s
; RUN: llc -mtriple i386-pc-mingw32 < %s \
; RUN:    | FileCheck --check-prefix=NOTEXPORTED %s

; CHECK: .text

; CHECK: .globl _notHidden
define void @notHidden() {
	ret void
}

; CHECK: .globl _f1
define hidden void @f1() {
	ret void
}

; CHECK: .globl _f2
define hidden void @f2() unnamed_addr {
	ret void
}

declare hidden void @notDefined()

; CHECK: .globl _stdfun@0
define hidden x86_stdcallcc void @stdfun() nounwind {
	ret void
}

; CHECK: .globl _lnk1
$lnk1 = comdat any

define linkonce_odr hidden void @lnk1() comdat {
	ret void
}

; CHECK: .globl _lnk2
$lnk2 = comdat any

define linkonce_odr hidden void @lnk2() alwaysinline comdat {
	ret void
}

; CHECK: .data
; CHECK: .globl _Var1
@Var1 = hidden global i32 1, align 4

; CHECK: .rdata,"dr"
; CHECK: .globl _Var2
@Var2 = hidden unnamed_addr constant i32 1

; CHECK: .comm _Var3
@Var3 = common hidden global i32 0, align 4

; CHECK: .globl "_complex-name"
@"complex-name" = hidden global i32 1, align 4

; CHECK: .globl _complex.name
@"complex.name" = hidden global i32 1, align 4


; Verify items that should not be marked hidden do not appear in the directives.
; We use a separate check prefix to avoid confusion between -NOT and -SAME.
; NOTEXPORTED: .section .drectve
; NOTEXPORTED-NOT: :notHidden
; NOTEXPORTED-NOT: :notDefined

; CHECK-MSVC-NOT: .section .drectve
; CHECK-MINGW: .section .drectve
; CHECK-MINGW: .ascii " -exclude-symbols:f1"
; CHECK-MINGW: .ascii " -exclude-symbols:f2"
; CHECK-MINGW: .ascii " -exclude-symbols:stdfun@0"
; CHECK-MINGW: .ascii " -exclude-symbols:lnk1"
; CHECK-MINGW: .ascii " -exclude-symbols:lnk2"
; CHECK-MINGW: .ascii " -exclude-symbols:Var1"
; CHECK-MINGW: .ascii " -exclude-symbols:Var2"
; CHECK-MINGW: .ascii " -exclude-symbols:Var3"
; CHECK-MINGW: .ascii " -exclude-symbols:\"complex-name\""
; CHECK-MINGW: .ascii " -exclude-symbols:\"complex.name\""
