; RUN: llc -mtriple x86_64-pc-win32   < %s | FileCheck -check-prefix=CHECK -check-prefix=WIN32 %s
; RUN: llc -mtriple x86_64-pc-mingw32 < %s | FileCheck -check-prefix=CHECK -check-prefix=MINGW %s
; RUN: llc -mtriple x86_64-pc-win32   < %s | FileCheck -check-prefix=NOTEXPORTED %s
; RUN: llc -mtriple x86_64-pc-mingw32 < %s | FileCheck -check-prefix=NOTEXPORTED %s

; CHECK: .text

define void @notExported() {
	ret void
}

; CHECK: .globl f1
define dllexport void @f1() {
	ret void
}

; CHECK: .globl f2
define dllexport void @f2() unnamed_addr {
	ret void
}

; CHECK: .globl lnk1
$lnk1 = comdat any

define linkonce_odr dllexport void @lnk1() comdat {
	ret void
}

; CHECK: .globl lnk2
$lnk2 = comdat any

define linkonce_odr dllexport void @lnk2() alwaysinline comdat {
	ret void
}

; CHECK: .weak weak1
define weak_odr dllexport void @weak1() {
	ret void
}


; CHECK: .data
; CHECK: .globl Var1
@Var1 = dllexport global i32 1, align 4

; CHECK: .rdata,"dr"
; CHECK: .globl Var2
@Var2 = dllexport unnamed_addr constant i32 1

; CHECK: .comm Var3
@Var3 = common dllexport global i32 0, align 4

; CHECK: .weak WeakVar1
@WeakVar1 = weak_odr dllexport global i32 1, align 4

; CHECK: .weak WeakVar2
@WeakVar2 = weak_odr dllexport unnamed_addr constant i32 1

; CHECK: .bss
; CHECK: .weak WeakVar3
@WeakVar3 = weak_odr dllexport global i32 0, align 4


; Verify items that should not be exported do not appear in the export table.
; We use a separate check prefix to avoid confusion between -NOT and -SAME.
; NOTEXPORTED: .section .drectve
; NOTEXPORTED-NOT: :notExported
; NOTEXPORTED-NOT: :aliasNotExported
; NOTEXPORTED-NOT: :exportedButNotDefinedVariable
; NOTEXPORTED-NOT: :exportedButNotDefinedFunction

; CHECK: .section .drectve
; WIN32: .ascii " /EXPORT:f1"
; WIN32: .ascii " /EXPORT:f2"
; WIN32: .ascii " /EXPORT:lnk1"
; WIN32: .ascii " /EXPORT:lnk2"
; WIN32: .ascii " /EXPORT:weak1"
; WIN32: .ascii " /EXPORT:Var1,DATA"
; WIN32: .ascii " /EXPORT:Var2,DATA"
; WIN32: .ascii " /EXPORT:Var3,DATA"
; WIN32: .ascii " /EXPORT:WeakVar1,DATA"
; WIN32: .ascii " /EXPORT:WeakVar2,DATA"
; WIN32: .ascii " /EXPORT:WeakVar3,DATA"
; WIN32: .ascii " /EXPORT:alias"
; WIN32: .ascii " /EXPORT:alias2"
; WIN32: .ascii " /EXPORT:alias3"
; WIN32: .ascii " /EXPORT:weak_alias"
; WIN32: .ascii " /EXPORT:blob_alias"
; MINGW: .ascii " -export:f1"
; MINGW: .ascii " -export:f2"
; MINGW: .ascii " -export:lnk1"
; MINGW: .ascii " -export:lnk2"
; MINGW: .ascii " -export:weak1"
; MINGW: .ascii " -export:Var1,data"
; MINGW: .ascii " -export:Var2,data"
; MINGW: .ascii " -export:Var3,data"
; MINGW: .ascii " -export:WeakVar1,data"
; MINGW: .ascii " -export:WeakVar2,data"
; MINGW: .ascii " -export:WeakVar3,data"
; MINGW: .ascii " -export:alias"
; MINGW: .ascii " -export:alias2"
; MINGW: .ascii " -export:alias3"
; MINGW: .ascii " -export:weak_alias"
; MINGW: .ascii " -export:blob_alias"

; CHECK: .globl alias
; CHECK: .set alias, notExported
@alias = dllexport alias void(), ptr @notExported

; CHECK: .globl aliasNotExported
; CHECK: .set aliasNotExported, f1
@aliasNotExported = alias void(), ptr @f1

; CHECK: .globl alias2
; CHECK: .set alias2, f1
@alias2 = dllexport alias void(), ptr @f1

; CHECK: .globl alias3
; CHECK: .set alias3, notExported
@alias3 = dllexport alias void(), ptr @notExported

; CHECK: .weak weak_alias
; CHECK: .set weak_alias, f1
@weak_alias = weak_odr dllexport alias void(), ptr @f1

@blob = global [6 x i8] c"\B8*\00\00\00\C3", section ".text", align 16
@blob_alias = dllexport alias i32 (), ptr @blob

@exportedButNotDefinedVariable = external dllexport global i32
declare dllexport void @exportedButNotDefinedFunction()
define void @foo() {
entry:
  store i32 4, ptr @exportedButNotDefinedVariable, align 4
  call void @exportedButNotDefinedFunction()
  ret void
}
