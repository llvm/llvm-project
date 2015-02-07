; RUN: llc -mtriple x86_64-pc-win32 < %s | FileCheck -check-prefix=CHECK -check-prefix=WIN32 %s
; RUN: llc -mtriple x86_64-pc-mingw32 < %s | FileCheck -check-prefix=CHECK -check-prefix=MINGW %s

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

; CHECK: .section .text,"xr",discard,lnk1
; CHECK: .globl lnk1
define linkonce_odr dllexport void @lnk1() {
	ret void
}

; CHECK: .section .text,"xr",discard,lnk2
; CHECK: .globl lnk2
define linkonce_odr dllexport void @lnk2() alwaysinline {
	ret void
}

; CHECK: .section .text,"xr",discard,weak1
; CHECK: .globl weak1
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

; CHECK: .section .data,"dw",discard,WeakVar1
; CHECK: .globl WeakVar1
@WeakVar1 = weak_odr dllexport global i32 1, align 4

; CHECK: .section .rdata,"dr",discard,WeakVar2
; CHECK: .globl WeakVar2
@WeakVar2 = weak_odr dllexport unnamed_addr constant i32 1


; CHECK: .globl alias
; CHECK: alias = notExported
@alias = dllexport alias void()* @notExported

; CHECK: .globl alias2
; CHECK: alias2 = f1
@alias2 = dllexport alias void()* @f1

; CHECK: .globl alias3
; CHECK: alias3 = notExported
@alias3 = dllexport alias void()* @notExported

; CHECK: .weak weak_alias
; CHECK: weak_alias = f1
@weak_alias = weak_odr dllexport alias void()* @f1

@blob = global [6 x i8] c"\B8*\00\00\00\C3", section ".text", align 16
@blob_alias = dllexport alias bitcast ([6 x i8]* @blob to i32 ()*)

; CHECK: .section .drectve
; WIN32: " /EXPORT:Var1,DATA"
; WIN32: " /EXPORT:Var2,DATA"
; WIN32: " /EXPORT:Var3,DATA"
; WIN32: " /EXPORT:WeakVar1,DATA"
; WIN32: " /EXPORT:WeakVar2,DATA"
; WIN32: " /EXPORT:f1"
; WIN32: " /EXPORT:f2"
; WIN32: " /EXPORT:lnk1"
; WIN32: " /EXPORT:lnk2"
; WIN32: " /EXPORT:weak1"
; WIN32: " /EXPORT:alias"
; WIN32: " /EXPORT:alias2"
; WIN32: " /EXPORT:alias3"
; WIN32: " /EXPORT:weak_alias"
; WIN32: " /EXPORT:blob_alias"
; MINGW: " -export:Var1,data"
; MINGW: " -export:Var2,data"
; MINGW: " -export:Var3,data"
; MINGW: " -export:WeakVar1,data"
; MINGW: " -export:WeakVar2,data"
; MINGW: " -export:f1"
; MINGW: " -export:f2"
; MINGW: " -export:lnk1"
; MINGW: " -export:lnk2"
; MINGW: " -export:weak1"
; MINGW: " -export:alias"
; MINGW: " -export:alias2"
; MINGW: " -export:alias3"
; MINGW: " -export:weak_alias"
; MINGW: " -export:blob_alias"
