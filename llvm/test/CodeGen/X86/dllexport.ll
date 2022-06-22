; RUN: llc -mtriple i386-pc-win32 < %s \
; RUN:    | FileCheck -check-prefix CHECK -check-prefix CHECK-CL %s
; RUN: llc -mtriple i386-pc-mingw32 < %s \
; RUN:    | FileCheck -check-prefix CHECK -check-prefix CHECK-GCC %s
; RUN: llc -mtriple i686-pc-cygwin %s -o - \
; RUN:    | FileCheck -check-prefix CHECK -check-prefix CHECK-GCC %s
; RUN: llc -mtriple i386-pc-win32 < %s \
; RUN:    | FileCheck -check-prefix NOTEXPORTED %s
; RUN: llc -mtriple i386-pc-mingw32 < %s \
; RUN:    | FileCheck -check-prefix NOTEXPORTED %s
; RUN: llc -mtriple i686-pc-cygwin %s -o - \
; RUN:    | FileCheck -check-prefix NOTEXPORTED %s

; CHECK: .text

define void @notExported() {
	ret void
}

; CHECK: .globl _f1
define dllexport void @f1() {
	ret void
}

; CHECK: .globl _f2
define dllexport void @f2() unnamed_addr {
	ret void
}

declare dllexport void @notDefined()

; CHECK: .globl _stdfun@0
define dllexport x86_stdcallcc void @stdfun() nounwind {
	ret void
}

; CHECK: .globl @fastfun@0
define dllexport x86_fastcallcc i32 @fastfun() nounwind {
	ret i32 0
}

; CHECK: .globl _thisfun
define dllexport x86_thiscallcc void @thisfun() nounwind {
	ret void
}

; CHECK: .globl _lnk1
$lnk1 = comdat any

define linkonce_odr dllexport void @lnk1() comdat {
	ret void
}

; CHECK: .globl _lnk2
$lnk2 = comdat any

define linkonce_odr dllexport void @lnk2() alwaysinline comdat {
	ret void
}

; CHECK: .weak _weak1
define weak_odr dllexport void @weak1() {
	ret void
}


; CHECK: .data
; CHECK: .globl _Var1
@Var1 = dllexport global i32 1, align 4

; CHECK: .rdata,"dr"
; CHECK: .globl _Var2
@Var2 = dllexport unnamed_addr constant i32 1

; CHECK: .comm _Var3
@Var3 = common dllexport global i32 0, align 4

; CHECK: .weak _WeakVar1
@WeakVar1 = weak_odr dllexport global i32 1, align 4

; CHECK: .weak _WeakVar2
@WeakVar2 = weak_odr dllexport unnamed_addr constant i32 1

; CHECK: .globl "_complex-name"
@"complex-name" = dllexport global i32 1, align 4

; CHECK: .globl _complex.name
@"complex.name" = dllexport global i32 1, align 4


; Verify items that should not be exported do not appear in the export table.
; We use a separate check prefix to avoid confusion between -NOT and -SAME.
; NOTEXPORTED: .section .drectve
; NOTEXPORTED-NOT: :notExported
; NOTEXPORTED-NOT: :notDefined

; CHECK: .section .drectve
; CHECK-CL: .ascii " /EXPORT:_f1"
; CHECK-CL: .ascii " /EXPORT:_f2"
; CHECK-CL: .ascii " /EXPORT:_stdfun@0"
; CHECK-CL: .ascii " /EXPORT:@fastfun@0"
; CHECK-CL: .ascii " /EXPORT:_thisfun"
; CHECK-CL: .ascii " /EXPORT:_lnk1"
; CHECK-CL: .ascii " /EXPORT:_lnk2"
; CHECK-CL: .ascii " /EXPORT:_weak1"
; CHECK-CL: .ascii " /EXPORT:_Var1,DATA"
; CHECK-CL: .ascii " /EXPORT:_Var2,DATA"
; CHECK-CL: .ascii " /EXPORT:_Var3,DATA"
; CHECK-CL: .ascii " /EXPORT:_WeakVar1,DATA"
; CHECK-CL: .ascii " /EXPORT:_WeakVar2,DATA"
; CHECK-CL: .ascii " /EXPORT:\"_complex-name\",DATA"
; CHECK-CL: .ascii " /EXPORT:\"_complex.name\",DATA"
; CHECK-CL: .ascii " /EXPORT:_alias"
; CHECK-CL: .ascii " /EXPORT:_alias2"
; CHECK-CL: .ascii " /EXPORT:_alias3"
; CHECK-CL: .ascii " /EXPORT:_weak_alias"
; CHECK-GCC: .ascii " -export:f1"
; CHECK-GCC: .ascii " -export:f2"
; CHECK-GCC: .ascii " -export:stdfun@0"
; CHECK-GCC: .ascii " -export:@fastfun@0"
; CHECK-GCC: .ascii " -export:thisfun"
; CHECK-GCC: .ascii " -export:lnk1"
; CHECK-GCC: .ascii " -export:lnk2"
; CHECK-GCC: .ascii " -export:weak1"
; CHECK-GCC: .ascii " -export:Var1,data"
; CHECK-GCC: .ascii " -export:Var2,data"
; CHECK-GCC: .ascii " -export:Var3,data"
; CHECK-GCC: .ascii " -export:WeakVar1,data"
; CHECK-GCC: .ascii " -export:WeakVar2,data"
; CHECK-GCC: .ascii " -export:\"complex-name\",data"
; CHECK-GCC: .ascii " -export:\"complex.name\",data"
; CHECK-GCC: .ascii " -export:alias"
; CHECK-GCC: .ascii " -export:alias2"
; CHECK-GCC: .ascii " -export:alias3"
; CHECK-GCC: .ascii " -export:weak_alias"

; CHECK: .globl _alias
; CHECK: .set _alias, _notExported
@alias = dllexport alias void(), ptr @notExported

; CHECK: .globl _alias2
; CHECK: .set _alias2, _f1
@alias2 = dllexport alias void(), ptr @f1

; CHECK: .globl _alias3
; CHECK: .set _alias3, _notExported
@alias3 = dllexport alias void(), ptr @notExported

; CHECK: .weak _weak_alias
; CHECK: .set _weak_alias, _f1
@weak_alias = weak_odr dllexport alias void(), ptr @f1
