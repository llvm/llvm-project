; RUN: llc -mtriple i386-pc-win32 < %s | FileCheck %s
; RUN: llc -mtriple i386-pc-mingw32 < %s | FileCheck %s
;
; RUN: llc -mtriple i386-pc-mingw32 -O0 < %s | FileCheck %s
; RUN: llc -mtriple i386-pc-windows-msvc -O0 < %s | FileCheck %s
; PR6275
;
; RUN: opt -mtriple i386-pc-win32 -O3 -S < %s | FileCheck %s -check-prefix=OPT

@Var1 = external dllimport global i32
@Var2 = available_externally dllimport unnamed_addr constant i32 1

declare dllimport void @fun()

define available_externally dllimport void @inline1() {
	ret void
}

define available_externally dllimport void @inline2() alwaysinline {
	ret void
}

declare dllimport x86_stdcallcc void @stdfun() nounwind
declare dllimport x86_fastcallcc void @fastfun() nounwind
declare dllimport x86_thiscallcc void @thisfun() nounwind

declare void @dummy(...)

define void @use() nounwind {
; CHECK:     calll *__imp__fun
  call void @fun()

; CHECK: calll *__imp__inline1
; CHECK: calll *__imp__inline2
  call void @inline1()
  call void @inline2()

; CHECK: calll *__imp__stdfun@0
; CHECK: calll *__imp_@fastfun@0
; CHECK: calll *__imp__thisfun
  call void @stdfun()
  call void @fastfun()
  call void @thisfun()

; available_externally uses go away
; OPT-NOT: call void @inline1()
; OPT-NOT: call void @inline2()
; OPT-NOT: load i32, ptr @Var2
; OPT: call void (...) @dummy(i32 %1, i32 1)

; CHECK-DAG: movl __imp__Var1, [[R1:%[a-z]{3}]]
; CHECK-DAG: movl __imp__Var2, [[R2:%[a-z]{3}]]
  %1 = load i32, ptr @Var1
  %2 = load i32, ptr @Var2
  call void(...) @dummy(i32 %1, i32 %2)

  ret void
}

; CHECK: _fp:
; CHECK-NEXT: .long _fun
@fp = constant ptr @fun
