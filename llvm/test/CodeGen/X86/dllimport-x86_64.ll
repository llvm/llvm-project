; RUN: llc -mtriple x86_64-pc-win32 < %s | FileCheck %s
; RUN: llc -mtriple x86_64-pc-mingw32 < %s | FileCheck %s
;
; RUN: llc -mtriple x86_64-pc-mingw32 -O0 < %s | FileCheck %s
; RUN: llc -mtriple x86_64-pc-windows-msvc -O0 < %s | FileCheck %s
; PR6275
;
; RUN: opt -mtriple x86_64-pc-win32 -O3 -S < %s | FileCheck %s -check-prefix=OPT

@Var1 = external dllimport global i32
@Var2 = available_externally dllimport unnamed_addr constant i32 1

declare dllimport void @fun()

define available_externally dllimport void @inline1() {
	ret void
}

define available_externally dllimport void @inline2() {
	ret void
}

declare void @dummy(...)

define void @use() nounwind {
; CHECK:     callq *__imp_fun(%rip)
  call void @fun()

; CHECK: callq *__imp_inline1(%rip)
; CHECK: callq *__imp_inline2(%rip)
  call void @inline1()
  call void @inline2()

; available_externally uses go away
; OPT-NOT: call void @inline1()
; OPT-NOT: call void @inline2()
; OPT-NOT: load i32, ptr @Var2
; OPT: call void (...) @dummy(i32 %1, i32 1)

; CHECK-DAG: movq __imp_Var1(%rip), [[R1:%[a-z]{3}]]
; CHECK-DAG: movq __imp_Var2(%rip), [[R2:%[a-z]{3}]]
  %1 = load i32, ptr @Var1
  %2 = load i32, ptr @Var2
  call void(...) @dummy(i32 %1, i32 %2)

  ret void
}
