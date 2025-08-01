; RUN: llc -mtriple mipsel-windows < %s | FileCheck %s

@Var1 = external dllimport global i32
@Var2 = available_externally dllimport unnamed_addr constant i32 1

declare dllimport void @fun()

define available_externally dllimport void @inline1() {
	ret void
}

define available_externally dllimport void @inline2() alwaysinline {
	ret void
}

declare void @dummy(...)

define void @use() nounwind {
; CHECK:     lui $1, %hi(__imp_fun)
; CHECK:     addiu $1, $1, %lo(__imp_fun)
; CHECK:     lw $25, 0($1)
; CHECK:     jalr $25
  call void @fun()

; CHECK:     lui $1, %hi(__imp_inline1)
; CHECK:     addiu $1, $1, %lo(__imp_inline1)
; CHECK:     lw $25, 0($1)
; CHECK:     jalr $25
  call void @inline1()

; CHECK:     lui $1, %hi(__imp_inline2)
; CHECK:     addiu $1, $1, %lo(__imp_inline2)
; CHECK:     lw $25, 0($1)
; CHECK:     jalr $25
  call void @inline2()

; CHECK:     lui $1, %hi(__imp_Var2)
; CHECK:     addiu $1, $1, %lo(__imp_Var2)
; CHECK:     lw $1, 0($1)
; CHECK:     lw $5, 0($1)
; CHECK:     lui $1, %hi(__imp_Var1)
; CHECK:     addiu $1, $1, %lo(__imp_Var1)
; CHECK:     lw $1, 0($1)
; CHECK:     lw $4, 0($1)
  %1 = load i32, ptr @Var1
  %2 = load i32, ptr @Var2
  call void(...) @dummy(i32 %1, i32 %2)

  ret void
}

; CHECK: fp:
; CHECK-NEXT: .long fun
@fp = constant ptr @fun

