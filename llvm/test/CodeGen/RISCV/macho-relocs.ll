; RUN: llc -mtriple=riscv32-apple-macho %s -o - -asm-verbose=false                          | FileCheck %s
; RUN: llc -mtriple=riscv32-apple-macho %s -o - -asm-verbose=false -relocation-model=static | FileCheck %s --check-prefix=CHECK-STATIC

; CHECK-LABEL: _simple:
; CHECK-NEXT: ret
define void @simple() nounwind {
  ret void
}

; CHECK-LABEL: _call:
; CHECK-NEXT: addi sp, sp, -16
; CHECK-NEXT: sw ra, 12(sp)
; CHECK-NEXT: call _tail_call
; CHECK-NEXT: lw ra, 12(sp)
; CHECK-NEXT: addi sp, sp, 16
; CHECK-NEXT: ret
define void @call() nounwind {
  call void @tail_call()
  ret void
}

; CHECK-LABEL: _tail_call:
; CHECK-NEXT: tail _call
define void @tail_call() nounwind {
  tail call void @call()
  ret void
}

; CHECK-LABEL: _direct_global:
; CHECK-NEXT: [[AUI_BB:Lpcrel_hi.*]]:
; CHECK-NEXT:     auipc a0, %pcrel_hi(_var)
; CHECK-NEXT:     addi a0, a0, %pcrel_lo([[AUI_BB]])
; CHECK-NEXT:     ret
; CHECK-STATIC-LABEL: _direct_global:
; CHECK-STATIC-NEXT: lui a0, %hi(_var)
; CHECK-STATIC-NEXT: addi a0, a0, %lo(_var)
; CHECK-STATIC-NEXT: ret
@var = global i32 0
define i32* @direct_global() nounwind {
  ret i32* @var
}

; CHECK-LABEL: _got_global:
; CHECK-NEXT: [[AUI_BB:Lpcrel_hi.*]]:
; CHECK-NEXT:     auipc a0, %got_pcrel_hi(_var2)
; CHECK-NEXT:     lw a0, %pcrel_lo([[AUI_BB]])(a0)
; CHECK-NEXT:     ret
; No GOTs in static CodeGen.
; CHECK-STATIC-LABEL: _got_global:
; CHECK-STATIC-NEXT: lui a0, %hi(_var2)
; CHECK-STATIC-NEXT: addi a0, a0, %lo(_var2)
; CHECK-STATIC-NEXT: ret
@var2 = external global i32
define i32* @got_global() nounwind {
  ret i32* @var2
}

; CHECK-LABEL: unnamed_const:
; CHECK-NEXT: [[AUI_BB:Lpcrel_hi.*]]:
; CHECK-NEXT: auipc a0, %pcrel_hi(l_anon)
; CHECK-NEXT: addi a0, a0, %pcrel_lo([[AUI_BB]])
; CHECK-STATIC-LABEL: unnamed_const:
; CHECK-STATIC-NEXT: lui a0, %hi(l_anon)
; CHECK-STATIC-NEXT: addi a0, a0, %lo(l_anon)
@anon = private unnamed_addr constant i32 42
define i32* @unnamed_const() nounwind {
  ret i32* @anon
}
; CHECK-LABEL: l_anon:
; CHECK-NEXT:     .word 42

; CHECK-LABEL: .section	__DATA,__data
; CHECK-LABEL: _addend:
; CHECK-NEXT:     .word _simple+42
@addend = global i32 add(i32 ptrtoint(void()* @simple to i32), i32 42)

; CHECK-LABEL: _sub:
; CHECK-NEXT:     .word _simple-_call
@sub = global i32 sub(i32 ptrtoint(void()* @simple to i32), i32 ptrtoint(void()* @call to i32))
