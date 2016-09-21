; RUN: llc -march=mips -mcpu=mips32 -O0 -relocation-model=pic < %s | FileCheck \
; RUN:     %s -check-prefix=MIPS32
; RUN: llc -march=mips64 -mcpu=mips64 -O0 -relocation-model=pic -target-abi n64 \
; RUN:     < %s | FileCheck %s -check-prefix=MIPS64
; RUN: llc -march=mips64 -mcpu=mips64 -O0 -relocation-model=pic -target-abi n32 \
; RUN:     < %s | FileCheck %s -check-prefix=MIPS64


; LLVM PR/30197
; Test that the scheduler does not order loads and stores of arguments that
; are passed on the stack such that the arguments of the caller are clobbered
; too early.

; This test is more fragile than I'd like. The -NEXT directives enforce an
; assumption that any GOT related instructions will not appear between the
; loads and stores.

; O32 case: The last two arguments should appear at 16(sp), 20(sp). The order
;           of the loads doesn't matter, but they have to become before the
;           stores
declare i32 @func2(i32, i32, i32, i32, i32, i32)

define i32 @func1(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f){

; MIPS32: lw ${{[0-9]+}}, {{[0-9]+}}($sp)
; MIPS32-NEXT: lw ${{[0-9]+}}, {{[0-9]+}}($sp)
; MIPS32-NEXT: sw ${{[0-9]+}}, {{[0-9]+}}($sp)
; MIPS32-NEXT: sw ${{[0-9]+}}, {{[0-9]+}}($sp)
  %retval = tail call i32 @func1(i32 %a, i32 %f, i32 %c, i32 %d, i32 %e, i32 %b)

  ret i32 %retval
}

; N64, N32 cases: N64 and N32 both pass 8 arguments in registers. The order
;           of the loads doesn't matter, but they have to become before the
;           stores

declare i64 @func4(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)

define i64 @func3(i64 %a, i64 %b, i64 %c, i64 %d,
                  i64 %e, i64 %f, i64 %g, i64 %h,
                  i64 %i, i64 %j){

; MIPS64: ld ${{[0-9]+}}, {{[0-9]+}}($sp)
; MIPS64-NEXT: ld ${{[0-9]+}}, {{[0-9]+}}($sp)
; MIPS64-NEXT: sd ${{[0-9]+}}, {{[0-9]+}}($sp)
; MIPS64-NEXT: sd ${{[0-9]+}}, {{[0-9]+}}($sp)
  %retval = tail call i64 @func4(i64 %a, i64 %j, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h, i64 %i, i64 %b)

  ret i64 %retval
}


