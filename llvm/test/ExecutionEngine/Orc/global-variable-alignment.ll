; Check that lli respects alignment on global variables.
;
; Returns ((uint32_t)&B & 0x7) - A + C. Variables A and C have byte-alignment,
; and are intended to increase the chance of misalignment, but don't contribute
; to the result, since they have the same initial value.
;
; A failure may indicate a problem with alignment handling in the JIT linker or
; JIT memory manager.
;
; RUN: %lli %s

@A = internal global i8 1, align 1
@B = global i64 1, align 8
@C = internal global i8 1, align 1

define i32 @main(i32 %argc, i8** %argv) {
entry:
  %0 = ptrtoint i8* @B to i32
  %1 = and i32 %0, 7
  %2 = load i8, i8* @A
  %3 = zext i8 %2 to i32
  %4 = add i32 %1, %3
  %5 = load i8, i8* @B
  %6 = zext i8 %5 to i32
  %7 = sub i32 %4, %6
  ret i32 %7
}
