; RUN: %lli -jit-kind=mcjit %s
; RUN: %lli %s
;
; Verify relocations to global symbols with addend work correctly.
;
; Compiled from this C code:
;
; int test[2] = { -1, 0 };
; int *p = &test[1];
; 
; int main (void)
; {
;   return *p;
; }
; 

@test = global [2 x i32] [i32 -1, i32 0], align 4
@p = global ptr getelementptr inbounds ([2 x i32], ptr @test, i64 0, i64 1), align 8

define i32 @main() {
entry:
  %0 = load ptr, ptr @p, align 8
  %1 = load i32, ptr %0, align 4
  ret i32 %1
}

