; RUN: llc < %s -mtriple=x86_64-pc-win32 -O0 -mattr=+avx | FileCheck %s

; Background:
; If fast-isel bails out to normal selection, then the DAG combiner will run,
; even at -O0. In principle this should not happen (those are optimizations,
; and we said -O0) but as a practical matter there are some instruction
; selection patterns that depend on the legalizations and transforms that the
; DAG combiner does.
;
; The 'optnone' attribute implicitly sets -O0 and fast-isel for the function.
; The DAG combiner was disabled for 'optnone' (but not -O0) by r221168, then
; re-enabled in r233153 because of problems with instruction selection patterns
; mentioned above. (Note: because 'optnone' is supposed to match -O0, r221168
; really should have disabled the combiner for both.)
;
; If instruction selection eventually becomes smart enough to run without DAG
; combiner, then the combiner can be turned off for -O0 (not just 'optnone')
; and this test can go away. (To be replaced by a different test that verifies
; the DAG combiner does *not* run at -O0 or for 'optnone' functions.)
;
; In the meantime, this test wants to make sure the combiner stays enabled for
; 'optnone' functions, just as it is for -O0.


; The test cases @foo[WithOptnone] prove that the same DAG combine happens
; with -O0 and with 'optnone' set.  To prove this, we use a varags to cause
; fast-isel to bail out (varags aren't handled in fast isel).  Then we have
; a repeated fadd that can be combined into an fmul.  We show that this
; happens in both the non-optnone function and the optnone function.

define float @foo(float %x, ...) #0 {
entry:
  %add = fadd fast float %x, %x
  %add1 = fadd fast float %add, %x
  ret float %add1
}

; CHECK-LABEL: @foo
; CHECK-NOT:   add
; CHECK:       mul
; CHECK-NEXT:  ret

define float @fooWithOptnone(float %x, ...) #1 {
entry:
  %add = fadd fast float %x, %x
  %add1 = fadd fast float %add, %x
  ret float %add1
}

; CHECK-LABEL: @fooWithOptnone
; CHECK-NOT:   add
; CHECK:       mul
; CHECK-NEXT:  ret


; The test case @bar is derived from an instruction selection failure case
; that was solved by r233153. It depends on -mattr=+avx.
; Really all we're trying to prove is that it doesn't crash any more.

@id84 = common global <16 x i32> zeroinitializer, align 64

define void @bar(...) #1 {
entry:
  %id83 = alloca <16 x i8>, align 16
  %0 = load <16 x i32>, <16 x i32>* @id84, align 64
  %conv = trunc <16 x i32> %0 to <16 x i8>
  store <16 x i8> %conv, <16 x i8>* %id83, align 16
  ret void
}

attributes #0 = { "unsafe-fp-math"="true" }
attributes #1 = { noinline optnone "unsafe-fp-math"="true" }
