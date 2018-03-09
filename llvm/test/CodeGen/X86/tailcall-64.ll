; RUN: llc -mtriple=x86_64-apple-macosx -mcpu=core2 < %s | FileCheck %s

declare i64 @testi()

define i64 @test_trivial() {
 %A = tail call i64 @testi()
 ret i64 %A
}
; CHECK-LABEL: test_trivial:
; CHECK: jmp	_testi                  ## TAILCALL


define i64 @test_noop_bitcast() {
 %A = tail call i64 @testi()
 %B = bitcast i64 %A to i64
 ret i64 %B
}
; CHECK-LABEL: test_noop_bitcast:
; CHECK: jmp	_testi                  ## TAILCALL


; Tail call shouldn't be blocked by no-op inttoptr.
define i8* @test_inttoptr() {
  %A = tail call i64 @testi()
  %B = inttoptr i64 %A to i8*
  ret i8* %B
}

; CHECK-LABEL: test_inttoptr:
; CHECK: jmp	_testi                  ## TAILCALL


declare <4 x float> @testv()

define <4 x i32> @test_vectorbitcast() {
  %A = tail call <4 x float> @testv()
  %B = bitcast <4 x float> %A to <4 x i32>
  ret <4 x i32> %B
}
; CHECK-LABEL: test_vectorbitcast:
; CHECK: jmp	_testv                  ## TAILCALL


declare { i64, i64 } @testp()

define {i64, i64} @test_pair_trivial() {
  %A = tail call { i64, i64} @testp()
  ret { i64, i64} %A
}
; CHECK-LABEL: test_pair_trivial:
; CHECK: jmp	_testp                  ## TAILCALL

define {i64, i64} @test_pair_notail() {
  %A = tail call i64 @testi()

  %b = insertvalue {i64, i64} undef, i64 %A, 0
  %c = insertvalue {i64, i64} %b, i64 %A, 1

  ret { i64, i64} %c
}
; CHECK-LABEL: test_pair_notail:
; CHECK-NOT: jmp	_testi

define {i64, i64} @test_pair_extract_trivial() {
  %A = tail call { i64, i64} @testp()
  %x = extractvalue { i64, i64} %A, 0
  %y = extractvalue { i64, i64} %A, 1
  
  %b = insertvalue {i64, i64} undef, i64 %x, 0
  %c = insertvalue {i64, i64} %b, i64 %y, 1
  
  ret { i64, i64} %c
}

; CHECK-LABEL: test_pair_extract_trivial:
; CHECK: jmp	_testp                  ## TAILCALL

define {i64, i64} @test_pair_extract_notail() {
  %A = tail call { i64, i64} @testp()
  %x = extractvalue { i64, i64} %A, 0
  %y = extractvalue { i64, i64} %A, 1
  
  %b = insertvalue {i64, i64} undef, i64 %y, 0
  %c = insertvalue {i64, i64} %b, i64 %x, 1
  
  ret { i64, i64} %c
}

; CHECK-LABEL: test_pair_extract_notail:
; CHECK-NOT: jmp	_testp

define {i8*, i64} @test_pair_extract_conv() {
  %A = tail call { i64, i64} @testp()
  %x = extractvalue { i64, i64} %A, 0
  %y = extractvalue { i64, i64} %A, 1
  
  %x1 = inttoptr i64 %x to i8*
  
  %b = insertvalue {i8*, i64} undef, i8* %x1, 0
  %c = insertvalue {i8*, i64} %b, i64 %y, 1
  
  ret { i8*, i64} %c
}

; CHECK-LABEL: test_pair_extract_conv:
; CHECK: jmp	_testp                  ## TAILCALL

define {i64, i64} @test_pair_extract_multiple() {
  %A = tail call { i64, i64} @testp()
  %x = extractvalue { i64, i64} %A, 0
  %y = extractvalue { i64, i64} %A, 1
  
  %b = insertvalue {i64, i64} undef, i64 %x, 0
  %c = insertvalue {i64, i64} %b, i64 %y, 1

  %x1 = extractvalue { i64, i64} %b, 0
  %y1 = extractvalue { i64, i64} %c, 1

  %d = insertvalue {i64, i64} undef, i64 %x1, 0
  %e = insertvalue {i64, i64} %b, i64 %y1, 1
  
  ret { i64, i64} %e
}

; CHECK-LABEL: test_pair_extract_multiple:
; CHECK: jmp	_testp                  ## TAILCALL

define {i64, i64} @test_pair_extract_undef() {
  %A = tail call { i64, i64} @testp()
  %x = extractvalue { i64, i64} %A, 0
  
  %b = insertvalue {i64, i64} undef, i64 %x, 0
  
  ret { i64, i64} %b
}

; CHECK-LABEL: test_pair_extract_undef:
; CHECK: jmp	_testp                  ## TAILCALL

declare { i64, { i32, i32 } } @testn()

define {i64, {i32, i32}} @test_nest() {
  %A = tail call { i64, { i32, i32 } } @testn()
  %x = extractvalue { i64, { i32, i32}} %A, 0
  %y = extractvalue { i64, { i32, i32}} %A, 1
  %y1 = extractvalue { i32, i32} %y, 0
  %y2 = extractvalue { i32, i32} %y, 1
  
  %b = insertvalue {i64, {i32, i32}} undef, i64 %x, 0
  %c1 = insertvalue {i32, i32} undef, i32 %y1, 0
  %c2 = insertvalue {i32, i32} %c1, i32 %y2, 1
  %c = insertvalue {i64, {i32, i32}} %b, {i32, i32} %c2, 1
 
  ret { i64, { i32, i32}} %c
}

; CHECK-LABEL: test_nest:
; CHECK: jmp	_testn                  ## TAILCALL

%struct.A = type { i32 }
%struct.B = type { %struct.A, i32 }

declare %struct.B* @testu()

define %struct.A* @test_upcast() {
entry:
  %A = tail call %struct.B* @testu()
  %x = getelementptr inbounds %struct.B, %struct.B* %A, i32 0, i32 0
  ret %struct.A* %x
}

; CHECK-LABEL: test_upcast:
; CHECK: jmp	_testu                  ## TAILCALL

; PR13006
define { i64, i64 } @crash(i8* %this) {
  %c = tail call { i64, i64 } @testp()
  %mrv7 = insertvalue { i64, i64 } %c, i64 undef, 1
  ret { i64, i64 } %mrv7
}

; Check that we can fold an indexed load into a tail call instruction.
; CHECK: fold_indexed_load
; CHECK: leaq (%rsi,%rsi,4), %[[rax:r..]]
; CHECK: jmpq *16(%{{r..}},%[[rax]],8)  ## TAILCALL
%struct.funcs = type { i32 (i8*, i32*, i32)*, i32 (i8*)*, i32 (i8*)*, i32 (i8*, i32)*, i32 }
@func_table = external global [0 x %struct.funcs]
define void @fold_indexed_load(i8* %mbstr, i64 %idxprom) nounwind uwtable ssp {
entry:
  %dsplen = getelementptr inbounds [0 x %struct.funcs], [0 x %struct.funcs]* @func_table, i64 0, i64 %idxprom, i32 2
  %x1 = load i32 (i8*)*, i32 (i8*)** %dsplen, align 8
  %call = tail call i32 %x1(i8* %mbstr) nounwind
  ret void
}

; <rdar://problem/12282281> Fold an indexed load into the tail call instruction.
; Calling a varargs function with 6 arguments requires 7 registers (%al is the
; vector count for varargs functions). This leaves %r11 as the only available
; scratch register.
;
; It is not possible to fold an indexed load into TCRETURNmi64 in that case.
;
; typedef int (*funcptr)(void*, ...);
; extern const funcptr funcs[];
; int f(int n) {
;   return funcs[n](0, 0, 0, 0, 0, 0);
; }
;
; CHECK-LABEL: rdar12282281
; CHECK: jmpq *%r11 ## TAILCALL
@funcs = external constant [0 x i32 (i8*, ...)*]

define i32 @rdar12282281(i32 %n) nounwind uwtable ssp {
entry:
  %idxprom = sext i32 %n to i64
  %arrayidx = getelementptr inbounds [0 x i32 (i8*, ...)*], [0 x i32 (i8*, ...)*]* @funcs, i64 0, i64 %idxprom
  %0 = load i32 (i8*, ...)*, i32 (i8*, ...)** %arrayidx, align 8
  %call = tail call i32 (i8*, ...) %0(i8* null, i32 0, i32 0, i32 0, i32 0, i32 0) nounwind
  ret i32 %call
}

define x86_fp80 @fp80_call(x86_fp80 %x) nounwind  {
entry:
; CHECK-LABEL: fp80_call:
; CHECK: jmp _fp80_callee
  %call = tail call x86_fp80 @fp80_callee(x86_fp80 %x) nounwind
  ret x86_fp80 %call
}

declare x86_fp80 @fp80_callee(x86_fp80)

; rdar://12229511
define x86_fp80 @trunc_fp80(x86_fp80 %x) nounwind  {
entry:
; CHECK-LABEL: trunc_fp80
; CHECK: callq _trunc
; CHECK-NOT: jmp _trunc
; CHECK: ret
  %conv = fptrunc x86_fp80 %x to double
  %call = tail call double @trunc(double %conv) nounwind readnone
  %conv1 = fpext double %call to x86_fp80
  ret x86_fp80 %conv1
}

declare double @trunc(double) nounwind readnone
