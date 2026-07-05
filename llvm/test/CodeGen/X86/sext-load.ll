; RUN: llc < %s -mtriple=i686-- | FileCheck %s

; When doing sign extension, use the sext-load lowering to take advantage of
; x86's sign extension during loads.
;
; CHECK-LABEL: test1:
; CHECK:      movsbl {{.*}}, %eax
; CHECK-NEXT: ret
define i32 @test1(i32 %X) nounwind  {
entry:
	%tmp12 = trunc i32 %X to i8		; <i8> [#uses=1]
	%tmp123 = sext i8 %tmp12 to i32		; <i32> [#uses=1]
	ret i32 %tmp123
}

; When using a sextload representation, ensure that the sign extension is
; preserved even when removing shifted-out low bits.
;
; CHECK-LABEL: test2:
; CHECK:      movswl {{.*}}, %eax
; CHECK-NEXT: ret
define i32 @test2(ptr %this) {
entry:
  %b48 = getelementptr inbounds { i16, [6 x i8] }, ptr %this, i32 0, i32 1
  %bf.load = load i48, ptr %b48, align 2
  %bf.ashr = ashr i48 %bf.load, 32
  %bf.cast = trunc i48 %bf.ashr to i32
  ret i32 %bf.cast
}
