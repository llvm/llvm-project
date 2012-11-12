; A store or load cannot alias a global if the accessed amount is larger then
; the global.

; RUN: opt < %s -basicaa -gvn -S | FileCheck %s
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

@B = global i16 8

; CHECK: @test1
define i16 @test1(i32* %P) {
        %X = load i16* @B
        store i32 7, i32* %P
        %Y = load i16* @B
        %Z = sub i16 %Y, %X
        ret i16 %Z
; CHECK: ret i16 0
}

; Cannot know anything about the size of this global.
; rdar://8813415
@window = external global [0 x i8]

; CHECK: @test2
define i8 @test2(i32 %tmp79, i32 %w.2, i32 %indvar89) nounwind {
  %tmp92 = add i32 %tmp79, %indvar89
  %arrayidx412 = getelementptr [0 x i8]* @window, i32 0, i32 %tmp92
  %tmp93 = add i32 %w.2, %indvar89
  %arrayidx416 = getelementptr [0 x i8]* @window, i32 0, i32 %tmp93

  %A = load i8* %arrayidx412, align 1
  store i8 4, i8* %arrayidx416, align 1

  %B = load i8* %arrayidx412, align 1
  %C = sub i8 %A, %B
  ret i8 %C

; CHECK: %B = load i8
; CHECK: ret i8 %C
}

