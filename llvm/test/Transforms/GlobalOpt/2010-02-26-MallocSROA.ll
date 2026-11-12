; RUN: opt -passes=globalopt -S < %s
; PR6435
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

%struct.xyz = type { double, i32 }

@Y = internal global ptr null            ; <ptr> [#uses=2]
@numf2s = external global i32                     ; <ptr> [#uses=1]

define fastcc void @init_net() nounwind {
entry:
  %0 = load i32, ptr @numf2s, align 4                 ; <i32> [#uses=1]
  %mallocsize2 = shl i32 %0, 4                    ; <i32> [#uses=1]
  %malloccall3 = tail call ptr @malloc(i32 %mallocsize2) nounwind ; <ptr> [#uses=1]
  store ptr %malloccall3, ptr @Y, align 8
  ret void
}

define fastcc void @load_train(ptr %trainfile, i32 %mode, i32 %objects) nounwind {
entry:
  %0 = load ptr, ptr @Y, align 8             ; <ptr> [#uses=0]
  ret void
}

declare noalias ptr @malloc(i32)
