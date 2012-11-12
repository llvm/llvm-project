; RUN: opt < %s -indvars

; An invoke has a result value which is used in an "Interesting"
; expression inside the loop. IndVars should be able to rewrite
; the expression in the correct place.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
  %struct.string___XUB = type { i32, i32 }
  %struct.string___XUP = type { [0 x i8]*, %struct.string___XUB* }
@.str7 = external constant [24 x i8]            ; <[24 x i8]*> [#uses=1]
@C.17.316 = external constant %struct.string___XUB              ; <%struct.string___XUB*> [#uses=1]

define void @_ada_c35503g() {
entry:
  br label %bb

bb:             ; preds = %bb, %entry
  br i1 false, label %bb65.loopexit, label %bb

bb65.loopexit:          ; preds = %bb
  br label %bb123

bb123:          ; preds = %bb178, %bb65.loopexit
  %i.0 = phi i32 [ %3, %bb178 ], [ 0, %bb65.loopexit ]          ; <i32> [#uses=3]
  %0 = invoke i32 @report__ident_int(i32 1)
      to label %invcont127 unwind label %lpad266                ; <i32> [#uses=1]

invcont127:             ; preds = %bb123
  %1 = sub i32 %i.0, %0         ; <i32> [#uses=1]
  %2 = icmp eq i32 0, %1                ; <i1> [#uses=1]
  br i1 %2, label %bb178, label %bb128

bb128:          ; preds = %invcont127
  invoke void @system__img_int__image_integer(%struct.string___XUP* noalias sret null, i32 %i.0)
      to label %invcont129 unwind label %lpad266

invcont129:             ; preds = %bb128
  invoke void @system__string_ops__str_concat(%struct.string___XUP* noalias sret null, [0 x i8]* bitcast ([24 x i8]* @.str7 to [0 x i8]*), %struct.string___XUB* @C.17.316, [0 x i8]* null, %struct.string___XUB* null)
      to label %invcont138 unwind label %lpad266

invcont138:             ; preds = %invcont129
  unreachable

bb178:          ; preds = %invcont127
  %3 = add i32 %i.0, 1          ; <i32> [#uses=1]
  br label %bb123

lpad266:                ; preds = %invcont129, %bb128, %bb123
  %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
            cleanup
  unreachable
}

declare i32 @__gxx_personality_v0(...)

declare void @system__img_int__image_integer(%struct.string___XUP* noalias sret, i32)

declare void @system__string_ops__str_concat(%struct.string___XUP* noalias sret, [0 x i8]*, %struct.string___XUB*, [0 x i8]*, %struct.string___XUB*)

declare i32 @report__ident_int(i32)
