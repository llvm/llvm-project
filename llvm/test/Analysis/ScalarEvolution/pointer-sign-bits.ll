; RUN: opt < %s -passes='print<scalar-evolution>'

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
  %JavaObject = type { ptr, ptr }

define void @JnJVM_antlr_CSharpCodeGenerator_genBitSet__Lantlr_collections_impl_BitSet_2I(ptr, ptr, i32, i1 %arg) {
start:
  br i1 %arg, label %"stack overflow", label %"no stack overflow"

"GOTO or IF*2":         ; preds = %"true verifyAndComputePtr89", %verifyNullCont84
  unreachable

"GOTO or IF*5":         ; preds = %"true verifyAndComputePtr127", %"GOTO or IF*6"
  unreachable

"GOTO or IF*6":         ; preds = %"true verifyAndComputePtr131.GOTO or IF*6_crit_edge", %"true verifyAndComputePtr89"
  %indvar = phi i32 [ %indvar.next, %"true verifyAndComputePtr131.GOTO or IF*6_crit_edge" ], [ 0, %"true verifyAndComputePtr89" ]               ; <i32> [#uses=2]
  %.0.in = add i32 %indvar, 0           ; <i32> [#uses=1]
  %.0 = add i32 %.0.in, 1               ; <i32> [#uses=1]
  %3 = icmp slt i32 %.0, %4             ; <i1> [#uses=1]
  br i1 %3, label %verifyNullCont126, label %"GOTO or IF*5"

end:            ; preds = %"no exception block35"
  ret void

"stack overflow":               ; preds = %start
  ret void

"no stack overflow":            ; preds = %start
  br i1 %arg, label %verifyNullCont, label %"no stack overflow.end_crit_edge"

"no stack overflow.end_crit_edge":              ; preds = %"no stack overflow"
  ret void

verifyNullCont:         ; preds = %"no stack overflow"
  br i1 %arg, label %verifyNullCont9, label %verifyNullCont.end_crit_edge

verifyNullCont.end_crit_edge:           ; preds = %verifyNullCont
  ret void

verifyNullCont9:                ; preds = %verifyNullCont
  br i1 %arg, label %verifyNullCont12, label %verifyNullCont9.end_crit_edge

verifyNullCont9.end_crit_edge:          ; preds = %verifyNullCont9
  ret void

verifyNullCont12:               ; preds = %verifyNullCont9
  br i1 %arg, label %"no exception block13", label %verifyNullCont12.end_crit_edge

verifyNullCont12.end_crit_edge:         ; preds = %verifyNullCont12
  ret void

"no exception block13":         ; preds = %verifyNullCont12
  br i1 %arg, label %verifyNullExit14, label %verifyNullCont15

verifyNullExit14:               ; preds = %"no exception block13"
  ret void

verifyNullCont15:               ; preds = %"no exception block13"
  br i1 %arg, label %"no exception block16", label %verifyNullCont15.end_crit_edge

verifyNullCont15.end_crit_edge:         ; preds = %verifyNullCont15
  ret void

"no exception block16":         ; preds = %verifyNullCont15
  br i1 %arg, label %verifyNullExit17, label %verifyNullCont18

verifyNullExit17:               ; preds = %"no exception block16"
  ret void

verifyNullCont18:               ; preds = %"no exception block16"
  br i1 %arg, label %"no exception block19", label %verifyNullCont18.end_crit_edge

verifyNullCont18.end_crit_edge:         ; preds = %verifyNullCont18
  ret void

"no exception block19":         ; preds = %verifyNullCont18
  br i1 %arg, label %verifyNullExit20, label %verifyNullCont21

verifyNullExit20:               ; preds = %"no exception block19"
  ret void

verifyNullCont21:               ; preds = %"no exception block19"
  br i1 %arg, label %verifyNullCont24, label %verifyNullCont21.end_crit_edge

verifyNullCont21.end_crit_edge:         ; preds = %verifyNullCont21
  ret void

verifyNullCont24:               ; preds = %verifyNullCont21
  br i1 %arg, label %verifyNullCont27, label %verifyNullCont24.end_crit_edge

verifyNullCont24.end_crit_edge:         ; preds = %verifyNullCont24
  ret void

verifyNullCont27:               ; preds = %verifyNullCont24
  br i1 %arg, label %verifyNullCont32, label %verifyNullCont27.end_crit_edge

verifyNullCont27.end_crit_edge:         ; preds = %verifyNullCont27
  ret void

verifyNullCont32:               ; preds = %verifyNullCont27
  br i1 %arg, label %verifyNullExit33, label %verifyNullCont34

verifyNullExit33:               ; preds = %verifyNullCont32
  ret void

verifyNullCont34:               ; preds = %verifyNullCont32
  br i1 %arg, label %"no exception block35", label %verifyNullCont34.end_crit_edge

verifyNullCont34.end_crit_edge:         ; preds = %verifyNullCont34
  ret void

"no exception block35":         ; preds = %verifyNullCont34
  br i1 %arg, label %end, label %verifyNullCont60

verifyNullCont60:               ; preds = %"no exception block35"
  br i1 %arg, label %verifyNullCont63, label %verifyNullCont60.end_crit_edge

verifyNullCont60.end_crit_edge:         ; preds = %verifyNullCont60
  ret void

verifyNullCont63:               ; preds = %verifyNullCont60
  br i1 %arg, label %"no exception block64", label %verifyNullCont63.end_crit_edge

verifyNullCont63.end_crit_edge:         ; preds = %verifyNullCont63
  ret void

"no exception block64":         ; preds = %verifyNullCont63
  br i1 %arg, label %verifyNullExit65, label %verifyNullCont66

verifyNullExit65:               ; preds = %"no exception block64"
  ret void

verifyNullCont66:               ; preds = %"no exception block64"
  br i1 %arg, label %"no exception block67", label %verifyNullCont66.end_crit_edge

verifyNullCont66.end_crit_edge:         ; preds = %verifyNullCont66
  ret void

"no exception block67":         ; preds = %verifyNullCont66
  br i1 %arg, label %verifyNullExit68, label %verifyNullCont69

verifyNullExit68:               ; preds = %"no exception block67"
  ret void

verifyNullCont69:               ; preds = %"no exception block67"
  br i1 %arg, label %"no exception block70", label %verifyNullCont69.end_crit_edge

verifyNullCont69.end_crit_edge:         ; preds = %verifyNullCont69
  ret void

"no exception block70":         ; preds = %verifyNullCont69
  br i1 %arg, label %verifyNullExit71, label %verifyNullCont72

verifyNullExit71:               ; preds = %"no exception block70"
  ret void

verifyNullCont72:               ; preds = %"no exception block70"
  br i1 %arg, label %verifyNullCont75, label %verifyNullCont72.end_crit_edge

verifyNullCont72.end_crit_edge:         ; preds = %verifyNullCont72
  ret void

verifyNullCont75:               ; preds = %verifyNullCont72
  br i1 %arg, label %verifyNullCont78, label %verifyNullCont75.end_crit_edge

verifyNullCont75.end_crit_edge:         ; preds = %verifyNullCont75
  ret void

verifyNullCont78:               ; preds = %verifyNullCont75
  br i1 %arg, label %"verifyNullCont78.GOTO or IF*4_crit_edge", label %verifyNullCont78.end_crit_edge

"verifyNullCont78.GOTO or IF*4_crit_edge":              ; preds = %verifyNullCont78
  br i1 %arg, label %verifyNullExit80, label %verifyNullCont81

verifyNullCont78.end_crit_edge:         ; preds = %verifyNullCont78
  ret void

verifyNullExit80:               ; preds = %"verifyNullCont78.GOTO or IF*4_crit_edge"
  ret void

verifyNullCont81:               ; preds = %"verifyNullCont78.GOTO or IF*4_crit_edge"
  %4 = ptrtoint ptr undef to i32                ; <i32> [#uses=2]
  %5 = icmp slt i32 0, %4               ; <i1> [#uses=1]
  br i1 %5, label %verifyNullCont84, label %verifyNullCont172

verifyNullCont84:               ; preds = %verifyNullCont81
  br i1 %arg, label %"GOTO or IF*2", label %verifyNullCont86

verifyNullCont86:               ; preds = %verifyNullCont84
  br i1 %arg, label %"true verifyAndComputePtr", label %"false verifyAndComputePtr"

"true verifyAndComputePtr":             ; preds = %verifyNullCont86
  br i1 %arg, label %"true verifyAndComputePtr89", label %"false verifyAndComputePtr90"

"false verifyAndComputePtr":            ; preds = %verifyNullCont86
  ret void

"true verifyAndComputePtr89":           ; preds = %"true verifyAndComputePtr"
  br i1 %arg, label %"GOTO or IF*6", label %"GOTO or IF*2"

"false verifyAndComputePtr90":          ; preds = %"true verifyAndComputePtr"
  ret void

verifyNullCont126:              ; preds = %"GOTO or IF*6"
  br i1 %arg, label %"true verifyAndComputePtr127", label %"false verifyAndComputePtr128"

"true verifyAndComputePtr127":          ; preds = %verifyNullCont126
  br i1 %arg, label %"true verifyAndComputePtr131.GOTO or IF*6_crit_edge", label %"GOTO or IF*5"

"false verifyAndComputePtr128":         ; preds = %verifyNullCont126
  ret void

"true verifyAndComputePtr131.GOTO or IF*6_crit_edge":           ; preds = %"true verifyAndComputePtr127"
  %indvar.next = add i32 %indvar, 1             ; <i32> [#uses=1]
  br label %"GOTO or IF*6"

verifyNullCont172:              ; preds = %verifyNullCont81
  unreachable
}
