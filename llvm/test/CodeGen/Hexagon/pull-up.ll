; RUN: llc -march=hexagon -mcpu=hexagonv73 -debug < %s
; This test used to ICE with "[MIisDualJumpCandidate] To BB(0) From BB(1)".

;REQUIRES: asserts

target datalayout = "e-m:e-p:32:32-i1:32-i64:64-a:0-v32:32-n16:32"
target triple = "hexagon-unknown--elf"

define void @main(i1 %cond) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  br label %for.body.ur.i.i

for.body.ur.i.i:                                  ; preds = %for.body.ur.i.i, %entry
  br i1 %cond, label %for.inc26.6, label %for.body.ur.i.i

invoke.cont:                                      ; preds = %for.inc26.6
  %call.i.i.i.i174 = invoke noalias i8* @_Znwj()
          to label %_ZNSt6vectorIlSaIlEE12_Construct_nEjRKl.exit unwind label %lpad

lpad:                                             ; preds = %for.inc26.6, %invoke.cont
  %0 = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } %0

_ZNSt6vectorIlSaIlEE12_Construct_nEjRKl.exit:     ; preds = %invoke.cont
  %incdec.ptr.i.ur.i.1 = getelementptr inbounds i8, i8* %call.i.i.i.i174, i32 36
  call void @llvm.memset.p0i8.i64(i8* %call.i.i.i.i174, i8 0, i64 36, i32 4, i1 false)
  call void @llvm.memset.p0i8.i64(i8* %incdec.ptr.i.ur.i.1, i8 0, i64 32, i32 4, i1 false)
  unreachable

for.inc26.6:                                      ; preds = %for.body.ur.i.i
  invoke void @_ZNSt6vectorIlSaIlEE7_InsertIPlEEvSt16_Vector_iteratorIlS0_ET_S6_St20forward_iterator_tag()
          to label %invoke.cont unwind label %lpad
}

declare i32 @__gxx_personality_v0(...)
declare noalias i8* @_Znwj()
declare void @_ZNSt6vectorIlSaIlEE7_InsertIPlEEvSt16_Vector_iteratorIlS0_ET_S6_St20forward_iterator_tag() align 2
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1)
