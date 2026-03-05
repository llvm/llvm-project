; RUN: llc -march=hexagon -O3 -enable-machine-unroller=true < %s
; REQUIRES: asserts

; This test used to fail with an "UNREACHABLE" executed in Machine Unroller due to a bug
; in computeDelta function.

%class.mrObjectRecord = type { i32, i32, %class.mrSurfaceList, i32, i32, i32, i32, i32, i32 }
%class.mrSurfaceList = type { %class.ggSolidTexture, %class.ggTrain }
%class.ggSolidTexture = type { ptr }
%class.ggTrain = type { ptr, i32, i32 }

@g = dso_local global i32 0, align 4

declare i32 @__gxx_personality_v0(...)

declare void @_Znaj() local_unnamed_addr

declare dso_local fastcc ptr @_ZN12ggDictionaryI14mrObjectRecordE6lookUpERK8ggString() unnamed_addr align 2

define dso_local fastcc void @_ZN7mrScene9AddObjectEP9mrSurfaceRK8ggStringS4_i(i1 %cond0, i1 %cond1) unnamed_addr align 2 personality ptr @__gxx_personality_v0 {
entry:
  br i1 %cond0, label %_ZN12ggDictionaryI10ggMaterialE6lookUpERK8ggString.exit, label %while.body.i.i.lr.ph

while.body.i.i.lr.ph:                             ; preds = %entry
  unreachable

_ZN12ggDictionaryI10ggMaterialE6lookUpERK8ggString.exit: ; preds = %entry
  %call5 = tail call fastcc ptr @_ZN12ggDictionaryI14mrObjectRecordE6lookUpERK8ggString()
  br i1 %cond1, label %if.then7, label %if.end11

if.then7:                                         ; preds = %_ZN12ggDictionaryI10ggMaterialE6lookUpERK8ggString.exit
  invoke void @_Znaj()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %if.then7
  br label %if.end11

lpad:                                             ; preds = %if.then7
  %0 = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } %0

if.end11:                                         ; preds = %invoke.cont, %_ZN12ggDictionaryI10ggMaterialE6lookUpERK8ggString.exit
  %surfaces.i.i7 = getelementptr inbounds %class.mrObjectRecord, ptr %call5, i32 0, i32 2, i32 1
  br label %for.body.i.i.i

for.cond.cleanup.i.i.i:                           ; preds = %for.body.i.i.i
  ret void

for.body.i.i.i:                                   ; preds = %for.body.i.i.i, %if.end11
  %i.0.i.i.i52 = phi i32 [ %inc.i.i.i, %for.body.i.i.i ], [ 0, %if.end11 ]
  %1 = load i32, ptr @g, align 4
  %2 = load ptr, ptr %surfaces.i.i7, align 4
  %arrayidx9.i.i.i = getelementptr inbounds ptr, ptr %2, i32 %i.0.i.i.i52
  store i32 %1, ptr %arrayidx9.i.i.i, align 4
  %inc.i.i.i = add nuw nsw i32 %i.0.i.i.i52, 1
  br i1 false, label %for.body.i.i.i, label %for.cond.cleanup.i.i.i
}
