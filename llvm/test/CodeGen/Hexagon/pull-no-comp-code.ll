; RUN: llc -march=hexagon -O3 < %s
; This test checks that compensation code should not be inserted into the
; same packet where MI is to be pulled which prevents the pull-up of MI.
; REQUIRES: asserts
; ModuleID = 'bugpoint-reduced-simplified.bc'

%class.ggJitterSample2.5.77.107.149.155.167.179.215.251.269.275.281.287.293.317.329.347.425.437.687 = type { %class.ggSample2.4.76.106.148.154.166.178.214.250.268.274.280.286.292.316.328.346.424.436.686, i32, i32, double, double }
%class.ggSample2.4.76.106.148.154.166.178.214.250.268.274.280.286.292.316.328.346.424.436.686 = type { ptr, %class.ggTrain.3.75.105.147.153.165.177.213.249.267.273.279.285.291.315.327.345.423.435.685 }
%class.ggTrain.3.75.105.147.153.165.177.213.249.267.273.279.285.291.315.327.345.423.435.685 = type { ptr, i32, i32 }
%class.ggPoint2.2.74.104.146.152.164.176.212.248.266.272.278.284.290.314.326.344.422.434.684 = type { [2 x double] }

define void @_ZN15ggJitterSample28GenerateEv(ptr nocapture %this, i1 %cond0, i1 %cond1, i1 %cond2, i1 %cond3) unnamed_addr #0 align 2 {
entry:
  %nData.i = getelementptr inbounds %class.ggJitterSample2.5.77.107.149.155.167.179.215.251.269.275.281.287.293.317.329.347.425.437.687, ptr %this, i32 0, i32 0, i32 1, i32 1
  %memset.buf = alloca i8, i32 128, align 8
  %nan.sink = alloca %class.ggPoint2.2.74.104.146.152.164.176.212.248.266.272.278.284.290.314.326.344.422.434.684, align 8
  br i1 %cond0, label %for.cond2.preheader.lr.ph, label %return

for.cond2.preheader.lr.ph:                        ; preds = %entry
  %data.i = getelementptr inbounds %class.ggJitterSample2.5.77.107.149.155.167.179.215.251.269.275.281.287.293.317.329.347.425.437.687, ptr %this, i32 0, i32 0, i32 1, i32 0
  br label %for.body4

for.body4:                                        ; preds = %_ZN7ggTrainI8ggPoint2E6AppendES0_.exit, %for.cond2.preheader.lr.ph
  %call.i19 = tail call i32 @_Z7my_randv()
  %call.i21 = tail call i32 @_Z7my_randv()
  br i1 %cond1, label %if.then.i, label %_ZN7ggTrainI8ggPoint2E6AppendES0_.exit

if.then.i:                                        ; preds = %for.body4
  %0 = load ptr, ptr %data.i, align 4
  br i1 %cond2, label %for.end.i, label %arrayctor.loop.i

arrayctor.loop.i:                                 ; preds = %arrayctor.loop.i, %if.then.i
  call void @llvm.memset.p0.i64(ptr align 8 %memset.buf, i8 0, i64 128, i1 false)
  br i1 %cond3, label %arrayctor.loop.i, label %arrayctor.cont.loopexit.ur-lcssa.i

arrayctor.cont.loopexit.ur-lcssa.i:               ; preds = %arrayctor.loop.i
  %.pre.i = load i32, ptr %nData.i, align 4
  %cmp7.i27 = icmp sgt i32 %.pre.i, 1
  br i1 %cmp7.i27, label %for.body.for.body_crit_edge.i, label %for.end.i

for.body.for.body_crit_edge.i:                    ; preds = %for.body.for.body_crit_edge.i.for.body.for.body_crit_edge.i_crit_edge, %arrayctor.cont.loopexit.ur-lcssa.i
  %.pre24.i = phi ptr [ %.pre24.i.pre, %for.body.for.body_crit_edge.i.for.body.for.body_crit_edge.i_crit_edge ], [ null, %arrayctor.cont.loopexit.ur-lcssa.i ]
  %inc.i29 = phi i32 [ %inc.i, %for.body.for.body_crit_edge.i.for.body.for.body_crit_edge.i_crit_edge ], [ 1, %arrayctor.cont.loopexit.ur-lcssa.i ]
  %arrayidx9.phi.i28 = phi ptr [ %arrayidx9.inc.i, %for.body.for.body_crit_edge.i.for.body.for.body_crit_edge.i_crit_edge ], [ %0, %arrayctor.cont.loopexit.ur-lcssa.i ]
  %arrayidx9.inc.i = getelementptr %class.ggPoint2.2.74.104.146.152.164.176.212.248.266.272.278.284.290.314.326.344.422.434.684, ptr %arrayidx9.phi.i28, i32 1
  %arrayidx.i = getelementptr inbounds %class.ggPoint2.2.74.104.146.152.164.176.212.248.266.272.278.284.290.314.326.344.422.434.684, ptr %.pre24.i, i32 %inc.i29
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %arrayidx.i, ptr align 8 %arrayidx9.inc.i, i32 16, i1 false)
  %inc.i = add nsw i32 %inc.i29, 1
  %1 = load i32, ptr %nData.i, align 4
  %cmp7.i = icmp slt i32 %inc.i, %1
  br i1 %cmp7.i, label %for.body.for.body_crit_edge.i.for.body.for.body_crit_edge.i_crit_edge, label %for.end.i

for.body.for.body_crit_edge.i.for.body.for.body_crit_edge.i_crit_edge: ; preds = %for.body.for.body_crit_edge.i
  %.pre24.i.pre = load ptr, ptr %data.i, align 4
  br label %for.body.for.body_crit_edge.i

for.end.i:                                        ; preds = %for.body.for.body_crit_edge.i, %arrayctor.cont.loopexit.ur-lcssa.i, %if.then.i
  %isnull.i = icmp eq ptr %0, null
  br i1 %isnull.i, label %_ZN7ggTrainI8ggPoint2E6AppendES0_.exit, label %delete.notnull.i

delete.notnull.i:                                 ; preds = %for.end.i
  tail call void @_ZdaPv() #4
  unreachable

_ZN7ggTrainI8ggPoint2E6AppendES0_.exit:           ; preds = %for.end.i, %for.body4
  %nan.sink.0 = getelementptr inbounds %class.ggPoint2.2.74.104.146.152.164.176.212.248.266.272.278.284.290.314.326.344.422.434.684, ptr %nan.sink, i32 0, i32 0, i32 0
  %nan.sink.1 = getelementptr inbounds %class.ggPoint2.2.74.104.146.152.164.176.212.248.266.272.278.284.290.314.326.344.422.434.684, ptr %nan.sink, i32 0, i32 0, i32 1
  store double 0x7FF8000000000000, ptr %nan.sink.0, align 8
  store double 0x7FF8000000000000, ptr %nan.sink.1, align 8
  br label %for.body4

return:                                           ; preds = %entry
  ret void
}

; Function Attrs: nobuiltin nounwind
declare void @_ZdaPv() #1

declare i32 @_Z7my_randv() #0

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i32(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg) #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #3

attributes #0 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nobuiltin nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #4 = { builtin nounwind }
