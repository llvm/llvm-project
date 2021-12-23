; RUN: opt %loadPolly -polly-vectorizer=stripmine -polly-isl-arg=--no-schedule-serialize-sccs -polly-tiling=0 -polly-opt-isl -analyze - < %s | FileCheck %s

; isl_schedule_node_band_sink may sink into multiple children.
; https://llvm.org/PR52637

%struct.v4l2_sliced_vbi_data = type { [48 x i8] }

define void @vivid_vbi_gen_sliced() {
entry:
  br label %for.body

for.body:                                         ; preds = %vivid_vbi_gen_teletext.exit, %entry
  %i.015 = phi i32 [ 0, %entry ], [ %inc, %vivid_vbi_gen_teletext.exit ]
  %data0.014 = phi %struct.v4l2_sliced_vbi_data* [ null, %entry ], [ %incdec.ptr, %vivid_vbi_gen_teletext.exit ]
  %arraydecay = getelementptr inbounds %struct.v4l2_sliced_vbi_data, %struct.v4l2_sliced_vbi_data* %data0.014, i32 0, i32 0, i32 0
  %arrayidx.i = getelementptr inbounds %struct.v4l2_sliced_vbi_data, %struct.v4l2_sliced_vbi_data* %data0.014, i32 0, i32 0, i32 6
  %0 = load i8, i8* %arrayidx.i, align 1
  store i8 %0, i8* %arraydecay, align 1
  br label %for.body.for.body_crit_edge.i

for.body.for.body_crit_edge.i:                    ; preds = %for.body.for.body_crit_edge.i, %for.body
  %inc10.i13 = phi i32 [ 1, %for.body ], [ %inc10.i, %for.body.for.body_crit_edge.i ]
  %arrayidx2.phi.trans.insert.i = getelementptr inbounds %struct.v4l2_sliced_vbi_data, %struct.v4l2_sliced_vbi_data* %data0.014, i32 0, i32 0, i32 %inc10.i13
  store i8 0, i8* %arrayidx2.phi.trans.insert.i, align 1
  %inc10.i = add nuw nsw i32 %inc10.i13, 1
  %exitcond.not.i = icmp eq i32 %inc10.i13, 42
  br i1 %exitcond.not.i, label %vivid_vbi_gen_teletext.exit, label %for.body.for.body_crit_edge.i

vivid_vbi_gen_teletext.exit:                      ; preds = %for.body.for.body_crit_edge.i
  %incdec.ptr = getelementptr inbounds %struct.v4l2_sliced_vbi_data, %struct.v4l2_sliced_vbi_data* %data0.014, i32 1
  %inc = add nuw nsw i32 %i.015, 1
  %exitcond.not = icmp eq i32 %i.015, 1
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %vivid_vbi_gen_teletext.exit
  ret void
}


; CHECK: schedule:
; CHECK:   schedule:
; CHECK:     mark: "SIMD"
; CHECK:       schedule:
; CHECK:     mark: "SIMD"
; CHECK:       schedule:
