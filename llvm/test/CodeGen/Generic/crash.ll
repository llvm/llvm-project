; RUN: llc %s -o -

; PR6332
%struct.AVCodecTag = type {}
@ff_codec_bmp_tags = external global [0 x %struct.AVCodecTag]
@tags = global [1 x %struct.AVCodecTag*] [%struct.AVCodecTag* getelementptr
inbounds ([0 x %struct.AVCodecTag]* @ff_codec_bmp_tags, i32 0, i32 0)]


; rdar://8878965

%struct.CAMERA = type { [3 x double], [3 x double], [3 x double], [3 x double], [3 x double], [3 x double], double, double, i32, double, double, i32, double, i32* }

define void @Parse_Camera(%struct.CAMERA** nocapture %Camera_Ptr) nounwind {
entry:
%.pre = load %struct.CAMERA** %Camera_Ptr, align 4
%0 = getelementptr inbounds %struct.CAMERA* %.pre, i32 0, i32 1, i32 0
%1 = getelementptr inbounds %struct.CAMERA* %.pre, i32 0, i32 1, i32 2
br label %bb32

bb32:                                             ; preds = %bb6
%2 = load double* %0, align 4
%3 = load double* %1, align 4
%4 = load double* %0, align 4
call void @Parse_Vector(double* %0) nounwind
%5 = call i32 @llvm.objectsize.i32(i8* undef, i1 false)
%6 = icmp eq i32 %5, -1
br i1 %6, label %bb34, label %bb33

bb33:                                             ; preds = %bb32
unreachable

bb34:                                             ; preds = %bb32
unreachable

}

declare void @Parse_Vector(double*)
declare i32 @llvm.objectsize.i32(i8*, i1)


; PR9578
%struct.S0 = type { i32, i8, i32 }

define void @func_82() nounwind optsize {
entry:
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i, %entry
  br i1 undef, label %func_74.exit.for.cond29.thread_crit_edge, label %for.body.i

func_74.exit.for.cond29.thread_crit_edge:         ; preds = %for.body.i
  %f13576.pre = getelementptr inbounds %struct.S0* undef, i64 0, i32 1
  store i8 0, i8* %f13576.pre, align 4, !tbaa !0
  br label %lbl_468

lbl_468:                                          ; preds = %lbl_468, %func_74.exit.for.cond29.thread_crit_edge
  %f13577.ph = phi i8* [ %f13576.pre, %func_74.exit.for.cond29.thread_crit_edge ], [ %f135.pre, %lbl_468 ]
  store i8 1, i8* %f13577.ph, align 1
  %f135.pre = getelementptr inbounds %struct.S0* undef, i64 0, i32 1
  br i1 undef, label %lbl_468, label %for.end74

for.end74:                                        ; preds = %lbl_468
  ret void
}

!0 = metadata !{metadata !"omnipotent char", metadata !1}
!1 = metadata !{metadata !"Simple C/C++ TBAA", null}
