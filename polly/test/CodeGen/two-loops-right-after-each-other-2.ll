; RUN: opt %loadNPMPolly -passes=polly-codegen -S < %s | FileCheck %s

; CHECK:       polly.merge_new_and_old:
; CHECK-NEXT:    merge = phi

%struct.ImageParameters = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, i32, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [9 x [16 x [16 x i16]]], [5 x [16 x [16 x i16]]], [9 x [8 x [8 x i16]]], [2 x [4 x [16 x [16 x i16]]]], [16 x [16 x i16]], [16 x [16 x i32]], ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, [4 x [4 x i32]], i32, i32, i32, i32, i32, double, i32, i32, i32, i32, ptr, ptr, ptr, ptr, [15 x i16], i32, i32, i32, i32, i32, i32, i32, i32, [6 x [32 x i32]], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [1 x i32], i32, i32, [2 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [3 x [2 x i32]], [2 x i32], i32, i32, i16, i32, i32, i32, i32, i32 }
%struct.Picture = type { i32, i32, [100 x ptr], i32, float, float, float }
%struct.Slice = type { i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, i32, ptr, ptr, ptr, i32, ptr, ptr, ptr, ptr, [3 x [2 x i32]] }
%struct.datapartition = type { ptr, %struct.EncodingEnvironment, %struct.EncodingEnvironment }
%struct.Bitstream = type { i32, i32, i8, i32, i32, i8, i8, i32, i32, ptr, i32 }
%struct.EncodingEnvironment = type { i32, i32, i32, i32, i32, ptr, ptr, i32, i32 }
%struct.MotionInfoContexts = type { [3 x [11 x %struct.BiContextType]], [2 x [9 x %struct.BiContextType]], [2 x [10 x %struct.BiContextType]], [2 x [6 x %struct.BiContextType]], [4 x %struct.BiContextType], [4 x %struct.BiContextType], [3 x %struct.BiContextType] }
%struct.BiContextType = type { i16, i8, i64 }
%struct.TextureInfoContexts = type { [2 x %struct.BiContextType], [3 x [4 x %struct.BiContextType]], [10 x [4 x %struct.BiContextType]], [10 x [15 x %struct.BiContextType]], [10 x [15 x %struct.BiContextType]], [10 x [5 x %struct.BiContextType]], [10 x [5 x %struct.BiContextType]], [10 x [15 x %struct.BiContextType]], [10 x [15 x %struct.BiContextType]] }
%struct.macroblock = type { i32, i32, i32, [2 x i32], i32, [8 x i32], ptr, i32, [2 x [4 x [4 x [2 x i32]]]], [16 x i8], [16 x i8], i32, i64, [4 x i32], [4 x i32], i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i16, double, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%struct.DecRefPicMarking = type { i32, i32, i32, i32, i32, ptr }

@img = external global ptr, align 8

define void @intrapred_luma() {
entry:
  %PredPel = alloca [13 x i16], align 16
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  br i1 undef, label %for.body, label %for.body.262

for.body.262:                                     ; preds = %for.body
  %0 = load ptr, ptr @img, align 8
  br label %for.body.280

for.body.280:                                     ; preds = %for.body.280, %for.body.262
  %indvars.iv66 = phi i64 [ 0, %for.body.262 ], [ %indvars.iv.next67, %for.body.280 ]
  %arrayidx282 = getelementptr inbounds [13 x i16], ptr %PredPel, i64 0, i64 1
  %arrayidx283 = getelementptr inbounds i16, ptr %arrayidx282, i64 %indvars.iv66
  %1 = load i16, ptr %arrayidx283, align 2
  %arrayidx289 = getelementptr inbounds %struct.ImageParameters, ptr %0, i64 0, i32 47, i64 0, i64 2, i64 %indvars.iv66
  store i16 %1, ptr %arrayidx289, align 2
  %indvars.iv.next67 = add nuw nsw i64 %indvars.iv66, 1
  br i1 false, label %for.body.280, label %for.end.298

for.end.298:                                      ; preds = %for.body.280
  %2 = load ptr, ptr @img, align 8
  br label %for.body.310

for.body.310:                                     ; preds = %for.body.310, %for.end.298
  %indvars.iv = phi i64 [ 0, %for.end.298 ], [ %indvars.iv.next, %for.body.310 ]
  %InterScopSext = sext i16 %1 to i64
  %arrayidx312 = getelementptr inbounds [13 x i16], ptr %PredPel, i64 0, i64 %InterScopSext
  %arrayidx313 = getelementptr inbounds i16, ptr %arrayidx312, i64 %indvars.iv
  %3 = load i16, ptr %arrayidx313, align 2
  %arrayidx322 = getelementptr inbounds %struct.ImageParameters, ptr %2, i64 0, i32 47, i64 1, i64 %indvars.iv, i64 1
  store i16 %3, ptr %arrayidx322, align 2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br i1 false, label %for.body.310, label %for.end.328

for.end.328:                                      ; preds = %for.body.310
  ret void
}
