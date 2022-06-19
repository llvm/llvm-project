; RUN: opt %loadPolly -polly-print-scops -disable-output < %s
;
; This crased at some point as we place %1 and %4 in the same equivalence class
; for invariant loads and when we remap SCEVs to use %4 instead of %1 AddRec SCEVs
; for the for.body.10 loop caused a crash as their operands were not invariant
; in the loop. While we know they are, ScalarEvolution does not. However, we can simply
; rewrite the AddRecs to hoist everything from the "start" out of the AddRec.
;
; Check we do not crash.
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.codebook.48.118.748.1882.2972 = type { i64, i64, i64, %struct.static_codebook.19.89.719.1853.2971*, float*, i32*, i32*, i8*, i32*, i32, i32 }
%struct.static_codebook.19.89.719.1853.2971 = type { i64, i64, i64*, i32, i64, i64, i32, i32, i64*, %struct.encode_aux_nearestmatch.16.86.716.1850.2968*, %struct.encode_aux_threshmatch.17.87.717.1851.2969*, %struct.encode_aux_pigeonhole.18.88.718.1852.2970*, i32 }
%struct.encode_aux_nearestmatch.16.86.716.1850.2968 = type { i64*, i64*, i64*, i64*, i64, i64 }
%struct.encode_aux_threshmatch.17.87.717.1851.2969 = type { float*, i64*, i32, i32 }
%struct.encode_aux_pigeonhole.18.88.718.1852.2970 = type { float, float, i32, i32, i64*, i64, i64*, i64*, i64* }

; Function Attrs: inlinehint nounwind uwtable
declare i64 @decode_packed_entry_number() #0

; Function Attrs: nounwind uwtable
define void @vorbis_book_decodev_set(%struct.codebook.48.118.748.1882.2972* %book) #1 {
entry:
  br i1 undef, label %for.body, label %return

for.cond.loopexit:                                ; preds = %for.body.10, %if.end
  br i1 undef, label %for.body, label %return

for.body:                                         ; preds = %for.cond.loopexit, %entry
  %call = tail call i64 @decode_packed_entry_number()
  br i1 undef, label %return, label %if.end

if.end:                                           ; preds = %for.body
  %valuelist = getelementptr inbounds %struct.codebook.48.118.748.1882.2972, %struct.codebook.48.118.748.1882.2972* %book, i64 0, i32 4
  %0 = load float*, float** %valuelist, align 8
  %sext = shl i64 %call, 32
  %conv4 = ashr exact i64 %sext, 32
  %dim = getelementptr inbounds %struct.codebook.48.118.748.1882.2972, %struct.codebook.48.118.748.1882.2972* %book, i64 0, i32 0
  %1 = load i64, i64* %dim, align 8
  %mul = mul nsw i64 %1, %conv4
  %add.ptr = getelementptr inbounds float, float* %0, i64 %mul
  %cmp8.7 = icmp sgt i64 %1, 0
  br i1 %cmp8.7, label %for.body.10, label %for.cond.loopexit

for.body.10:                                      ; preds = %for.body.10, %if.end
  %indvars.iv15 = phi i64 [ %indvars.iv.next16, %for.body.10 ], [ 0, %if.end ]
  %indvars.iv.next16 = add nuw nsw i64 %indvars.iv15, 1
  %arrayidx = getelementptr inbounds float, float* %add.ptr, i64 %indvars.iv15
  %2 = bitcast float* %arrayidx to i32*
  %3 = load i32, i32* %2, align 4
  %4 = load i64, i64* %dim, align 8
  %cmp8 = icmp slt i64 %indvars.iv.next16, %4
  br i1 %cmp8, label %for.body.10, label %for.cond.loopexit

return:                                           ; preds = %for.body, %for.cond.loopexit, %entry
  ret void
}
