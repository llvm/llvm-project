; RUN: llc < %s -mcpu=cortex-a15 -verify-machineinstrs -arm-atomic-cfg-tidy=0 | FileCheck %s

; Check a spill right after a function call with large struct byval is correctly
; generated.
; PR16393

; We expect 4-byte spill and reload to be generated.

; CHECK: set_stored_macroblock_parameters
; CHECK: str r{{.*}}, [sp, {{#[0-9]+}}] @ 4-byte Spill
; CHECK: ldr r{{.*}}, [lr, {{#[0-9]+}}] @ 4-byte Reload

target triple = "armv7l-unknown-linux-gnueabihf"

%structA = type { double, [16 x [16 x i16]], [16 x [16 x i16]], [16 x [16 x i16]], i32****, i32***, i32, i16, [4 x i32], [4 x i32], i8**, [16 x i8], [16 x i8], i32, i64, i32, i16******, i16******, [2 x [4 x [4 x i8]]], i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%structB = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8**, i8**, i32, i32***, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [9 x [16 x [16 x i16]]], [5 x [16 x [16 x i16]]], [9 x [8 x [8 x i16]]], [2 x [4 x [16 x [16 x i16]]]], [16 x [16 x i16]], [16 x [16 x i32]], i32****, i32***, i32***, i32***, i32****, i32****, %structC*, %structD*, %structK*, i32*, i32*, i32, i32, i32, i32, [4 x [4 x i32]], i32, i32, i32, i32, i32, double, i32, i32, i32, i32, i16******, i16******, i16******, i16******, [15 x i16], i32, i32, i32, i32, i32, i32, i32, i32, [6 x [32 x i32]], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [1 x i32], i32, i32, [2 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %structL*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, double**, double***, i32***, double**, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [3 x [2 x i32]], [2 x i32], i32, i32, i16, i32, i32, i32, i32, i32 }
%structC = type { i32, i32, [100 x %structD*], i32, float, float, float }
%structD = type { i32, i32, i32, i32, i32, i32, %structE*, %structH*, %structJ*, i32, i32*, i32*, i32*, i32, i32*, i32*, i32*, i32 (i32)*, [3 x [2 x i32]] }
%structE = type { %structF*, %structG, %structG }
%structF = type { i32, i32, i8, i32, i32, i8, i8, i32, i32, i8*, i32 }
%structG = type { i32, i32, i32, i32, i32, i8*, i32*, i32, i32 }
%structH = type { [3 x [11 x %structI]], [2 x [9 x %structI]], [2 x [10 x %structI]], [2 x [6 x %structI]], [4 x %structI], [4 x %structI], [3 x %structI] }
%structI = type { i16, i8, i32 }
%structJ = type { [2 x %structI], [4 x %structI], [3 x [4 x %structI]], [10 x [4 x %structI]], [10 x [15 x %structI]], [10 x [15 x %structI]], [10 x [5 x %structI]], [10 x [5 x %structI]], [10 x [15 x %structI]], [10 x [15 x %structI]] }
%structK = type { i32, i32, i32, [2 x i32], i32, [8 x i32], %structK*, %structK*, i32, [2 x [4 x [4 x [2 x i32]]]], [16 x i8], [16 x i8], i32, i64, [4 x i32], [4 x i32], i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i16, double, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%structL = type { i32, i32, i32, i32, i32, %structL* }
%structM = type { i32, i32, i32, i32, i32, i32, [6 x [33 x i64]], [6 x [33 x i64]], [6 x [33 x i64]], [6 x [33 x i64]], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i16**, i16****, i16****, i16*****, i16***, i8*, i8***, i64***, i64***, i16****, i8**, i8**, %structM*, %structM*, %structM*, i32, i32, i32, i32, i32, i32, i32 }
%structN = type { i32, [16 x [16 x i32]], [16 x [16 x i32]], [16 x [16 x i32]], [3 x [16 x [16 x i32]]], [4 x i16], [4 x i8], [4 x i8], [4 x i8], [16 x [16 x i16]], [16 x [16 x i16]], [16 x [16 x i32]] }

@cofAC = external global i32****, align 4
@cofDC = external global i32***, align 4
@rdopt = external global %structA*, align 4
@img = external global %structB*
@enc_picture = external global %structM*
@si_frame_indicator = external global i32, align 4
@sp2_frame_indicator = external global i32, align 4
@lrec = external global i32**, align 4
@tr8x8 = external global %structN, align 4
@best_mode = external global i16, align 2
@best_c_imode = external global i32, align 4
@best_i16offset = external global i32, align 4
@bi_pred_me = external global i16, align 2
@b8mode = external global [4 x i32], align 4
@b8pdir = external global [4 x i32], align 4
@b4_intra_pred_modes = external global [16 x i8], align 1
@b8_intra_pred_modes8x8 = external global [16 x i8], align 1
@b4_ipredmode = external global [16 x i8], align 1
@b8_ipredmode8x8 = external global [4 x [4 x i8]], align 1
@rec_mbY = external global [16 x [16 x i16]], align 2
@lrec_rec = external global [16 x [16 x i32]], align 4
@rec_mbU = external global [16 x [16 x i16]], align 2
@rec_mbV = external global [16 x [16 x i16]], align 2
@lrec_rec_U = external global [16 x [16 x i32]], align 4
@lrec_uv = external global i32***, align 4
@lrec_rec_V = external global [16 x [16 x i32]], align 4
@cbp = external global i32, align 4
@cbp_blk = external global i64, align 8
@luma_transform_size_8x8_flag = external global i32, align 4
@frefframe = external global [4 x [4 x i8]], align 1
@brefframe = external global [4 x [4 x i8]], align 1

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) #0

; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1) #0

; Function Attrs: nounwind
declare void @SetMotionVectorsMB(%structK* nocapture, i32) #1

; Function Attrs: nounwind
define void @set_stored_macroblock_parameters(i16 %a0, i32 %a1) #1 {
entry:
  %0 = load %structB*, %structB** @img, align 4
  %1 = load i32, i32* undef, align 4
  %mb_data = getelementptr inbounds %structB, %structB* %0, i32 0, i32 61
  %2 = load %structK*, %structK** %mb_data, align 4
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  br i1 undef, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  br i1 undef, label %for.body20, label %if.end

for.body20:                                       ; preds = %for.end
  unreachable

if.end:                                           ; preds = %for.end
  br i1 undef, label %if.end40, label %for.cond31.preheader

for.cond31.preheader:                             ; preds = %if.end
  unreachable

if.end40:                                         ; preds = %if.end
  br i1 undef, label %if.end43, label %if.then42

if.then42:                                        ; preds = %if.end40
  br label %if.end43

if.end43:                                         ; preds = %if.then42, %if.end40
  br i1 undef, label %if.end164, label %for.cond47.preheader

for.cond47.preheader:                             ; preds = %if.end43
  br i1 undef, label %for.body119, label %if.end164

for.body119:                                      ; preds = %for.body119, %for.cond47.preheader
  br i1 undef, label %for.body119, label %if.end164

if.end164:                                        ; preds = %for.body119, %for.cond47.preheader, %if.end43
  store i32*** null, i32**** @cofDC, align 4
  %mb_type = getelementptr inbounds %structK, %structK* %2, i32 %1, i32 8
  br i1 undef, label %if.end230, label %if.then169

if.then169:                                       ; preds = %if.end164
  br i1 undef, label %for.cond185.preheader, label %for.cond210.preheader

for.cond185.preheader:                            ; preds = %if.then169
  unreachable

for.cond210.preheader:                            ; preds = %if.then169
  unreachable

if.end230:                                        ; preds = %if.end164
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 undef, i8* align 4 bitcast ([4 x i32]* @b8mode to i8*), i32 16, i1 false)
  %b8pdir = getelementptr inbounds %structK, %structK* %2, i32 %1, i32 15
  %3 = bitcast [4 x i32]* %b8pdir to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %3, i8* align 4 bitcast ([4 x i32]* @b8pdir to i8*), i32 16, i1 false)
  %tobool.if.end230 = icmp ne i8* %3, null
  br i1 %tobool.if.end230, label %if.end236, label %if.then233

if.then233:                                       ; preds = %if.end230
  unreachable

if.end236:                                        ; preds = %if.end230
  %cmp242 = icmp ne i16 %a0, 8
  %4 = load i32, i32* @luma_transform_size_8x8_flag, align 4
  %tobool245 = icmp ne i32 %4, 0
  %or.cond812 = or i1 %cmp242, %tobool245
  br i1 %or.cond812, label %if.end249, label %land.lhs.true246

land.lhs.true246:                                 ; preds = %if.end236
  %tobool246 = icmp sgt i32 %4, 1
  br i1 %tobool246, label %if.end249, label %if.then248

if.then248:                                       ; preds = %land.lhs.true246
  tail call void asm sideeffect "", "~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11}"() nounwind
  tail call void @RestoreMVBlock8x8(i32 1, i32 0, %structN* byval @tr8x8, i32 0) #0
  tail call void @RestoreMVBlock8x8(i32 1, i32 2, %structN* byval @tr8x8, i32 0) #0
  tail call void @RestoreMVBlock8x8(i32 1, i32 3, %structN* byval @tr8x8, i32 0) #0
  br label %if.end249

if.end249:                                        ; preds = %if.then248, %land.lhs.true246, %if.end236
  %5 = load i32, i32* @luma_transform_size_8x8_flag, align 4
  %6 = load %structA*, %structA** @rdopt, align 4
  %luma_transform_size_8x8_flag264 = getelementptr inbounds %structA, %structA* %6, i32 0, i32 21
  store i32 %5, i32* %luma_transform_size_8x8_flag264, align 4
  %7 = load i32, i32* undef, align 4
  %add281 = add nsw i32 %7, %4
  br label %for.body285

for.body285:                                      ; preds = %for.inc503, %if.end249
  %8 = phi %structB* [ undef, %if.end249 ], [ %.pre1155, %for.inc503 ]
  %i.21103 = phi i32 [ 0, %if.end249 ], [ %inc504, %for.inc503 ]
  %block_x286 = getelementptr inbounds %structB, %structB* %8, i32 0, i32 37
  %9 = load i32, i32* %block_x286, align 4
  %add287 = add nsw i32 %9, %i.21103
  %shr289 = ashr i32 %i.21103, 1
  %add290 = add nsw i32 %shr289, 0
  %arrayidx292 = getelementptr inbounds %structK, %structK* %2, i32 %1, i32 15, i32 %add290
  %10 = load %structM*, %structM** @enc_picture, align 4
  %ref_idx = getelementptr inbounds %structM, %structM* %10, i32 0, i32 35
  %11 = load i8***, i8**** %ref_idx, align 4
  %12 = load i8**, i8*** %11, align 4
  %arrayidx313 = getelementptr inbounds i8*, i8** %12, i32 %add281
  %13 = load i8*, i8** %arrayidx313, align 4
  %arrayidx314 = getelementptr inbounds i8, i8* %13, i32 %add287
  store i8 -1, i8* %arrayidx314, align 1
  %14 = load %structB*, %structB** @img, align 4
  %MbaffFrameFlag327 = getelementptr inbounds %structB, %structB* %14, i32 0, i32 100
  %15 = load i32, i32* %MbaffFrameFlag327, align 4
  %tobool328 = icmp eq i32 %15, %a1
  br i1 %tobool328, label %if.end454, label %if.then329

if.then329:                                       ; preds = %for.body285
  %16 = load %structA*, %structA** @rdopt, align 4
  br label %if.end454

if.end454:                                        ; preds = %if.then329, %for.body285
  %17 = load i32, i32* %arrayidx292, align 4
  %cmp457 = icmp eq i32 %17, 0
  br i1 %cmp457, label %if.then475, label %lor.lhs.false459

lor.lhs.false459:                                 ; preds = %if.end454
  %18 = load i32, i32* %mb_type, align 4
  switch i32 %18, label %for.inc503 [
    i32 9, label %if.then475
    i32 11, label %if.then475
    i32 13, label %if.then475
    i32 14, label %if.then475
  ]

if.then475:                                       ; preds = %lor.lhs.false459, %lor.lhs.false459, %lor.lhs.false459, %lor.lhs.false459, %if.end454
  store i16 0, i16* undef, align 2
  br label %for.inc503

for.inc503:                                       ; preds = %if.then475, %lor.lhs.false459
  %inc504 = add nsw i32 %i.21103, 1
  %.pre1155 = load %structB*, %structB** @img, align 4
  br label %for.body285
}

; Function Attrs: nounwind
declare void @update_offset_params(i32, i32) #1

; Function Attrs: nounwind
declare void @RestoreMVBlock8x8(i32, i32, %structN* byval nocapture, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
