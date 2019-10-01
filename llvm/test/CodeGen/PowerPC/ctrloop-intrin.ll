; RUN: llc -verify-machineinstrs < %s
; ModuleID = 'new.bc'
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le--linux-gnu"

@.str.87 = external hidden unnamed_addr constant [5 x i8], align 1
@.str.1.88 = external hidden unnamed_addr constant [4 x i8], align 1
@.str.2.89 = external hidden unnamed_addr constant [5 x i8], align 1
@.str.3.90 = external hidden unnamed_addr constant [4 x i8], align 1
@.str.4.91 = external hidden unnamed_addr constant [14 x i8], align 1
@.str.5.92 = external hidden unnamed_addr constant [13 x i8], align 1
@.str.6.93 = external hidden unnamed_addr constant [10 x i8], align 1
@.str.7.94 = external hidden unnamed_addr constant [9 x i8], align 1
@.str.8.95 = external hidden unnamed_addr constant [2 x i8], align 1
@.str.9.96 = external hidden unnamed_addr constant [2 x i8], align 1
@.str.10.97 = external hidden unnamed_addr constant [3 x i8], align 1
@.str.11.98 = external hidden unnamed_addr constant [3 x i8], align 1

; Function Attrs: nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #0

; Function Attrs: nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #0

; Function Attrs: nounwind
declare i8* @halide_string_to_string(i8*, i8*, i8*) #1

; Function Attrs: nounwind
declare i8* @halide_int64_to_string(i8*, i8*, i64, i32) #1

; Function Attrs: nounwind
define weak i8* @halide_double_to_string(i8* %dst, i8* %end, double %arg, i32 %scientific) #1 {
entry:
  %arg.addr = alloca double, align 8
  %bits = alloca i64, align 8
  %buf = alloca [512 x i8], align 1
  store double %arg, double* %arg.addr, align 8, !tbaa !4
  %0 = bitcast i64* %bits to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #0
  store i64 0, i64* %bits, align 8, !tbaa !8
  %1 = bitcast double* %arg.addr to i8*
  %call = call i8* @memcpy(i8* %0, i8* %1, i64 8) #2
  %2 = load i64, i64* %bits, align 8, !tbaa !8
  %and = and i64 %2, 4503599627370495
  %shr = lshr i64 %2, 52
  %shr.tr = trunc i64 %shr to i32
  %conv = and i32 %shr.tr, 2047
  %shr2 = lshr i64 %2, 63
  %conv3 = trunc i64 %shr2 to i32
  %cmp = icmp eq i32 %conv, 2047
  br i1 %cmp, label %if.then, label %if.else.15

if.then:                                          ; preds = %entry
  %tobool = icmp eq i64 %and, 0
  %tobool5 = icmp ne i32 %conv3, 0
  br i1 %tobool, label %if.else.9, label %if.then.4

if.then.4:                                        ; preds = %if.then
  br i1 %tobool5, label %if.then.6, label %if.else

if.then.6:                                        ; preds = %if.then.4
  %call7 = call i8* @halide_string_to_string(i8* %dst, i8* %end, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.87, i64 0, i64 0)) #3
  br label %cleanup.148

if.else:                                          ; preds = %if.then.4
  %call8 = call i8* @halide_string_to_string(i8* %dst, i8* %end, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1.88, i64 0, i64 0)) #3
  br label %cleanup.148

if.else.9:                                        ; preds = %if.then
  br i1 %tobool5, label %if.then.11, label %if.else.13

if.then.11:                                       ; preds = %if.else.9
  %call12 = call i8* @halide_string_to_string(i8* %dst, i8* %end, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.2.89, i64 0, i64 0)) #3
  br label %cleanup.148

if.else.13:                                       ; preds = %if.else.9
  %call14 = call i8* @halide_string_to_string(i8* %dst, i8* %end, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.3.90, i64 0, i64 0)) #3
  br label %cleanup.148

if.else.15:                                       ; preds = %entry
  %cmp16 = icmp eq i32 %conv, 0
  %cmp17 = icmp eq i64 %and, 0
  %or.cond = and i1 %cmp17, %cmp16
  br i1 %or.cond, label %if.then.18, label %if.end.32

if.then.18:                                       ; preds = %if.else.15
  %tobool19 = icmp eq i32 %scientific, 0
  %tobool21 = icmp ne i32 %conv3, 0
  br i1 %tobool19, label %if.else.26, label %if.then.20

if.then.20:                                       ; preds = %if.then.18
  br i1 %tobool21, label %if.then.22, label %if.else.24

if.then.22:                                       ; preds = %if.then.20
  %call23 = call i8* @halide_string_to_string(i8* %dst, i8* %end, i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str.4.91, i64 0, i64 0)) #3
  br label %cleanup.148

if.else.24:                                       ; preds = %if.then.20
  %call25 = call i8* @halide_string_to_string(i8* %dst, i8* %end, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.5.92, i64 0, i64 0)) #3
  br label %cleanup.148

if.else.26:                                       ; preds = %if.then.18
  br i1 %tobool21, label %if.then.28, label %if.else.30

if.then.28:                                       ; preds = %if.else.26
  %call29 = call i8* @halide_string_to_string(i8* %dst, i8* %end, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.6.93, i64 0, i64 0)) #3
  br label %cleanup.148

if.else.30:                                       ; preds = %if.else.26
  %call31 = call i8* @halide_string_to_string(i8* %dst, i8* %end, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.7.94, i64 0, i64 0)) #3
  br label %cleanup.148

if.end.32:                                        ; preds = %if.else.15
  %tobool33 = icmp eq i32 %conv3, 0
  br i1 %tobool33, label %if.end.37, label %if.then.34

if.then.34:                                       ; preds = %if.end.32
  %call35 = call i8* @halide_string_to_string(i8* %dst, i8* %end, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.8.95, i64 0, i64 0)) #3
  %sub36 = fsub double -0.000000e+00, %arg
  store double %sub36, double* %arg.addr, align 8, !tbaa !4
  br label %if.end.37

if.end.37:                                        ; preds = %if.then.34, %if.end.32
  %.pr = phi double [ %sub36, %if.then.34 ], [ %arg, %if.end.32 ]
  %dst.addr.0 = phi i8* [ %call35, %if.then.34 ], [ %dst, %if.end.32 ]
  %tobool38 = icmp eq i32 %scientific, 0
  br i1 %tobool38, label %if.else.62, label %while.condthread-pre-split

while.condthread-pre-split:                       ; preds = %if.end.37
  %cmp40.261 = fcmp olt double %.pr, 1.000000e+00
  br i1 %cmp40.261, label %while.body, label %while.cond.41thread-pre-split

while.body:                                       ; preds = %while.body, %while.condthread-pre-split
  %exponent_base_10.0262 = phi i32 [ %dec, %while.body ], [ 0, %while.condthread-pre-split ]
  %3 = phi double [ %mul, %while.body ], [ %.pr, %while.condthread-pre-split ]
  %mul = fmul double %3, 1.000000e+01
  %dec = add nsw i32 %exponent_base_10.0262, -1
  %cmp40 = fcmp olt double %mul, 1.000000e+00
  br i1 %cmp40, label %while.body, label %while.cond.while.cond.41thread-pre-split_crit_edge

while.cond.while.cond.41thread-pre-split_crit_edge: ; preds = %while.body
  store double %mul, double* %arg.addr, align 8, !tbaa !4
  br label %while.cond.41thread-pre-split

while.cond.41thread-pre-split:                    ; preds = %while.cond.while.cond.41thread-pre-split_crit_edge, %while.condthread-pre-split
  %.pr246 = phi double [ %mul, %while.cond.while.cond.41thread-pre-split_crit_edge ], [ %.pr, %while.condthread-pre-split ]
  %exponent_base_10.0.lcssa = phi i32 [ %dec, %while.cond.while.cond.41thread-pre-split_crit_edge ], [ 0, %while.condthread-pre-split ]
  %cmp42.257 = fcmp ult double %.pr246, 1.000000e+01
  br i1 %cmp42.257, label %while.end.44, label %while.body.43

while.body.43:                                    ; preds = %while.body.43, %while.cond.41thread-pre-split
  %exponent_base_10.1258 = phi i32 [ %inc, %while.body.43 ], [ %exponent_base_10.0.lcssa, %while.cond.41thread-pre-split ]
  %4 = phi double [ %div, %while.body.43 ], [ %.pr246, %while.cond.41thread-pre-split ]
  %div = fdiv double %4, 1.000000e+01
  %inc = add nsw i32 %exponent_base_10.1258, 1
  %cmp42 = fcmp ult double %div, 1.000000e+01
  br i1 %cmp42, label %while.cond.41.while.end.44_crit_edge, label %while.body.43

while.cond.41.while.end.44_crit_edge:             ; preds = %while.body.43
  store double %div, double* %arg.addr, align 8, !tbaa !4
  br label %while.end.44

while.end.44:                                     ; preds = %while.cond.41.while.end.44_crit_edge, %while.cond.41thread-pre-split
  %exponent_base_10.1.lcssa = phi i32 [ %inc, %while.cond.41.while.end.44_crit_edge ], [ %exponent_base_10.0.lcssa, %while.cond.41thread-pre-split ]
  %.lcssa = phi double [ %div, %while.cond.41.while.end.44_crit_edge ], [ %.pr246, %while.cond.41thread-pre-split ]
  %mul45 = fmul double %.lcssa, 1.000000e+06
  %add = fadd double %mul45, 5.000000e-01
  %conv46 = fptoui double %add to i64
  %div47 = udiv i64 %conv46, 1000000
  %5 = mul i64 %div47, -1000000
  %sub49 = add i64 %conv46, %5
  %call50 = call i8* @halide_int64_to_string(i8* %dst.addr.0, i8* %end, i64 %div47, i32 1) #3
  %call51 = call i8* @halide_string_to_string(i8* %call50, i8* %end, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.9.96, i64 0, i64 0)) #3
  %call52 = call i8* @halide_int64_to_string(i8* %call51, i8* %end, i64 %sub49, i32 6) #3
  %cmp53 = icmp sgt i32 %exponent_base_10.1.lcssa, -1
  br i1 %cmp53, label %if.then.54, label %if.else.56

if.then.54:                                       ; preds = %while.end.44
  %call55 = call i8* @halide_string_to_string(i8* %call52, i8* %end, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.10.97, i64 0, i64 0)) #3
  br label %if.end.59

if.else.56:                                       ; preds = %while.end.44
  %call57 = call i8* @halide_string_to_string(i8* %call52, i8* %end, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.11.98, i64 0, i64 0)) #3
  %sub58 = sub nsw i32 0, %exponent_base_10.1.lcssa
  br label %if.end.59

if.end.59:                                        ; preds = %if.else.56, %if.then.54
  %exponent_base_10.2 = phi i32 [ %exponent_base_10.1.lcssa, %if.then.54 ], [ %sub58, %if.else.56 ]
  %dst.addr.1 = phi i8* [ %call55, %if.then.54 ], [ %call57, %if.else.56 ]
  %conv60 = sext i32 %exponent_base_10.2 to i64
  %call61 = call i8* @halide_int64_to_string(i8* %dst.addr.1, i8* %end, i64 %conv60, i32 2) #3
  br label %cleanup.148

if.else.62:                                       ; preds = %if.end.37
  br i1 %cmp16, label %if.then.64, label %if.end.66

if.then.64:                                       ; preds = %if.else.62
  %call65 = call i8* @halide_double_to_string(i8* %dst.addr.0, i8* %end, double 0.000000e+00, i32 0) #3
  br label %cleanup.148

if.end.66:                                        ; preds = %if.else.62
  %add68 = or i64 %and, 4503599627370496
  %sub70 = add nsw i32 %conv, -1075
  %cmp71 = icmp ult i32 %conv, 1075
  br i1 %cmp71, label %if.then.72, label %if.end.105

if.then.72:                                       ; preds = %if.end.66
  %cmp73 = icmp slt i32 %sub70, -52
  br i1 %cmp73, label %if.end.84, label %if.else.76

if.else.76:                                       ; preds = %if.then.72
  %sub77 = sub nsw i32 1075, %conv
  %sh_prom = zext i32 %sub77 to i64
  %shr78 = lshr i64 %add68, %sh_prom
  %shl81 = shl i64 %shr78, %sh_prom
  %sub82 = sub i64 %add68, %shl81
  br label %if.end.84

if.end.84:                                        ; preds = %if.else.76, %if.then.72
  %integer_part.0 = phi i64 [ %shr78, %if.else.76 ], [ 0, %if.then.72 ]
  %f.0.in = phi i64 [ %sub82, %if.else.76 ], [ %add68, %if.then.72 ]
  %f.0 = uitofp i64 %f.0.in to double
  %conv85.244 = zext i32 %sub70 to i64
  %shl86 = shl i64 %conv85.244, 52
  %add88 = add i64 %shl86, 4696837146684686336
  %6 = bitcast i64 %add88 to double
  %mul90 = fmul double %6, %f.0
  %add91 = fadd double %mul90, 5.000000e-01
  %conv92 = fptoui double %add91 to i64
  %conv93 = uitofp i64 %conv92 to double
  %and96 = and i64 %conv92, 1
  %notlhs = fcmp oeq double %conv93, %add91
  %notrhs = icmp ne i64 %and96, 0
  %not.or.cond245 = and i1 %notrhs, %notlhs
  %dec99 = sext i1 %not.or.cond245 to i64
  %fractional_part.0 = add i64 %dec99, %conv92
  %cmp101 = icmp eq i64 %fractional_part.0, 1000000
  %inc103 = zext i1 %cmp101 to i64
  %inc103.integer_part.0 = add i64 %inc103, %integer_part.0
  %.fractional_part.0 = select i1 %cmp101, i64 0, i64 %fractional_part.0
  br label %if.end.105

if.end.105:                                       ; preds = %if.end.84, %if.end.66
  %integer_part.2 = phi i64 [ %inc103.integer_part.0, %if.end.84 ], [ %add68, %if.end.66 ]
  %integer_exponent.0 = phi i32 [ 0, %if.end.84 ], [ %sub70, %if.end.66 ]
  %fractional_part.2 = phi i64 [ %.fractional_part.0, %if.end.84 ], [ 0, %if.end.66 ]
  %7 = bitcast [512 x i8]* %buf to i8*
  call void @llvm.lifetime.start.p0i8(i64 512, i8* %7) #0
  %add.ptr = getelementptr inbounds [512 x i8], [512 x i8]* %buf, i64 0, i64 512
  %add.ptr106 = getelementptr inbounds [512 x i8], [512 x i8]* %buf, i64 0, i64 480
  %call109 = call i8* @halide_int64_to_string(i8* %add.ptr106, i8* %add.ptr, i64 %integer_part.2, i32 1) #3
  %cmp110.252 = icmp sgt i32 %integer_exponent.0, 0
  br i1 %cmp110.252, label %for.cond.112.preheader, label %for.cond.cleanup

for.cond.112.preheader:                           ; preds = %if.end.138, %if.end.105
  %i.0255 = phi i32 [ %inc140, %if.end.138 ], [ 0, %if.end.105 ]
  %int_part_ptr.0253 = phi i8* [ %int_part_ptr.1, %if.end.138 ], [ %add.ptr106, %if.end.105 ]
  %int_part_ptr.02534 = ptrtoint i8* %int_part_ptr.0253 to i64
  %cmp114.249 = icmp eq i8* %call109, %int_part_ptr.0253
  br i1 %cmp114.249, label %if.end.138, label %for.body.116.preheader

for.body.116.preheader:                           ; preds = %for.cond.112.preheader
  %8 = sub i64 0, %int_part_ptr.02534
  %scevgep5 = getelementptr i8, i8* %call109, i64 %8
  %scevgep56 = ptrtoint i8* %scevgep5 to i64
  call void @llvm.set.loop.iterations.i64(i64 %scevgep56)
  br label %for.body.116

for.cond.cleanup:                                 ; preds = %if.end.138, %if.end.105
  %int_part_ptr.0.lcssa = phi i8* [ %add.ptr106, %if.end.105 ], [ %int_part_ptr.1, %if.end.138 ]
  %9 = bitcast [512 x i8]* %buf to i8*
  %call142 = call i8* @halide_string_to_string(i8* %dst.addr.0, i8* %end, i8* %int_part_ptr.0.lcssa) #3
  %call143 = call i8* @halide_string_to_string(i8* %call142, i8* %end, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.9.96, i64 0, i64 0)) #3
  %call144 = call i8* @halide_int64_to_string(i8* %call143, i8* %end, i64 %fractional_part.2, i32 6) #3
  call void @llvm.lifetime.end.p0i8(i64 512, i8* %9) #0
  br label %cleanup.148

for.cond.cleanup.115:                             ; preds = %for.body.116
  br i1 %cmp125, label %if.then.136, label %if.end.138

for.body.116:                                     ; preds = %for.body.116, %for.body.116.preheader
  %call109.pn = phi i8* [ %p.0251, %for.body.116 ], [ %call109, %for.body.116.preheader ]
  %carry.0250 = phi i32 [ %carry.1, %for.body.116 ], [ 0, %for.body.116.preheader ]
  %call109.pn2 = ptrtoint i8* %call109.pn to i64
  %p.0251 = getelementptr inbounds i8, i8* %call109.pn, i64 -1
  %scevgep3 = getelementptr i8, i8* inttoptr (i64 -1 to i8*), i64 %call109.pn2
  %10 = load i8, i8* %scevgep3, align 1, !tbaa !10
  %sub118 = add i8 %10, -48
  %conv120 = sext i8 %sub118 to i32
  %mul121 = shl nsw i32 %conv120, 1
  %add122 = or i32 %mul121, %carry.0250
  %11 = trunc i32 %add122 to i8
  %cmp125 = icmp sgt i8 %11, 9
  %sub128 = add nsw i32 %add122, 246
  %carry.1 = zext i1 %cmp125 to i32
  %new_digit.0.in = select i1 %cmp125, i32 %sub128, i32 %add122
  %add133 = add nsw i32 %new_digit.0.in, 48
  %conv134 = trunc i32 %add133 to i8
  %scevgep = getelementptr i8, i8* inttoptr (i64 -1 to i8*), i64 %call109.pn2
  store i8 %conv134, i8* %scevgep, align 1, !tbaa !10
  %12 = call i64 @llvm.loop.dec(i64 %scevgep56, i64 1)
  %dec.cmp = icmp ne i64 %12, 0
  br i1 %dec.cmp, label %for.body.116, label %for.cond.cleanup.115

if.then.136:                                      ; preds = %for.cond.cleanup.115
  %incdec.ptr137 = getelementptr inbounds i8, i8* %int_part_ptr.0253, i64 -1
  store i8 49, i8* %incdec.ptr137, align 1, !tbaa !10
  br label %if.end.138

if.end.138:                                       ; preds = %if.then.136, %for.cond.cleanup.115, %for.cond.112.preheader
  %int_part_ptr.1 = phi i8* [ %incdec.ptr137, %if.then.136 ], [ %call109, %for.cond.112.preheader ], [ %int_part_ptr.0253, %for.cond.cleanup.115 ]
  %inc140 = add nuw nsw i32 %i.0255, 1
  %exitcond = icmp eq i32 %inc140, %integer_exponent.0
  br i1 %exitcond, label %for.cond.cleanup, label %for.cond.112.preheader

cleanup.148:                                      ; preds = %for.cond.cleanup, %if.then.64, %if.end.59, %if.else.30, %if.then.28, %if.else.24, %if.then.22, %if.else.13, %if.then.11, %if.else, %if.then.6
  %retval.1 = phi i8* [ %call7, %if.then.6 ], [ %call8, %if.else ], [ %call12, %if.then.11 ], [ %call14, %if.else.13 ], [ %call23, %if.then.22 ], [ %call25, %if.else.24 ], [ %call29, %if.then.28 ], [ %call31, %if.else.30 ], [ %call65, %if.then.64 ], [ %call61, %if.end.59 ], [ %call144, %for.cond.cleanup ]
  %13 = bitcast i64* %bits to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %13) #0
  ret i8* %retval.1
}

; Function Attrs: nounwind
declare i8* @memcpy(i8*, i8* nocapture readonly, i64) #1

; Function Attrs: nounwind
declare void @llvm.set.loop.iterations.i64(i64) #0

; Function Attrs: nounwind
declare i64 @llvm.loop.dec(i64, i64) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { nounwind }

!llvm.ident = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!llvm.module.flags = !{!1, !2, !3}

!0 = !{!"clang version 3.7.0 (branches/release_37 246867) (llvm/branches/release_37 246866)"}
!1 = !{i32 2, !"halide_use_soft_float_abi", i32 0}
!2 = !{i32 2, !"halide_mcpu", !"pwr8"}
!3 = !{i32 2, !"halide_mattrs", !"+altivec,+vsx,+power8-altivec,+direct-move"}
!4 = !{!5, !5, i64 0}
!5 = !{!"double", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{!9, !9, i64 0}
!9 = !{!"long long", !6, i64 0}
!10 = !{!6, !6, i64 0}
