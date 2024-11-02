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
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #0

; Function Attrs: nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #0

; Function Attrs: nounwind
declare ptr @halide_string_to_string(ptr, ptr, ptr) #1

; Function Attrs: nounwind
declare ptr @halide_int64_to_string(ptr, ptr, i64, i32) #1

; Function Attrs: nounwind
define weak ptr @halide_double_to_string(ptr %dst, ptr %end, double %arg, i32 %scientific) #1 {
entry:
  %arg.addr = alloca double, align 8
  %bits = alloca i64, align 8
  %buf = alloca [512 x i8], align 1
  store double %arg, ptr %arg.addr, align 8, !tbaa !4
  call void @llvm.lifetime.start.p0(i64 8, ptr %bits) #0
  store i64 0, ptr %bits, align 8, !tbaa !8
  %call = call ptr @memcpy(ptr %bits, ptr %arg.addr, i64 8) #2
  %0 = load i64, ptr %bits, align 8, !tbaa !8
  %and = and i64 %0, 4503599627370495
  %shr = lshr i64 %0, 52
  %shr.tr = trunc i64 %shr to i32
  %conv = and i32 %shr.tr, 2047
  %shr2 = lshr i64 %0, 63
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
  %call7 = call ptr @halide_string_to_string(ptr %dst, ptr %end, ptr @.str.87) #3
  br label %cleanup.148

if.else:                                          ; preds = %if.then.4
  %call8 = call ptr @halide_string_to_string(ptr %dst, ptr %end, ptr @.str.1.88) #3
  br label %cleanup.148

if.else.9:                                        ; preds = %if.then
  br i1 %tobool5, label %if.then.11, label %if.else.13

if.then.11:                                       ; preds = %if.else.9
  %call12 = call ptr @halide_string_to_string(ptr %dst, ptr %end, ptr @.str.2.89) #3
  br label %cleanup.148

if.else.13:                                       ; preds = %if.else.9
  %call14 = call ptr @halide_string_to_string(ptr %dst, ptr %end, ptr @.str.3.90) #3
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
  %call23 = call ptr @halide_string_to_string(ptr %dst, ptr %end, ptr @.str.4.91) #3
  br label %cleanup.148

if.else.24:                                       ; preds = %if.then.20
  %call25 = call ptr @halide_string_to_string(ptr %dst, ptr %end, ptr @.str.5.92) #3
  br label %cleanup.148

if.else.26:                                       ; preds = %if.then.18
  br i1 %tobool21, label %if.then.28, label %if.else.30

if.then.28:                                       ; preds = %if.else.26
  %call29 = call ptr @halide_string_to_string(ptr %dst, ptr %end, ptr @.str.6.93) #3
  br label %cleanup.148

if.else.30:                                       ; preds = %if.else.26
  %call31 = call ptr @halide_string_to_string(ptr %dst, ptr %end, ptr @.str.7.94) #3
  br label %cleanup.148

if.end.32:                                        ; preds = %if.else.15
  %tobool33 = icmp eq i32 %conv3, 0
  br i1 %tobool33, label %if.end.37, label %if.then.34

if.then.34:                                       ; preds = %if.end.32
  %call35 = call ptr @halide_string_to_string(ptr %dst, ptr %end, ptr @.str.8.95) #3
  %sub36 = fsub double -0.000000e+00, %arg
  store double %sub36, ptr %arg.addr, align 8, !tbaa !4
  br label %if.end.37

if.end.37:                                        ; preds = %if.then.34, %if.end.32
  %.pr = phi double [ %sub36, %if.then.34 ], [ %arg, %if.end.32 ]
  %dst.addr.0 = phi ptr [ %call35, %if.then.34 ], [ %dst, %if.end.32 ]
  %tobool38 = icmp eq i32 %scientific, 0
  br i1 %tobool38, label %if.else.62, label %while.condthread-pre-split

while.condthread-pre-split:                       ; preds = %if.end.37
  %cmp40.261 = fcmp olt double %.pr, 1.000000e+00
  br i1 %cmp40.261, label %while.body, label %while.cond.41thread-pre-split

while.body:                                       ; preds = %while.body, %while.condthread-pre-split
  %exponent_base_10.0262 = phi i32 [ %dec, %while.body ], [ 0, %while.condthread-pre-split ]
  %1 = phi double [ %mul, %while.body ], [ %.pr, %while.condthread-pre-split ]
  %mul = fmul double %1, 1.000000e+01
  %dec = add nsw i32 %exponent_base_10.0262, -1
  %cmp40 = fcmp olt double %mul, 1.000000e+00
  br i1 %cmp40, label %while.body, label %while.cond.while.cond.41thread-pre-split_crit_edge

while.cond.while.cond.41thread-pre-split_crit_edge: ; preds = %while.body
  store double %mul, ptr %arg.addr, align 8, !tbaa !4
  br label %while.cond.41thread-pre-split

while.cond.41thread-pre-split:                    ; preds = %while.cond.while.cond.41thread-pre-split_crit_edge, %while.condthread-pre-split
  %.pr246 = phi double [ %mul, %while.cond.while.cond.41thread-pre-split_crit_edge ], [ %.pr, %while.condthread-pre-split ]
  %exponent_base_10.0.lcssa = phi i32 [ %dec, %while.cond.while.cond.41thread-pre-split_crit_edge ], [ 0, %while.condthread-pre-split ]
  %cmp42.257 = fcmp ult double %.pr246, 1.000000e+01
  br i1 %cmp42.257, label %while.end.44, label %while.body.43

while.body.43:                                    ; preds = %while.body.43, %while.cond.41thread-pre-split
  %exponent_base_10.1258 = phi i32 [ %inc, %while.body.43 ], [ %exponent_base_10.0.lcssa, %while.cond.41thread-pre-split ]
  %2 = phi double [ %div, %while.body.43 ], [ %.pr246, %while.cond.41thread-pre-split ]
  %div = fdiv double %2, 1.000000e+01
  %inc = add nsw i32 %exponent_base_10.1258, 1
  %cmp42 = fcmp ult double %div, 1.000000e+01
  br i1 %cmp42, label %while.cond.41.while.end.44_crit_edge, label %while.body.43

while.cond.41.while.end.44_crit_edge:             ; preds = %while.body.43
  store double %div, ptr %arg.addr, align 8, !tbaa !4
  br label %while.end.44

while.end.44:                                     ; preds = %while.cond.41.while.end.44_crit_edge, %while.cond.41thread-pre-split
  %exponent_base_10.1.lcssa = phi i32 [ %inc, %while.cond.41.while.end.44_crit_edge ], [ %exponent_base_10.0.lcssa, %while.cond.41thread-pre-split ]
  %.lcssa = phi double [ %div, %while.cond.41.while.end.44_crit_edge ], [ %.pr246, %while.cond.41thread-pre-split ]
  %mul45 = fmul double %.lcssa, 1.000000e+06
  %add = fadd double %mul45, 5.000000e-01
  %conv46 = fptoui double %add to i64
  %div47 = udiv i64 %conv46, 1000000
  %3 = mul i64 %div47, -1000000
  %sub49 = add i64 %conv46, %3
  %call50 = call ptr @halide_int64_to_string(ptr %dst.addr.0, ptr %end, i64 %div47, i32 1) #3
  %call51 = call ptr @halide_string_to_string(ptr %call50, ptr %end, ptr @.str.9.96) #3
  %call52 = call ptr @halide_int64_to_string(ptr %call51, ptr %end, i64 %sub49, i32 6) #3
  %cmp53 = icmp sgt i32 %exponent_base_10.1.lcssa, -1
  br i1 %cmp53, label %if.then.54, label %if.else.56

if.then.54:                                       ; preds = %while.end.44
  %call55 = call ptr @halide_string_to_string(ptr %call52, ptr %end, ptr @.str.10.97) #3
  br label %if.end.59

if.else.56:                                       ; preds = %while.end.44
  %call57 = call ptr @halide_string_to_string(ptr %call52, ptr %end, ptr @.str.11.98) #3
  %sub58 = sub nsw i32 0, %exponent_base_10.1.lcssa
  br label %if.end.59

if.end.59:                                        ; preds = %if.else.56, %if.then.54
  %exponent_base_10.2 = phi i32 [ %exponent_base_10.1.lcssa, %if.then.54 ], [ %sub58, %if.else.56 ]
  %dst.addr.1 = phi ptr [ %call55, %if.then.54 ], [ %call57, %if.else.56 ]
  %conv60 = sext i32 %exponent_base_10.2 to i64
  %call61 = call ptr @halide_int64_to_string(ptr %dst.addr.1, ptr %end, i64 %conv60, i32 2) #3
  br label %cleanup.148

if.else.62:                                       ; preds = %if.end.37
  br i1 %cmp16, label %if.then.64, label %if.end.66

if.then.64:                                       ; preds = %if.else.62
  %call65 = call ptr @halide_double_to_string(ptr %dst.addr.0, ptr %end, double 0.000000e+00, i32 0) #3
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
  %4 = bitcast i64 %add88 to double
  %mul90 = fmul double %4, %f.0
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
  call void @llvm.lifetime.start.p0(i64 512, ptr %buf) #0
  %add.ptr = getelementptr inbounds [512 x i8], ptr %buf, i64 0, i64 512
  %add.ptr106 = getelementptr inbounds [512 x i8], ptr %buf, i64 0, i64 480
  %call109 = call ptr @halide_int64_to_string(ptr %add.ptr106, ptr %add.ptr, i64 %integer_part.2, i32 1) #3
  %cmp110.252 = icmp sgt i32 %integer_exponent.0, 0
  br i1 %cmp110.252, label %for.cond.112.preheader, label %for.cond.cleanup

for.cond.112.preheader:                           ; preds = %if.end.138, %if.end.105
  %i.0255 = phi i32 [ %inc140, %if.end.138 ], [ 0, %if.end.105 ]
  %int_part_ptr.0253 = phi ptr [ %int_part_ptr.1, %if.end.138 ], [ %add.ptr106, %if.end.105 ]
  %int_part_ptr.02534 = ptrtoint ptr %int_part_ptr.0253 to i64
  %cmp114.249 = icmp eq ptr %call109, %int_part_ptr.0253
  br i1 %cmp114.249, label %if.end.138, label %for.body.116.preheader

for.body.116.preheader:                           ; preds = %for.cond.112.preheader
  %5 = sub i64 0, %int_part_ptr.02534
  %scevgep5 = getelementptr i8, ptr %call109, i64 %5
  %scevgep56 = ptrtoint ptr %scevgep5 to i64
  call void @llvm.set.loop.iterations.i64(i64 %scevgep56)
  br label %for.body.116

for.cond.cleanup:                                 ; preds = %if.end.138, %if.end.105
  %int_part_ptr.0.lcssa = phi ptr [ %add.ptr106, %if.end.105 ], [ %int_part_ptr.1, %if.end.138 ]
  %call142 = call ptr @halide_string_to_string(ptr %dst.addr.0, ptr %end, ptr %int_part_ptr.0.lcssa) #3
  %call143 = call ptr @halide_string_to_string(ptr %call142, ptr %end, ptr @.str.9.96) #3
  %call144 = call ptr @halide_int64_to_string(ptr %call143, ptr %end, i64 %fractional_part.2, i32 6) #3
  call void @llvm.lifetime.end.p0(i64 512, ptr %buf) #0
  br label %cleanup.148

for.cond.cleanup.115:                             ; preds = %for.body.116
  br i1 %cmp125, label %if.then.136, label %if.end.138

for.body.116:                                     ; preds = %for.body.116, %for.body.116.preheader
  %call109.pn = phi ptr [ %p.0251, %for.body.116 ], [ %call109, %for.body.116.preheader ]
  %carry.0250 = phi i32 [ %carry.1, %for.body.116 ], [ 0, %for.body.116.preheader ]
  %call109.pn2 = ptrtoint ptr %call109.pn to i64
  %p.0251 = getelementptr inbounds i8, ptr %call109.pn, i64 -1
  %scevgep3 = getelementptr i8, ptr inttoptr (i64 -1 to ptr), i64 %call109.pn2
  %6 = load i8, ptr %scevgep3, align 1, !tbaa !10
  %sub118 = add i8 %6, -48
  %conv120 = sext i8 %sub118 to i32
  %mul121 = shl nsw i32 %conv120, 1
  %add122 = or i32 %mul121, %carry.0250
  %7 = trunc i32 %add122 to i8
  %cmp125 = icmp sgt i8 %7, 9
  %sub128 = add nsw i32 %add122, 246
  %carry.1 = zext i1 %cmp125 to i32
  %new_digit.0.in = select i1 %cmp125, i32 %sub128, i32 %add122
  %add133 = add nsw i32 %new_digit.0.in, 48
  %conv134 = trunc i32 %add133 to i8
  %scevgep = getelementptr i8, ptr inttoptr (i64 -1 to ptr), i64 %call109.pn2
  store i8 %conv134, ptr %scevgep, align 1, !tbaa !10
  %8 = call i1 @llvm.loop.decrement(i64 1)
  br i1 %8, label %for.body.116, label %for.cond.cleanup.115

if.then.136:                                      ; preds = %for.cond.cleanup.115
  %incdec.ptr137 = getelementptr inbounds i8, ptr %int_part_ptr.0253, i64 -1
  store i8 49, ptr %incdec.ptr137, align 1, !tbaa !10
  br label %if.end.138

if.end.138:                                       ; preds = %if.then.136, %for.cond.cleanup.115, %for.cond.112.preheader
  %int_part_ptr.1 = phi ptr [ %incdec.ptr137, %if.then.136 ], [ %call109, %for.cond.112.preheader ], [ %int_part_ptr.0253, %for.cond.cleanup.115 ]
  %inc140 = add nuw nsw i32 %i.0255, 1
  %exitcond = icmp eq i32 %inc140, %integer_exponent.0
  br i1 %exitcond, label %for.cond.cleanup, label %for.cond.112.preheader

cleanup.148:                                      ; preds = %for.cond.cleanup, %if.then.64, %if.end.59, %if.else.30, %if.then.28, %if.else.24, %if.then.22, %if.else.13, %if.then.11, %if.else, %if.then.6
  %retval.1 = phi ptr [ %call7, %if.then.6 ], [ %call8, %if.else ], [ %call12, %if.then.11 ], [ %call14, %if.else.13 ], [ %call23, %if.then.22 ], [ %call25, %if.else.24 ], [ %call29, %if.then.28 ], [ %call31, %if.else.30 ], [ %call65, %if.then.64 ], [ %call61, %if.end.59 ], [ %call144, %for.cond.cleanup ]
  call void @llvm.lifetime.end.p0(i64 8, ptr %bits) #0
  ret ptr %retval.1
}

; Function Attrs: nounwind
declare ptr @memcpy(ptr, ptr nocapture readonly, i64) #1

; Function Attrs: nounwind
declare void @llvm.set.loop.iterations.i64(i64) #0

; Function Attrs: nounwind
declare i1 @llvm.loop.decrement(i64) #0

attributes #0 = { nounwind }
attributes #1 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
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
