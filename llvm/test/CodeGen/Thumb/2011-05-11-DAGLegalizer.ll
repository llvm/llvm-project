; RUN: llc -mtriple=thumbv6-apple-darwin < %s
; rdar://problem/9416774
; ModuleID = 'reduced.ll'

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-ios"

%struct.MMMMMMMMMMMM = type { [4 x %struct.RRRRRRRR] }
%struct.RRRRRRRR = type { [78 x i32] }

@kkkkkk = external constant i8*
@__PRETTY_FUNCTION__._ZN12CLGll = private unnamed_addr constant [62 x i8] c"static void tttttttttttt::lllllllllllll(const MMMMMMMMMMMM &)\00"
@.str = private unnamed_addr constant [75 x i8] c"\09GGGGGGGGGGGGGGGGGGGGGGG:,BE:0x%08lx,ALM:0x%08lx,LTO:0x%08lx,CBEE:0x%08lx\0A\00"

define void @_ZN12CLGll(%struct.MMMMMMMMMMMM* %aidData) ssp align 2 {
entry:
  %aidData.addr = alloca %struct.MMMMMMMMMMMM*, align 4
  %agg.tmp = alloca %struct.RRRRRRRR, align 4
  %agg.tmp4 = alloca %struct.RRRRRRRR, align 4
  %agg.tmp10 = alloca %struct.RRRRRRRR, align 4
  %agg.tmp16 = alloca %struct.RRRRRRRR, align 4
  store %struct.MMMMMMMMMMMM* %aidData, %struct.MMMMMMMMMMMM** %aidData.addr, align 4
  br label %do.body

do.body:                                          ; preds = %entry
  %tmp = load i8** @kkkkkk, align 4
  %tmp1 = load %struct.MMMMMMMMMMMM** %aidData.addr
  %eph = getelementptr inbounds %struct.MMMMMMMMMMMM* %tmp1, i32 0, i32 0
  %arrayidx = getelementptr inbounds [4 x %struct.RRRRRRRR]* %eph, i32 0, i32 0
  %tmp2 = bitcast %struct.RRRRRRRR* %agg.tmp to i8*
  %tmp3 = bitcast %struct.RRRRRRRR* %arrayidx to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp2, i8* %tmp3, i32 312, i32 4, i1 false)
  %tmp5 = load %struct.MMMMMMMMMMMM** %aidData.addr
  %eph6 = getelementptr inbounds %struct.MMMMMMMMMMMM* %tmp5, i32 0, i32 0
  %arrayidx7 = getelementptr inbounds [4 x %struct.RRRRRRRR]* %eph6, i32 0, i32 1
  %tmp8 = bitcast %struct.RRRRRRRR* %agg.tmp4 to i8*
  %tmp9 = bitcast %struct.RRRRRRRR* %arrayidx7 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp8, i8* %tmp9, i32 312, i32 4, i1 false)
  %tmp11 = load %struct.MMMMMMMMMMMM** %aidData.addr
  %eph12 = getelementptr inbounds %struct.MMMMMMMMMMMM* %tmp11, i32 0, i32 0
  %arrayidx13 = getelementptr inbounds [4 x %struct.RRRRRRRR]* %eph12, i32 0, i32 2
  %tmp14 = bitcast %struct.RRRRRRRR* %agg.tmp10 to i8*
  %tmp15 = bitcast %struct.RRRRRRRR* %arrayidx13 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp14, i8* %tmp15, i32 312, i32 4, i1 false)
  %tmp17 = load %struct.MMMMMMMMMMMM** %aidData.addr
  %eph18 = getelementptr inbounds %struct.MMMMMMMMMMMM* %tmp17, i32 0, i32 0
  %arrayidx19 = getelementptr inbounds [4 x %struct.RRRRRRRR]* %eph18, i32 0, i32 3
  %tmp20 = bitcast %struct.RRRRRRRR* %agg.tmp16 to i8*
  %tmp21 = bitcast %struct.RRRRRRRR* %arrayidx19 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp20, i8* %tmp21, i32 312, i32 4, i1 false)
  call void (i8*, i32, i8*, i8*, ...)* @CLLoggingLog(i8* %tmp, i32 2, i8* getelementptr inbounds ([62 x i8]* @__PRETTY_FUNCTION__._ZN12CLGll, i32 0, i32 0), i8* getelementptr inbounds ([75 x i8]* @.str, i32 0, i32 0), %struct.RRRRRRRR* byval %agg.tmp, %struct.RRRRRRRR* byval %agg.tmp4, %struct.RRRRRRRR* byval %agg.tmp10, %struct.RRRRRRRR* byval %agg.tmp16)
  br label %do.end

do.end:                                           ; preds = %do.body
  ret void
}

declare void @CLLoggingLog(i8*, i32, i8*, i8*, ...)

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
