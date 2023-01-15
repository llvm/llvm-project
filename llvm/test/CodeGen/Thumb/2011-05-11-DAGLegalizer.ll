; RUN: llc -mtriple=thumbv6-apple-darwin < %s
; rdar://problem/9416774
; ModuleID = 'reduced.ll'

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-ios"

%struct.MMMMMMMMMMMM = type { [4 x %struct.RRRRRRRR] }
%struct.RRRRRRRR = type { [78 x i32] }

@kkkkkk = external constant ptr
@__PRETTY_FUNCTION__._ZN12CLGll = private unnamed_addr constant [62 x i8] c"static void tttttttttttt::lllllllllllll(const MMMMMMMMMMMM &)\00"
@.str = private unnamed_addr constant [75 x i8] c"\09GGGGGGGGGGGGGGGGGGGGGGG:,BE:0x%08lx,ALM:0x%08lx,LTO:0x%08lx,CBEE:0x%08lx\0A\00"

define void @_ZN12CLGll(ptr %aidData) ssp align 2 {
entry:
  %aidData.addr = alloca ptr, align 4
  %agg.tmp = alloca %struct.RRRRRRRR, align 4
  %agg.tmp4 = alloca %struct.RRRRRRRR, align 4
  %agg.tmp10 = alloca %struct.RRRRRRRR, align 4
  %agg.tmp16 = alloca %struct.RRRRRRRR, align 4
  store ptr %aidData, ptr %aidData.addr, align 4
  br label %do.body

do.body:                                          ; preds = %entry
  %tmp = load ptr, ptr @kkkkkk, align 4
  %tmp1 = load ptr, ptr %aidData.addr
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %agg.tmp, ptr align 4 %tmp1, i32 312, i1 false)
  %tmp5 = load ptr, ptr %aidData.addr
  %arrayidx7 = getelementptr inbounds [4 x %struct.RRRRRRRR], ptr %tmp5, i32 0, i32 1
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %agg.tmp4, ptr align 4 %arrayidx7, i32 312, i1 false)
  %tmp11 = load ptr, ptr %aidData.addr
  %arrayidx13 = getelementptr inbounds [4 x %struct.RRRRRRRR], ptr %tmp11, i32 0, i32 2
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %agg.tmp10, ptr align 4 %arrayidx13, i32 312, i1 false)
  %tmp17 = load ptr, ptr %aidData.addr
  %arrayidx19 = getelementptr inbounds [4 x %struct.RRRRRRRR], ptr %tmp17, i32 0, i32 3
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %agg.tmp16, ptr align 4 %arrayidx19, i32 312, i1 false)
  call void (ptr, i32, ptr, ptr, ...) @CLLoggingLog(ptr %tmp, i32 2, ptr @__PRETTY_FUNCTION__._ZN12CLGll, ptr @.str, ptr byval(%struct.RRRRRRRR) %agg.tmp, ptr byval(%struct.RRRRRRRR) %agg.tmp4, ptr byval(%struct.RRRRRRRR) %agg.tmp10, ptr byval(%struct.RRRRRRRR) %agg.tmp16)
  br label %do.end

do.end:                                           ; preds = %do.body
  ret void
}

declare void @CLLoggingLog(ptr, i32, ptr, ptr, ...)

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind
