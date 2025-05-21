; RUN: llc -verify-machineinstrs -ppc-reduce-cr-logicals \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

; Test case is reduced from the snappy benchmark.
; Verify MachineLICM will always hoist trivially rematerializable instructions even when register pressure is high.

%"class.snappy::SnappyDecompressor" = type <{ ptr, ptr, ptr, i32, i8, [5 x i8], [6 x i8] }>
%"class.snappy::Source" = type { ptr }
%"struct.snappy::iovec" = type { ptr, i64 }
%"class.snappy::SnappyIOVecWriter" = type { ptr, i64, i64, i64, i64, i64 }

@_ZN6snappy8internalL10char_tableE = internal unnamed_addr constant [5 x i16] [i16 1, i16 2052, i16 4097, i16 8193, i16 2], align 2
@_ZN6snappy8internalL8wordmaskE = internal unnamed_addr constant [5 x i32] [i32 0, i32 255, i32 65535, i32 16777215, i32 -1], align 4

; Function Attrs: argmemonly nounwind
declare void @llvm.memmove.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1) #2
; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1) #2

define linkonce_odr void @ZN6snappyDecompressor_(ptr %this, ptr %writer) {
; CHECK-LABEL: ZN6snappyDecompressor_:
; CHECK:       # %bb.0: # %entry
; CHECK:       addis 4, 2, .L_MergedGlobals@toc@ha
; CHECK:       addi 26, 4, .L_MergedGlobals@toc@l
; CHECK:       .LBB0_2: # %for.cond
; CHECK-NOT:   addis {{[0-9]+}}, 2, .L_MergedGlobals@toc@ha
; CHECK:       bctrl
entry:
  %ip_limit_ = getelementptr inbounds %"class.snappy::SnappyDecompressor", ptr %this, i64 0, i32 2
  %curr_iov_index_.i = getelementptr inbounds %"class.snappy::SnappyIOVecWriter", ptr %writer, i64 0, i32 2
  %curr_iov_written_.i = getelementptr inbounds %"class.snappy::SnappyIOVecWriter", ptr %writer, i64 0, i32 3
  br label %for.cond

for.cond:                                         ; preds = %if.end82, %if.then56, %if.end49, %entry
  %ip.0 = phi ptr [ null, %entry ], [ %add.ptr50, %if.end49 ], [ null, %if.then56 ], [ undef, %if.end82 ]
  %incdec.ptr = getelementptr inbounds i8, ptr %ip.0, i64 1
  %0 = load i8, ptr %ip.0, align 1
  %conv = zext i8 %0 to i32
  br i1 undef, label %if.then7, label %if.else

if.then7:                                         ; preds = %for.cond
  %1 = lshr i32 %conv, 2
  %add = add nuw nsw i32 %1, 1
  %conv9 = zext i32 %add to i64
  %2 = load i64, ptr %ip_limit_, align 8
  %sub.ptr.sub13 = sub i64 %2, 0
  %3 = load i64, ptr undef, align 8
  %4 = load i64, ptr null, align 8
  %sub.i = sub i64 %3, %4
  %cmp.i = icmp ult i32 %add, 17
  %cmp2.i = icmp ugt i64 %sub.ptr.sub13, 20
  %or.cond.i = and i1 %cmp.i, %cmp2.i
  %cmp4.i = icmp ugt i64 %sub.i, 15
  %or.cond13.i = and i1 %or.cond.i, %cmp4.i
  br i1 %or.cond13.i, label %land.lhs.true5.i, label %if.end17

land.lhs.true5.i:                                 ; preds = %if.then7
  %5 = load ptr, ptr undef, align 8
  %6 = load i64, ptr %curr_iov_index_.i, align 8
  %7 = load i64, ptr %curr_iov_written_.i, align 8
  %sub6.i = sub i64 0, %7
  %cmp7.i = icmp ugt i64 %sub6.i, 15
  br i1 %cmp7.i, label %cleanup102, label %if.end17

if.end17:                                         ; preds = %land.lhs.true5.i, %if.then7
  %sub = add nsw i64 %conv9, -60
  %8 = load i32, ptr undef, align 4
  %arrayidx = getelementptr inbounds [5 x i32], ptr @_ZN6snappy8internalL8wordmaskE, i64 0, i64 %sub
  %9 = load i32, ptr %arrayidx, align 4
  %and21 = and i32 %9, %8
  %add22 = add i32 %and21, 1
  %conv23 = zext i32 %add22 to i64
  %add.ptr24 = getelementptr inbounds i8, ptr %incdec.ptr, i64 %sub
  br label %if.end25

if.end25:                                         ; preds = %if.end17
  %sub.ptr.rhs.cast28 = ptrtoint ptr %add.ptr24 to i64
  %cmp30233 = icmp ugt i64 %conv23, 0
  br i1 %cmp30233, label %while.body.preheader, label %while.end

while.body.preheader:                             ; preds = %if.end25
  %add.i158256 = add i64 %4, 0
  %cmp.i160257 = icmp ugt i64 %add.i158256, %3
  br i1 %cmp.i160257, label %cleanup105, label %while.cond.preheader.i

while.cond.preheader.i:                           ; preds = %while.body.preheader
  %call39 = call ptr undef(ptr undef, ptr nonnull undef)
  unreachable

while.end:                                        ; preds = %if.end25
  br label %while.cond.preheader.i176

while.cond.preheader.i176:                        ; preds = %while.end
  br i1 undef, label %if.end49, label %while.body.lr.ph.i182

while.body.lr.ph.i182:                            ; preds = %while.cond.preheader.i176
  %.pre.i181 = load i64, ptr %curr_iov_written_.i, align 8
  %10 = load ptr, ptr undef, align 8
  %11 = load i64, ptr %curr_iov_index_.i, align 8
  %iov_len.i185 = getelementptr inbounds %"struct.snappy::iovec", ptr %10, i64 %11, i32 1
  %12 = load i64, ptr %iov_len.i185, align 8
  br label %cond.end.i190

cond.end.i190:                                    ; preds = %while.body.lr.ph.i182
  br i1 undef, label %if.end18.i207, label %if.then10.i193

if.then10.i193:                                   ; preds = %cond.end.i190
  %add12.i191 = add i64 %11, 1
  %iov_len22.phi.trans.insert.i194 = getelementptr inbounds %"struct.snappy::iovec", ptr %10, i64 %add12.i191, i32 1
  %.pre48.i195 = load i64, ptr %iov_len22.phi.trans.insert.i194, align 8
  br label %if.end18.i207

if.end18.i207:                                    ; preds = %if.then10.i193, %cond.end.i190
  %13 = phi i64 [ %.pre.i181, %cond.end.i190 ], [ 0, %if.then10.i193 ]
  %14 = phi i64 [ %12, %cond.end.i190 ], [ %.pre48.i195, %if.then10.i193 ]
  %15 = phi i64 [ %11, %cond.end.i190 ], [ %add12.i191, %if.then10.i193 ]
  %sub.i197 = sub i64 %14, %13
  %cmp.i.i198 = icmp ult i64 %sub.i197, %conv23
  %.sroa.speculated.i199 = select i1 %cmp.i.i198, i64 %sub.i197, i64 %conv23
  %iov_base.i.i200 = getelementptr inbounds %"struct.snappy::iovec", ptr %10, i64 %15, i32 0
  %16 = load ptr, ptr %iov_base.i.i200, align 8
  %add.ptr.i.i201 = getelementptr inbounds i8, ptr %16, i64 %13
  call void @llvm.memcpy.p0.p0.i64(ptr %add.ptr.i.i201, ptr %add.ptr24, i64 %.sroa.speculated.i199, i1 false) #12
  %add30.i203 = add i64 0, %.sroa.speculated.i199
  store i64 %add30.i203, ptr null, align 8
  %.pre245 = load i64, ptr %ip_limit_, align 8
  br label %if.end49

if.end49:                                         ; preds = %if.end18.i207, %while.cond.preheader.i176
  %17 = phi i64 [ %.pre245, %if.end18.i207 ], [ %2, %while.cond.preheader.i176 ]
  %add.ptr50 = getelementptr inbounds i8, ptr %add.ptr24, i64 %conv23
  %sub.ptr.sub54 = sub i64 %17, 0
  %cmp55 = icmp slt i64 %sub.ptr.sub54, 5
  br i1 %cmp55, label %if.then56, label %for.cond

if.then56:                                        ; preds = %if.end49
  br label %for.cond

if.else:                                          ; preds = %for.cond
  %idxprom = zext i8 %0 to i64
  %arrayidx68 = getelementptr inbounds [5 x i16], ptr @_ZN6snappy8internalL10char_tableE, i64 0, i64 %idxprom
  %18 = load i16, ptr %arrayidx68, align 2
  %conv69 = zext i16 %18 to i64
  %19 = load i32, ptr undef, align 4
  %shr71 = lshr i64 %conv69, 11
  %arrayidx72 = getelementptr inbounds [5 x i32], ptr @_ZN6snappy8internalL8wordmaskE, i64 0, i64 %shr71
  %20 = load i32, ptr %arrayidx72, align 4
  %and73 = and i32 %20, %19
  %conv74 = zext i32 %and73 to i64
  %add79 = add nuw nsw i64 0, %conv74
  %call80 = call zeroext i1 @_ZN6snappy17SnappyIOVecWriterAppendFromSelfEmm(ptr %writer, i64 %add79, i64 undef)
  br i1 %call80, label %if.end82, label %cleanup105

if.end82:                                         ; preds = %if.else
  br label %for.cond

cleanup102:                                       ; preds = %land.lhs.true5.i
  %iov_base.i.i = getelementptr inbounds %"struct.snappy::iovec", ptr %5, i64 %6, i32 0
  %21 = load ptr, ptr %iov_base.i.i, align 8
  %add.ptr.i.i = getelementptr inbounds i8, ptr %21, i64 %7
  call void @llvm.memmove.p0.p0.i64(ptr %add.ptr.i.i, ptr %incdec.ptr, i64 16, i1 false) #12
  %22 = load <2 x i64>, ptr %curr_iov_written_.i, align 8
  %23 = insertelement <2 x i64> undef, i64 %conv9, i32 0
  %24 = shufflevector <2 x i64> %23, <2 x i64> undef, <2 x i32> zeroinitializer
  %25 = add <2 x i64> %22, %24
  store <2 x i64> %25, ptr undef, align 8
  unreachable

cleanup105:                                       ; preds = %if.else, %while.body.preheader
  ret void
}

; Function Attrs: inlinehint
declare zeroext i1 @_ZN6snappy17SnappyIOVecWriterAppendFromSelfEmm(ptr, i64, i64) local_unnamed_addr #10 align 2
