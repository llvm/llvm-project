; RUN: llc < %s -verify-machineinstrs
; PR13188
;
; The _Unwind_RaiseException function can return normally and via eh.return.
; This causes confusion about the function live-out registers, since the two
; different ways of returning have different return values.
;
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-freebsd9.0"

%struct._Unwind_Context = type { [18 x i8*], i8*, i8*, i8*, %struct.dwarf_eh_bases, i64, i64, i64, [18 x i8] }
%struct.dwarf_eh_bases = type { i8*, i8*, i8* }
%struct._Unwind_FrameState = type { %struct.frame_state_reg_info, i64, i64, i8*, i32, i8*, i32 (i32, i32, i64, %struct._Unwind_Exception*, %struct._Unwind_Context*)*, i64, i64, i64, i8, i8, i8, i8, i8* }
%struct.frame_state_reg_info = type { [18 x %struct.anon], %struct.frame_state_reg_info* }
%struct.anon = type { %union.anon, i32 }
%union.anon = type { i64 }
%struct._Unwind_Exception = type { i64, void (i32, %struct._Unwind_Exception*)*, i64, i64 }

@dwarf_reg_size_table = external hidden unnamed_addr global [18 x i8], align 16

declare void @abort() noreturn

declare fastcc i32 @uw_frame_state_for(%struct._Unwind_Context*, %struct._Unwind_FrameState*) uwtable

define hidden i32 @_Unwind_RaiseException(%struct._Unwind_Exception* %exc) uwtable {
entry:
  %fs.i = alloca %struct._Unwind_FrameState, align 8
  %this_context = alloca %struct._Unwind_Context, align 8
  %cur_context = alloca %struct._Unwind_Context, align 8
  %fs = alloca %struct._Unwind_FrameState, align 8
  call void @llvm.eh.unwind.init()
  %0 = call i8* @llvm.eh.dwarf.cfa(i32 0)
  %1 = call i8* @llvm.returnaddress(i32 0)
  call fastcc void @uw_init_context_1(%struct._Unwind_Context* %this_context, i8* %0, i8* %1)
  %2 = bitcast %struct._Unwind_Context* %cur_context to i8*
  %3 = bitcast %struct._Unwind_Context* %this_context to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %2, i8* %3, i64 240, i32 8, i1 false)
  %personality = getelementptr inbounds %struct._Unwind_FrameState, %struct._Unwind_FrameState* %fs, i64 0, i32 6
  %retaddr_column.i = getelementptr inbounds %struct._Unwind_FrameState, %struct._Unwind_FrameState* %fs, i64 0, i32 9
  %flags.i.i.i.i = getelementptr inbounds %struct._Unwind_Context, %struct._Unwind_Context* %cur_context, i64 0, i32 5
  %ra.i = getelementptr inbounds %struct._Unwind_Context, %struct._Unwind_Context* %cur_context, i64 0, i32 2
  %exception_class = getelementptr inbounds %struct._Unwind_Exception, %struct._Unwind_Exception* %exc, i64 0, i32 0
  br label %while.body

while.body:                                       ; preds = %uw_update_context.exit, %entry
  %call = call fastcc i32 @uw_frame_state_for(%struct._Unwind_Context* %cur_context, %struct._Unwind_FrameState* %fs)
  switch i32 %call, label %do.end21 [
    i32 5, label %do.end21.loopexit46
    i32 0, label %if.end3
  ]

if.end3:                                          ; preds = %while.body
  %4 = load i32 (i32, i32, i64, %struct._Unwind_Exception*, %struct._Unwind_Context*)*, i32 (i32, i32, i64, %struct._Unwind_Exception*, %struct._Unwind_Context*)** %personality, align 8
  %tobool = icmp eq i32 (i32, i32, i64, %struct._Unwind_Exception*, %struct._Unwind_Context*)* %4, null
  br i1 %tobool, label %if.end13, label %if.then4

if.then4:                                         ; preds = %if.end3
  %5 = load i64, i64* %exception_class, align 8
  %call6 = call i32 %4(i32 1, i32 1, i64 %5, %struct._Unwind_Exception* %exc, %struct._Unwind_Context* %cur_context)
  switch i32 %call6, label %do.end21.loopexit46 [
    i32 6, label %while.end
    i32 8, label %if.end13
  ]

if.end13:                                         ; preds = %if.then4, %if.end3
  call fastcc void @uw_update_context_1(%struct._Unwind_Context* %cur_context, %struct._Unwind_FrameState* %fs)
  %6 = load i64, i64* %retaddr_column.i, align 8
  %conv.i = trunc i64 %6 to i32
  %cmp.i.i.i = icmp slt i32 %conv.i, 18
  br i1 %cmp.i.i.i, label %cond.end.i.i.i, label %cond.true.i.i.i

cond.true.i.i.i:                                  ; preds = %if.end13
  call void @abort() noreturn
  unreachable

cond.end.i.i.i:                                   ; preds = %if.end13
  %sext.i = shl i64 %6, 32
  %idxprom.i.i.i = ashr exact i64 %sext.i, 32
  %arrayidx.i.i.i = getelementptr inbounds [18 x i8], [18 x i8]* @dwarf_reg_size_table, i64 0, i64 %idxprom.i.i.i
  %7 = load i8, i8* %arrayidx.i.i.i, align 1
  %arrayidx2.i.i.i = getelementptr inbounds %struct._Unwind_Context, %struct._Unwind_Context* %cur_context, i64 0, i32 0, i64 %idxprom.i.i.i
  %8 = load i8*, i8** %arrayidx2.i.i.i, align 8
  %9 = load i64, i64* %flags.i.i.i.i, align 8
  %and.i.i.i.i = and i64 %9, 4611686018427387904
  %tobool.i.i.i = icmp eq i64 %and.i.i.i.i, 0
  br i1 %tobool.i.i.i, label %if.end.i.i.i, label %land.lhs.true.i.i.i

land.lhs.true.i.i.i:                              ; preds = %cond.end.i.i.i
  %arrayidx4.i.i.i = getelementptr inbounds %struct._Unwind_Context, %struct._Unwind_Context* %cur_context, i64 0, i32 8, i64 %idxprom.i.i.i
  %10 = load i8, i8* %arrayidx4.i.i.i, align 1
  %tobool6.i.i.i = icmp eq i8 %10, 0
  br i1 %tobool6.i.i.i, label %if.end.i.i.i, label %if.then.i.i.i

if.then.i.i.i:                                    ; preds = %land.lhs.true.i.i.i
  %11 = ptrtoint i8* %8 to i64
  br label %uw_update_context.exit

if.end.i.i.i:                                     ; preds = %land.lhs.true.i.i.i, %cond.end.i.i.i
  %cmp8.i.i.i = icmp eq i8 %7, 8
  br i1 %cmp8.i.i.i, label %if.then10.i.i.i, label %cond.true14.i.i.i

if.then10.i.i.i:                                  ; preds = %if.end.i.i.i
  %12 = bitcast i8* %8 to i64*
  %13 = load i64, i64* %12, align 8
  br label %uw_update_context.exit

cond.true14.i.i.i:                                ; preds = %if.end.i.i.i
  call void @abort() noreturn
  unreachable

uw_update_context.exit:                           ; preds = %if.then10.i.i.i, %if.then.i.i.i
  %retval.0.i.i.i = phi i64 [ %11, %if.then.i.i.i ], [ %13, %if.then10.i.i.i ]
  %14 = inttoptr i64 %retval.0.i.i.i to i8*
  store i8* %14, i8** %ra.i, align 8
  br label %while.body

while.end:                                        ; preds = %if.then4
  %private_1 = getelementptr inbounds %struct._Unwind_Exception, %struct._Unwind_Exception* %exc, i64 0, i32 2
  store i64 0, i64* %private_1, align 8
  %15 = load i8*, i8** %ra.i, align 8
  %16 = ptrtoint i8* %15 to i64
  %private_2 = getelementptr inbounds %struct._Unwind_Exception, %struct._Unwind_Exception* %exc, i64 0, i32 3
  store i64 %16, i64* %private_2, align 8
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %2, i8* %3, i64 240, i32 8, i1 false)
  %17 = bitcast %struct._Unwind_FrameState* %fs.i to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %17)
  %personality.i = getelementptr inbounds %struct._Unwind_FrameState, %struct._Unwind_FrameState* %fs.i, i64 0, i32 6
  %retaddr_column.i22 = getelementptr inbounds %struct._Unwind_FrameState, %struct._Unwind_FrameState* %fs.i, i64 0, i32 9
  br label %while.body.i

while.body.i:                                     ; preds = %uw_update_context.exit44, %while.end
  %call.i = call fastcc i32 @uw_frame_state_for(%struct._Unwind_Context* %cur_context, %struct._Unwind_FrameState* %fs.i)
  %18 = load i8*, i8** %ra.i, align 8
  %19 = ptrtoint i8* %18 to i64
  %20 = load i64, i64* %private_2, align 8
  %cmp.i = icmp eq i64 %19, %20
  %cmp2.i = icmp eq i32 %call.i, 0
  br i1 %cmp2.i, label %if.end.i, label %do.end21

if.end.i:                                         ; preds = %while.body.i
  %21 = load i32 (i32, i32, i64, %struct._Unwind_Exception*, %struct._Unwind_Context*)*, i32 (i32, i32, i64, %struct._Unwind_Exception*, %struct._Unwind_Context*)** %personality.i, align 8
  %tobool.i = icmp eq i32 (i32, i32, i64, %struct._Unwind_Exception*, %struct._Unwind_Context*)* %21, null
  br i1 %tobool.i, label %if.end12.i, label %if.then3.i

if.then3.i:                                       ; preds = %if.end.i
  %or.i = select i1 %cmp.i, i32 6, i32 2
  %22 = load i64, i64* %exception_class, align 8
  %call5.i = call i32 %21(i32 1, i32 %or.i, i64 %22, %struct._Unwind_Exception* %exc, %struct._Unwind_Context* %cur_context)
  switch i32 %call5.i, label %do.end21 [
    i32 7, label %do.body19
    i32 8, label %if.end12.i
  ]

if.end12.i:                                       ; preds = %if.then3.i, %if.end.i
  br i1 %cmp.i, label %cond.true.i, label %cond.end.i

cond.true.i:                                      ; preds = %if.end12.i
  call void @abort() noreturn
  unreachable

cond.end.i:                                       ; preds = %if.end12.i
  call fastcc void @uw_update_context_1(%struct._Unwind_Context* %cur_context, %struct._Unwind_FrameState* %fs.i)
  %23 = load i64, i64* %retaddr_column.i22, align 8
  %conv.i23 = trunc i64 %23 to i32
  %cmp.i.i.i24 = icmp slt i32 %conv.i23, 18
  br i1 %cmp.i.i.i24, label %cond.end.i.i.i33, label %cond.true.i.i.i25

cond.true.i.i.i25:                                ; preds = %cond.end.i
  call void @abort() noreturn
  unreachable

cond.end.i.i.i33:                                 ; preds = %cond.end.i
  %sext.i26 = shl i64 %23, 32
  %idxprom.i.i.i27 = ashr exact i64 %sext.i26, 32
  %arrayidx.i.i.i28 = getelementptr inbounds [18 x i8], [18 x i8]* @dwarf_reg_size_table, i64 0, i64 %idxprom.i.i.i27
  %24 = load i8, i8* %arrayidx.i.i.i28, align 1
  %arrayidx2.i.i.i29 = getelementptr inbounds %struct._Unwind_Context, %struct._Unwind_Context* %cur_context, i64 0, i32 0, i64 %idxprom.i.i.i27
  %25 = load i8*, i8** %arrayidx2.i.i.i29, align 8
  %26 = load i64, i64* %flags.i.i.i.i, align 8
  %and.i.i.i.i31 = and i64 %26, 4611686018427387904
  %tobool.i.i.i32 = icmp eq i64 %and.i.i.i.i31, 0
  br i1 %tobool.i.i.i32, label %if.end.i.i.i39, label %land.lhs.true.i.i.i36

land.lhs.true.i.i.i36:                            ; preds = %cond.end.i.i.i33
  %arrayidx4.i.i.i34 = getelementptr inbounds %struct._Unwind_Context, %struct._Unwind_Context* %cur_context, i64 0, i32 8, i64 %idxprom.i.i.i27
  %27 = load i8, i8* %arrayidx4.i.i.i34, align 1
  %tobool6.i.i.i35 = icmp eq i8 %27, 0
  br i1 %tobool6.i.i.i35, label %if.end.i.i.i39, label %if.then.i.i.i37

if.then.i.i.i37:                                  ; preds = %land.lhs.true.i.i.i36
  %28 = ptrtoint i8* %25 to i64
  br label %uw_update_context.exit44

if.end.i.i.i39:                                   ; preds = %land.lhs.true.i.i.i36, %cond.end.i.i.i33
  %cmp8.i.i.i38 = icmp eq i8 %24, 8
  br i1 %cmp8.i.i.i38, label %if.then10.i.i.i40, label %cond.true14.i.i.i41

if.then10.i.i.i40:                                ; preds = %if.end.i.i.i39
  %29 = bitcast i8* %25 to i64*
  %30 = load i64, i64* %29, align 8
  br label %uw_update_context.exit44

cond.true14.i.i.i41:                              ; preds = %if.end.i.i.i39
  call void @abort() noreturn
  unreachable

uw_update_context.exit44:                         ; preds = %if.then10.i.i.i40, %if.then.i.i.i37
  %retval.0.i.i.i42 = phi i64 [ %28, %if.then.i.i.i37 ], [ %30, %if.then10.i.i.i40 ]
  %31 = inttoptr i64 %retval.0.i.i.i42 to i8*
  store i8* %31, i8** %ra.i, align 8
  br label %while.body.i

do.body19:                                        ; preds = %if.then3.i
  call void @llvm.lifetime.end(i64 -1, i8* %17)
  %call20 = call fastcc i64 @uw_install_context_1(%struct._Unwind_Context* %this_context, %struct._Unwind_Context* %cur_context)
  %32 = load i8*, i8** %ra.i, align 8
  call void @llvm.eh.return.i64(i64 %call20, i8* %32)
  unreachable

do.end21.loopexit46:                              ; preds = %if.then4, %while.body
  %retval.0.ph = phi i32 [ 3, %if.then4 ], [ 5, %while.body ]
  br label %do.end21

do.end21:                                         ; preds = %do.end21.loopexit46, %if.then3.i, %while.body.i, %while.body
  %retval.0 = phi i32 [ %retval.0.ph, %do.end21.loopexit46 ], [ 3, %while.body ], [ 2, %while.body.i ], [ 2, %if.then3.i ]
  ret i32 %retval.0
}

declare void @llvm.eh.unwind.init() nounwind

declare fastcc void @uw_init_context_1(%struct._Unwind_Context*, i8*, i8*) uwtable

declare i8* @llvm.eh.dwarf.cfa(i32) nounwind

declare i8* @llvm.returnaddress(i32) nounwind readnone

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind

declare fastcc i64 @uw_install_context_1(%struct._Unwind_Context*, %struct._Unwind_Context*) uwtable

declare void @llvm.eh.return.i64(i64, i8*) nounwind

declare fastcc void @uw_update_context_1(%struct._Unwind_Context*, %struct._Unwind_FrameState* nocapture) uwtable

declare void @llvm.lifetime.start(i64, i8* nocapture) nounwind

declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind
