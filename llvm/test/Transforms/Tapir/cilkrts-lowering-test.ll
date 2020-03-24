; RUN: opt < %s -tapir2target -tapir-target=cilk -S

; ModuleID = '/data/compilers/tapir/cilkrts/runtime/cilk-abi-cilk-for.cpp'
source_filename = "/data/compilers/tapir/cilkrts/runtime/cilk-abi-cilk-for.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.global_state_t = type { i16, i16, i32, i32, i32, i32, i32, %struct.global_sysdep_state*, %struct.__cilkrts_worker**, i64, i32, i32, i32, i32, i32, %struct.statistics, i32, i64, %struct.__cilkrts_frame_cache, %struct.cilk_fiber_pool, i32, i8*, i32, [64 x i8], i32, void (%struct.__cilkrts_worker*)*, [64 x i8], i32, i32 }
%struct.global_sysdep_state = type opaque
%struct.__cilkrts_worker = type { %struct.__cilkrts_stack_frame**, %struct.__cilkrts_stack_frame**, %struct.__cilkrts_stack_frame**, %struct.__cilkrts_stack_frame**, %struct.__cilkrts_stack_frame**, i32, %struct.global_state_t*, %struct.local_state*, %struct.cilkred_map*, %struct.__cilkrts_stack_frame*, i8*, %struct.__cilkrts_worker_sysdep_state*, %struct.__cilkrts_pedigree }
%struct.local_state = type opaque
%struct.cilkred_map = type opaque
%struct.__cilkrts_stack_frame = type { i32, i32, %struct.__cilkrts_stack_frame*, %struct.__cilkrts_worker*, i8*, [5 x i8*], i32, i16, i16, %union.anon }
%union.anon = type { %struct.__cilkrts_pedigree }
%struct.__cilkrts_worker_sysdep_state = type opaque
%struct.__cilkrts_pedigree = type { i64, %struct.__cilkrts_pedigree* }
%struct.statistics = type { [36 x i64], [36 x i64], [36 x i64], i64 }
%struct.__cilkrts_frame_cache = type { %struct.mutex, %struct.pool_cons*, i8*, i8*, [6 x %struct.free_list*], i64, i64, i32, i64, i64, i64 }
%struct.mutex = type { i32, %struct.__cilkrts_worker* }
%struct.pool_cons = type opaque
%struct.free_list = type { %struct.free_list* }
%struct.cilk_fiber_pool = type { %struct.spin_mutex*, i64, %struct.cilk_fiber_pool*, %struct.cilk_fiber**, i32, i32, i32, i32, i32 }
%struct.spin_mutex = type { i32, [15 x i8] }
%struct.cilk_fiber = type <{ %struct.cilk_fiber_data, void (%struct.cilk_fiber*)*, void (%struct.cilk_fiber*)*, %struct.cilk_fiber*, %struct.cilk_fiber_pool*, i32, [4 x i8] }>
%struct.cilk_fiber_data = type { i64, %struct.__cilkrts_worker*, %struct.__cilkrts_stack_frame*, i32 (i32, i8*)*, i8*, i8* }

@cilkg_singleton_ptr = external local_unnamed_addr global %struct.global_state_t*, align 8

; Function Attrs: uwtable
define protected void @__cilkrts_cilk_for_32(void (i8*, i32, i32)* nocapture %body, i8* %data, i32 %count, i32 %grain) local_unnamed_addr #0 {
entry:
  %loop_root_pedigree.i = alloca %struct.__cilkrts_pedigree, align 8
  %cmp = icmp eq i32 %count, 0
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %call.i = tail call %struct.__cilkrts_worker* @__cilkrts_get_tls_worker() #3
  %current_stack_frame.i = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %call.i, i64 0, i32 9
  %0 = load %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %current_stack_frame.i, align 8, !tbaa !3
  %pedigree.i = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %call.i, i64 0, i32 12
  %rank.i = getelementptr inbounds %struct.__cilkrts_pedigree, %struct.__cilkrts_pedigree* %pedigree.i, i64 0, i32 0
  %1 = load i64, i64* %rank.i, align 8, !tbaa !11
  %dec.i = add i64 %1, -1
  store i64 %dec.i, i64* %rank.i, align 8, !tbaa !11
  %2 = bitcast %struct.__cilkrts_pedigree* %loop_root_pedigree.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %2) #3
  %3 = bitcast %struct.__cilkrts_pedigree* %pedigree.i to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull %2, i8* %3, i64 16, i32 8, i1 false), !tbaa.struct !12
  %cmp.i.i = icmp sgt i32 %grain, 0
  br i1 %cmp.i.i, label %_ZL13cilk_for_rootIjPFvPvjjEEvT0_S0_T_i.exit, label %if.end3.i.i

if.end3.i.i:                                      ; preds = %if.then
  %4 = load %struct.global_state_t*, %struct.global_state_t** @cilkg_singleton_ptr, align 8, !tbaa !14
  %under_ptool.i.i = getelementptr inbounds %struct.global_state_t, %struct.global_state_t* %4, i64 0, i32 14
  %5 = load i32, i32* %under_ptool.i.i, align 8, !tbaa !15
  %tobool.i.i = icmp eq i32 %5, 0
  br i1 %tobool.i.i, label %if.else.i.i, label %_ZL13cilk_for_rootIjPFvPvjjEEvT0_S0_T_i.exit

if.else.i.i:                                      ; preds = %if.end3.i.i
  %P.i.i = getelementptr inbounds %struct.global_state_t, %struct.global_state_t* %4, i64 0, i32 27
  %6 = load i32, i32* %P.i.i, align 8, !tbaa !23
  %mul.i.i = shl nsw i32 %6, 3
  %add.i.i = add i32 %count, -1
  %sub.i.i = add i32 %add.i.i, %mul.i.i
  %div.i.i = udiv i32 %sub.i.i, %mul.i.i
  %7 = icmp ult i32 %div.i.i, 2048
  %.div.i.i = select i1 %7, i32 %div.i.i, i32 2048
  br label %_ZL13cilk_for_rootIjPFvPvjjEEvT0_S0_T_i.exit

_ZL13cilk_for_rootIjPFvPvjjEEvT0_S0_T_i.exit:     ; preds = %if.then, %if.end3.i.i, %if.else.i.i
  %retval.2.i.i = phi i32 [ %grain, %if.then ], [ %.div.i.i, %if.else.i.i ], [ 1, %if.end3.i.i ]
  call fastcc void @_ZL18cilk_for_recursiveIjPFvPvjjEEvT_S3_T0_S0_iP16__cilkrts_workerP18__cilkrts_pedigree(i32 0, i32 %count, void (i8*, i32, i32)* %body, i8* %data, i32 %retval.2.i.i, %struct.__cilkrts_worker* %call.i, %struct.__cilkrts_pedigree* nonnull %loop_root_pedigree.i)
  %worker.i = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %0, i64 0, i32 3
  %8 = load %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %worker.i, align 8, !tbaa !24
  %pedigree3.i = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %8, i64 0, i32 12
  %9 = bitcast %struct.__cilkrts_pedigree* %pedigree3.i to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %9, i8* nonnull %2, i64 16, i32 8, i1 false), !tbaa.struct !12
  %rank5.i = getelementptr inbounds %struct.__cilkrts_pedigree, %struct.__cilkrts_pedigree* %pedigree3.i, i64 0, i32 0
  %10 = load i64, i64* %rank5.i, align 8, !tbaa !11
  %inc.i = add i64 %10, 1
  store i64 %inc.i, i64* %rank5.i, align 8, !tbaa !11
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %2) #3
  br label %if.end

if.end:                                           ; preds = %entry, %_ZL13cilk_for_rootIjPFvPvjjEEvT0_S0_T_i.exit
  ret void
}

; Function Attrs: uwtable
define protected void @__cilkrts_cilk_for_64(void (i8*, i64, i64)* nocapture %body, i8* %data, i64 %count, i32 %grain) local_unnamed_addr #0 {
entry:
  %loop_root_pedigree.i = alloca %struct.__cilkrts_pedigree, align 8
  %cmp = icmp eq i64 %count, 0
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %call.i = tail call %struct.__cilkrts_worker* @__cilkrts_get_tls_worker() #3
  %current_stack_frame.i = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %call.i, i64 0, i32 9
  %0 = load %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %current_stack_frame.i, align 8, !tbaa !3
  %pedigree.i = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %call.i, i64 0, i32 12
  %rank.i = getelementptr inbounds %struct.__cilkrts_pedigree, %struct.__cilkrts_pedigree* %pedigree.i, i64 0, i32 0
  %1 = load i64, i64* %rank.i, align 8, !tbaa !11
  %dec.i = add i64 %1, -1
  store i64 %dec.i, i64* %rank.i, align 8, !tbaa !11
  %2 = bitcast %struct.__cilkrts_pedigree* %loop_root_pedigree.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %2) #3
  %3 = bitcast %struct.__cilkrts_pedigree* %pedigree.i to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull %2, i8* %3, i64 16, i32 8, i1 false), !tbaa.struct !12
  %cmp.i.i = icmp sgt i32 %grain, 0
  br i1 %cmp.i.i, label %_ZL13cilk_for_rootImPFvPvmmEEvT0_S0_T_i.exit, label %if.end3.i.i

if.end3.i.i:                                      ; preds = %if.then
  %4 = load %struct.global_state_t*, %struct.global_state_t** @cilkg_singleton_ptr, align 8, !tbaa !14
  %under_ptool.i.i = getelementptr inbounds %struct.global_state_t, %struct.global_state_t* %4, i64 0, i32 14
  %5 = load i32, i32* %under_ptool.i.i, align 8, !tbaa !15
  %tobool.i.i = icmp eq i32 %5, 0
  br i1 %tobool.i.i, label %if.else.i.i, label %_ZL13cilk_for_rootImPFvPvmmEEvT0_S0_T_i.exit

if.else.i.i:                                      ; preds = %if.end3.i.i
  %P.i.i = getelementptr inbounds %struct.global_state_t, %struct.global_state_t* %4, i64 0, i32 27
  %6 = load i32, i32* %P.i.i, align 8, !tbaa !23
  %mul.i.i = shl nsw i32 %6, 3
  %conv.i.i = sext i32 %mul.i.i to i64
  %add.i.i = add i64 %count, -1
  %sub.i.i = add i64 %add.i.i, %conv.i.i
  %div.i.i = udiv i64 %sub.i.i, %conv.i.i
  %7 = icmp ult i64 %div.i.i, 2048
  %retval.020.i.i = select i1 %7, i64 %div.i.i, i64 2048
  %8 = trunc i64 %retval.020.i.i to i32
  br label %_ZL13cilk_for_rootImPFvPvmmEEvT0_S0_T_i.exit

_ZL13cilk_for_rootImPFvPvmmEEvT0_S0_T_i.exit:     ; preds = %if.then, %if.end3.i.i, %if.else.i.i
  %retval.2.i.i = phi i32 [ %grain, %if.then ], [ 1, %if.end3.i.i ], [ %8, %if.else.i.i ]
  call fastcc void @_ZL18cilk_for_recursiveImPFvPvmmEEvT_S3_T0_S0_iP16__cilkrts_workerP18__cilkrts_pedigree(i64 0, i64 %count, void (i8*, i64, i64)* %body, i8* %data, i32 %retval.2.i.i, %struct.__cilkrts_worker* %call.i, %struct.__cilkrts_pedigree* nonnull %loop_root_pedigree.i)
  %worker.i = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %0, i64 0, i32 3
  %9 = load %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %worker.i, align 8, !tbaa !24
  %pedigree3.i = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %9, i64 0, i32 12
  %10 = bitcast %struct.__cilkrts_pedigree* %pedigree3.i to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %10, i8* nonnull %2, i64 16, i32 8, i1 false), !tbaa.struct !12
  %rank5.i = getelementptr inbounds %struct.__cilkrts_pedigree, %struct.__cilkrts_pedigree* %pedigree3.i, i64 0, i32 0
  %11 = load i64, i64* %rank5.i, align 8, !tbaa !11
  %inc.i = add i64 %11, 1
  store i64 %inc.i, i64* %rank5.i, align 8, !tbaa !11
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %2) #3
  br label %if.end

if.end:                                           ; preds = %entry, %_ZL13cilk_for_rootImPFvPvmmEEvT0_S0_T_i.exit
  ret void
}

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind
declare protected %struct.__cilkrts_worker* @__cilkrts_get_tls_worker() local_unnamed_addr #2

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #1

; Function Attrs: uwtable
define internal fastcc void @_ZL18cilk_for_recursiveIjPFvPvjjEEvT_S3_T0_S0_iP16__cilkrts_workerP18__cilkrts_pedigree(i32 %low, i32 %high, void (i8*, i32, i32)* nocapture %body, i8* %data, i32 %grain, %struct.__cilkrts_worker* nocapture %w, %struct.__cilkrts_pedigree* %loop_root_pedigree) unnamed_addr #0 {
entry:
  %loop_leaf_pedigree.i = alloca %struct.__cilkrts_pedigree, align 8
  %syncreg = tail call token @llvm.syncregion.start()
  %sub16 = sub i32 %high, %low
  %cmp17 = icmp ugt i32 %sub16, %grain
  br i1 %cmp17, label %if.then.preheader, label %cleanup.cont

if.then.preheader:                                ; preds = %entry
  br label %if.then

if.then:                                          ; preds = %if.then.preheader, %cleanup
  %sub20 = phi i32 [ %sub, %cleanup ], [ %sub16, %if.then.preheader ]
  %low.addr.019 = phi i32 [ %add, %cleanup ], [ %low, %if.then.preheader ]
  %w.addr.018 = phi %struct.__cilkrts_worker* [ %1, %cleanup ], [ %w, %if.then.preheader ]
  %div = lshr i32 %sub20, 1
  %add = add i32 %div, %low.addr.019
  %current_stack_frame.i = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %w.addr.018, i64 0, i32 9
  %0 = load %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %current_stack_frame.i, align 8, !tbaa !3
  detach within %syncreg, label %det.achd, label %cleanup

det.achd:                                         ; preds = %if.then
  tail call fastcc void @_ZL18cilk_for_recursiveIjPFvPvjjEEvT_S3_T0_S0_iP16__cilkrts_workerP18__cilkrts_pedigree(i32 %low.addr.019, i32 %add, void (i8*, i32, i32)* %body, i8* %data, i32 %grain, %struct.__cilkrts_worker* nonnull %w.addr.018, %struct.__cilkrts_pedigree* %loop_root_pedigree)
  reattach within %syncreg, label %cleanup

cleanup:                                          ; preds = %if.then, %det.achd
  %worker = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %0, i64 0, i32 3
  %1 = load %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %worker, align 8, !tbaa !24
  %sub = sub i32 %high, %add
  %cmp = icmp ugt i32 %sub, %grain
  br i1 %cmp, label %if.then, label %cleanup.cont

cleanup.cont:                                     ; preds = %cleanup, %entry
  %w.addr.0.lcssa = phi %struct.__cilkrts_worker* [ %w, %entry ], [ %1, %cleanup ]
  %low.addr.0.lcssa = phi i32 [ %low, %entry ], [ %add, %cleanup ]
  %current_stack_frame.i14 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %w.addr.0.lcssa, i64 0, i32 9
  %2 = load %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %current_stack_frame.i14, align 8, !tbaa !3
  %parent.i = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %w.addr.0.lcssa, i64 0, i32 12, i32 1
  %3 = bitcast %struct.__cilkrts_pedigree** %parent.i to i64*
  %4 = load i64, i64* %3, align 8, !tbaa !26
  %5 = bitcast %struct.__cilkrts_pedigree* %loop_leaf_pedigree.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #3
  %conv.i = zext i32 %low.addr.0.lcssa to i64
  %rank.i = getelementptr inbounds %struct.__cilkrts_pedigree, %struct.__cilkrts_pedigree* %loop_leaf_pedigree.i, i64 0, i32 0
  store i64 %conv.i, i64* %rank.i, align 8, !tbaa !27
  %parent1.i = getelementptr inbounds %struct.__cilkrts_pedigree, %struct.__cilkrts_pedigree* %loop_leaf_pedigree.i, i64 0, i32 1
  store %struct.__cilkrts_pedigree* %loop_root_pedigree, %struct.__cilkrts_pedigree** %parent1.i, align 8, !tbaa !28
  %rank3.i = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %w.addr.0.lcssa, i64 0, i32 12, i32 0
  store i64 0, i64* %rank3.i, align 8, !tbaa !11
  store %struct.__cilkrts_pedigree* %loop_leaf_pedigree.i, %struct.__cilkrts_pedigree** %parent.i, align 8, !tbaa !26
  call void %body(i8* %data, i32 %low.addr.0.lcssa, i32 %high)
  %worker.i = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %2, i64 0, i32 3
  %6 = load %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %worker.i, align 8, !tbaa !24
  %parent7.i = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 12, i32 1
  %7 = bitcast %struct.__cilkrts_pedigree** %parent7.i to i64*
  store i64 %4, i64* %7, align 8, !tbaa !26
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #3
  sync within %syncreg, label %preSyncL

preSyncL:                                         ; preds = %cleanup.cont
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: uwtable
define internal fastcc void @_ZL18cilk_for_recursiveImPFvPvmmEEvT_S3_T0_S0_iP16__cilkrts_workerP18__cilkrts_pedigree(i64 %low, i64 %high, void (i8*, i64, i64)* nocapture %body, i8* %data, i32 %grain, %struct.__cilkrts_worker* nocapture %w, %struct.__cilkrts_pedigree* %loop_root_pedigree) unnamed_addr #0 {
entry:
  %loop_leaf_pedigree.i = alloca %struct.__cilkrts_pedigree, align 8
  %syncreg = tail call token @llvm.syncregion.start()
  %sub16 = sub i64 %high, %low
  %conv = sext i32 %grain to i64
  %cmp17 = icmp ugt i64 %sub16, %conv
  br i1 %cmp17, label %if.then.preheader, label %cleanup.cont

if.then.preheader:                                ; preds = %entry
  br label %if.then

if.then:                                          ; preds = %if.then.preheader, %cleanup
  %sub20 = phi i64 [ %sub, %cleanup ], [ %sub16, %if.then.preheader ]
  %low.addr.019 = phi i64 [ %add, %cleanup ], [ %low, %if.then.preheader ]
  %w.addr.018 = phi %struct.__cilkrts_worker* [ %1, %cleanup ], [ %w, %if.then.preheader ]
  %div = lshr i64 %sub20, 1
  %add = add i64 %div, %low.addr.019
  %current_stack_frame.i = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %w.addr.018, i64 0, i32 9
  %0 = load %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %current_stack_frame.i, align 8, !tbaa !3
  detach within %syncreg, label %det.achd, label %cleanup

det.achd:                                         ; preds = %if.then
  tail call fastcc void @_ZL18cilk_for_recursiveImPFvPvmmEEvT_S3_T0_S0_iP16__cilkrts_workerP18__cilkrts_pedigree(i64 %low.addr.019, i64 %add, void (i8*, i64, i64)* %body, i8* %data, i32 %grain, %struct.__cilkrts_worker* nonnull %w.addr.018, %struct.__cilkrts_pedigree* %loop_root_pedigree)
  reattach within %syncreg, label %cleanup

cleanup:                                          ; preds = %if.then, %det.achd
  %worker = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %0, i64 0, i32 3
  %1 = load %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %worker, align 8, !tbaa !24
  %sub = sub i64 %high, %add
  %cmp = icmp ugt i64 %sub, %conv
  br i1 %cmp, label %if.then, label %cleanup.cont

cleanup.cont:                                     ; preds = %cleanup, %entry
  %w.addr.0.lcssa = phi %struct.__cilkrts_worker* [ %w, %entry ], [ %1, %cleanup ]
  %low.addr.0.lcssa = phi i64 [ %low, %entry ], [ %add, %cleanup ]
  %current_stack_frame.i14 = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %w.addr.0.lcssa, i64 0, i32 9
  %2 = load %struct.__cilkrts_stack_frame*, %struct.__cilkrts_stack_frame** %current_stack_frame.i14, align 8, !tbaa !3
  %parent.i = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %w.addr.0.lcssa, i64 0, i32 12, i32 1
  %3 = bitcast %struct.__cilkrts_pedigree** %parent.i to i64*
  %4 = load i64, i64* %3, align 8, !tbaa !26
  %5 = bitcast %struct.__cilkrts_pedigree* %loop_leaf_pedigree.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5) #3
  %rank.i = getelementptr inbounds %struct.__cilkrts_pedigree, %struct.__cilkrts_pedigree* %loop_leaf_pedigree.i, i64 0, i32 0
  store i64 %low.addr.0.lcssa, i64* %rank.i, align 8, !tbaa !27
  %parent1.i = getelementptr inbounds %struct.__cilkrts_pedigree, %struct.__cilkrts_pedigree* %loop_leaf_pedigree.i, i64 0, i32 1
  store %struct.__cilkrts_pedigree* %loop_root_pedigree, %struct.__cilkrts_pedigree** %parent1.i, align 8, !tbaa !28
  %rank3.i = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %w.addr.0.lcssa, i64 0, i32 12, i32 0
  store i64 0, i64* %rank3.i, align 8, !tbaa !11
  store %struct.__cilkrts_pedigree* %loop_leaf_pedigree.i, %struct.__cilkrts_pedigree** %parent.i, align 8, !tbaa !26
  call void %body(i8* %data, i64 %low.addr.0.lcssa, i64 %high)
  %worker.i = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %2, i64 0, i32 3
  %6 = load %struct.__cilkrts_worker*, %struct.__cilkrts_worker** %worker.i, align 8, !tbaa !24
  %parent7.i = getelementptr inbounds %struct.__cilkrts_worker, %struct.__cilkrts_worker* %6, i64 0, i32 12, i32 1
  %7 = bitcast %struct.__cilkrts_pedigree** %parent7.i to i64*
  store i64 %4, i64* %7, align 8, !tbaa !26
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5) #3
  sync within %syncreg, label %preSyncL

preSyncL:                                         ; preds = %cleanup.cont
  ret void
}

attributes #0 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{!"clang version 5.0.0 (git@github.com:wsmoses/Cilk-Clang.git 245c29d5cb99796c4107fd83f9bbe668c130b275) (git@github.com:wsmoses/Tapir-LLVM.git 4c29285ec4342f4eaba95987b05f84f964af9008)"}
!3 = !{!4, !5, i64 72}
!4 = !{!"_ZTS16__cilkrts_worker", !5, i64 0, !5, i64 8, !5, i64 16, !5, i64 24, !5, i64 32, !8, i64 40, !5, i64 48, !5, i64 56, !5, i64 64, !5, i64 72, !5, i64 80, !5, i64 88, !9, i64 96}
!5 = !{!"any pointer", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
!8 = !{!"int", !6, i64 0}
!9 = !{!"_ZTS18__cilkrts_pedigree", !10, i64 0, !5, i64 8}
!10 = !{!"long", !6, i64 0}
!11 = !{!4, !10, i64 96}
!12 = !{i64 0, i64 8, !13, i64 8, i64 8, !14}
!13 = !{!10, !10, i64 0}
!14 = !{!5, !5, i64 0}
!15 = !{!16, !8, i64 64}
!16 = !{!"_ZTS14global_state_t", !17, i64 0, !17, i64 2, !8, i64 4, !8, i64 8, !8, i64 12, !8, i64 16, !8, i64 20, !5, i64 24, !5, i64 32, !10, i64 40, !8, i64 48, !8, i64 52, !8, i64 56, !8, i64 60, !8, i64 64, !18, i64 72, !8, i64 944, !10, i64 952, !19, i64 960, !21, i64 1096, !8, i64 1152, !5, i64 1160, !22, i64 1168, !6, i64 1172, !8, i64 1236, !5, i64 1240, !6, i64 1248, !8, i64 1312, !8, i64 1316}
!17 = !{!"short", !6, i64 0}
!18 = !{!"_ZTS10statistics", !6, i64 0, !6, i64 288, !6, i64 576, !10, i64 864}
!19 = !{!"_ZTS21__cilkrts_frame_cache", !20, i64 0, !5, i64 16, !5, i64 24, !5, i64 32, !6, i64 40, !10, i64 88, !10, i64 96, !8, i64 104, !10, i64 112, !10, i64 120, !10, i64 128}
!20 = !{!"_ZTS5mutex", !8, i64 0, !5, i64 8}
!21 = !{!"_ZTS15cilk_fiber_pool", !5, i64 0, !10, i64 8, !5, i64 16, !5, i64 24, !8, i64 32, !8, i64 36, !8, i64 40, !8, i64 44, !8, i64 48}
!22 = !{!"_ZTS15record_replay_t", !6, i64 0}
!23 = !{!16, !8, i64 1312}
!24 = !{!25, !5, i64 16}
!25 = !{!"_ZTS21__cilkrts_stack_frame", !8, i64 0, !8, i64 4, !5, i64 8, !5, i64 16, !5, i64 24, !6, i64 32, !8, i64 72, !17, i64 76, !17, i64 78, !6, i64 80}
!26 = !{!4, !5, i64 104}
!27 = !{!9, !10, i64 0}
!28 = !{!9, !5, i64 8}
