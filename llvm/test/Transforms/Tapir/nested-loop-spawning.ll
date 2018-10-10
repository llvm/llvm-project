; Test to verify that LoopSpawning properly outlines helper functions
; for nested Tapir loops.
;
; Credit to Tim Kaler for producing the source code that inspired this test
; case.
;
; RUN: opt < %s -loop-spawning-ti -simplifycfg -instcombine -S | FileCheck %s
; RUN: opt < %s -passes='loop-spawning,function(simplify-cfg,instcombine)' -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::vector.0" = type { %"struct.std::_Vector_base.1" }
%"struct.std::_Vector_base.1" = type { %"struct.std::_Vector_base<std::tuple<int, double, int>, std::allocator<std::tuple<int, double, int> > >::_Vector_impl" }
%"struct.std::_Vector_base<std::tuple<int, double, int>, std::allocator<std::tuple<int, double, int> > >::_Vector_impl" = type { %"class.std::tuple"*, %"class.std::tuple"*, %"class.std::tuple"* }
%"class.std::tuple" = type { %"struct.std::_Tuple_impl.base", [4 x i8] }
%"struct.std::_Tuple_impl.base" = type <{ %"struct.std::_Tuple_impl.5", %"struct.std::_Head_base.8" }>
%"struct.std::_Tuple_impl.5" = type { %"struct.std::_Tuple_impl.6", %"struct.std::_Head_base.7" }
%"struct.std::_Tuple_impl.6" = type { %"struct.std::_Head_base" }
%"struct.std::_Head_base" = type { i32 }
%"struct.std::_Head_base.7" = type { double }
%"struct.std::_Head_base.8" = type { i32 }
%"class.std::vector" = type { %"struct.std::_Vector_base" }
%"struct.std::_Vector_base" = type { %"struct.std::_Vector_base<params, std::allocator<params> >::_Vector_impl" }
%"struct.std::_Vector_base<params, std::allocator<params> >::_Vector_impl" = type { %struct.params*, %struct.params*, %struct.params* }
%struct.params = type { i32, i32, float, float, float, i32 }

$_ZNSt6vectorISt5tupleIJidiEESaIS1_EE17_M_realloc_insertIJS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_ = comdat any

$_ZNSt6vectorISt5tupleIJidiEESaIS1_EE17_M_realloc_insertIJRKS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_ = comdat any

@.str = private unnamed_addr constant [16 x i8] c"vector::reserve\00", align 1

; Function Attrs: nounwind uwtable
define void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE(%"class.std::vector.0"* noalias sret %agg.result, i32 %trials, double %threshold, %"class.std::vector"* nocapture readonly dereferenceable(24) %ps) local_unnamed_addr #0 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %0 = bitcast %"class.std::vector.0"* %agg.result to i8*
  tail call void @llvm.memset.p0i8.i64(i8* %0, i8 0, i64 24, i32 8, i1 false) #7
  %_M_finish.i112 = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %ps, i64 0, i32 0, i32 0, i32 1
  %1 = bitcast %struct.params** %_M_finish.i112 to i64*
  %2 = load i64, i64* %1, align 8, !tbaa !2
  %3 = bitcast %"class.std::vector"* %ps to i64*
  %4 = load i64, i64* %3, align 8, !tbaa !8
  %sub.ptr.sub.i = sub i64 %2, %4
  %sub.ptr.div.i = sdiv exact i64 %sub.ptr.sub.i, 24
  %cmp.i122 = icmp ugt i64 %sub.ptr.div.i, 768614336404564650
  br i1 %cmp.i122, label %if.then.i123, label %if.end.i

if.then.i123:                                     ; preds = %entry
  tail call void @_ZSt20__throw_length_errorPKc(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str, i64 0, i64 0)) #8
  unreachable

if.end.i:                                         ; preds = %entry
  %_M_end_of_storage.i.i = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %agg.result, i64 0, i32 0, i32 0, i32 2
  %5 = icmp eq i64 %sub.ptr.sub.i, 0
  br i1 %5, label %pfor.cond.cleanup, label %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE20_M_allocate_and_copyISt13move_iteratorIPS1_EEES6_mT_S8_.exit.i

_ZNSt6vectorISt5tupleIJidiEESaIS1_EE20_M_allocate_and_copyISt13move_iteratorIPS1_EEES6_mT_S8_.exit.i: ; preds = %if.end.i
  %_M_finish.i.i = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %agg.result, i64 0, i32 0, i32 0, i32 1
  %_M_start.i124 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %agg.result, i64 0, i32 0, i32 0, i32 0
  %call2.i.i.i.i.i = tail call i8* @_Znwm(i64 %sub.ptr.sub.i) #7
  %6 = bitcast i8* %call2.i.i.i.i.i to %"class.std::tuple"*
  %7 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_start.i124, align 8, !tbaa !9
  %tobool.i.i = icmp eq %"class.std::tuple"* %7, null
  br i1 %tobool.i.i, label %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE7reserveEm.exit, label %if.then.i.i

if.then.i.i:                                      ; preds = %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE20_M_allocate_and_copyISt13move_iteratorIPS1_EEES6_mT_S8_.exit.i
  %8 = bitcast %"class.std::tuple"* %7 to i8*
  tail call void @_ZdlPv(i8* %8) #7
  br label %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE7reserveEm.exit

_ZNSt6vectorISt5tupleIJidiEESaIS1_EE7reserveEm.exit: ; preds = %if.then.i.i, %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE20_M_allocate_and_copyISt13move_iteratorIPS1_EEES6_mT_S8_.exit.i
  %9 = bitcast %"class.std::vector.0"* %agg.result to i8**
  store i8* %call2.i.i.i.i.i, i8** %9, align 8, !tbaa !9
  %10 = bitcast %"class.std::tuple"** %_M_finish.i.i to i8**
  store i8* %call2.i.i.i.i.i, i8** %10, align 8, !tbaa !12
  %add.ptr30.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %6, i64 %sub.ptr.div.i
  store %"class.std::tuple"* %add.ptr30.i, %"class.std::tuple"** %_M_end_of_storage.i.i, align 8, !tbaa !13
  %.pre = load i64, i64* %1, align 8, !tbaa !2
  %.pre248 = load i64, i64* %3, align 8, !tbaa !8
  %.pre249 = sub i64 %.pre, %.pre248
  %.pre250 = sdiv exact i64 %.pre249, 24
  %conv = trunc i64 %.pre250 to i32
  %sext = shl i64 %.pre250, 32
  %conv2 = ashr exact i64 %sext, 32
  %cmp.i.i.i.i129 = icmp eq i64 %sext, 0
  br i1 %cmp.i.i.i.i129, label %_ZNSt6vectorIS_ISt5tupleIJidiEESaIS1_EESaIS3_EEC2EmRKS4_.exit, label %cond.true.i.i.i.i

cond.true.i.i.i.i:                                ; preds = %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE7reserveEm.exit
  %cmp.i.i.i.i.i.i = icmp ugt i64 %conv2, 768614336404564650
  br i1 %cmp.i.i.i.i.i.i, label %if.then.i.i.i.i.i.i, label %for.body.lr.ph.i.i.i.i.i130

if.then.i.i.i.i.i.i:                              ; preds = %cond.true.i.i.i.i
  tail call void @_ZSt17__throw_bad_allocv() #8
  unreachable

for.body.lr.ph.i.i.i.i.i130:                      ; preds = %cond.true.i.i.i.i
  %mul.i.i.i.i.i.i = mul nsw i64 %conv2, 24
  %call2.i.i.i.i.i.i = tail call i8* @_Znwm(i64 %mul.i.i.i.i.i.i) #7
  %11 = bitcast i8* %call2.i.i.i.i.i.i to %"class.std::vector.0"*
  %12 = ptrtoint i8* %call2.i.i.i.i.i.i to i64
  %add.ptr.i.i.i = getelementptr %"class.std::vector.0", %"class.std::vector.0"* %11, i64 %conv2
  tail call void @llvm.memset.p0i8.i64(i8* nonnull %call2.i.i.i.i.i.i, i8 0, i64 %mul.i.i.i.i.i.i, i32 8, i1 false) #7
  br label %_ZNSt6vectorIS_ISt5tupleIJidiEESaIS1_EESaIS3_EEC2EmRKS4_.exit

_ZNSt6vectorIS_ISt5tupleIJidiEESaIS1_EESaIS3_EEC2EmRKS4_.exit: ; preds = %for.body.lr.ph.i.i.i.i.i130, %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE7reserveEm.exit
  %13 = phi i64 [ %12, %for.body.lr.ph.i.i.i.i.i130 ], [ 0, %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE7reserveEm.exit ]
  %14 = phi i8* [ %call2.i.i.i.i.i.i, %for.body.lr.ph.i.i.i.i.i130 ], [ null, %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE7reserveEm.exit ]
  %cond.i.i.i.i212 = phi %"class.std::vector.0"* [ %11, %for.body.lr.ph.i.i.i.i.i130 ], [ null, %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE7reserveEm.exit ]
  %__cur.0.lcssa.i.i.i.i.i = phi %"class.std::vector.0"* [ %add.ptr.i.i.i, %for.body.lr.ph.i.i.i.i.i130 ], [ null, %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE7reserveEm.exit ]
  %15 = ptrtoint %"class.std::vector.0"* %__cur.0.lcssa.i.i.i.i.i to i64
  %cmp237 = icmp sgt i32 %conv, 0
  br i1 %cmp237, label %pfor.detach.lr.ph, label %pfor.cond.cleanup

pfor.detach.lr.ph:                                ; preds = %_ZNSt6vectorIS_ISt5tupleIJidiEESaIS1_EESaIS3_EEC2EmRKS4_.exit
  %conv6 = sext i32 %trials to i64
  %cmp.i.i.i.i131 = icmp eq i32 %trials, 0
  %add.ptr.i.i.i161219 = getelementptr i32, i32* null, i64 %conv6
  %cmp19230 = icmp sgt i32 %trials, 0
  %cmp.i.i.i.i.i.i132223 = icmp slt i32 %trials, 0
  %mul.i.i.i.i.i.i135 = shl nsw i64 %conv6, 2
  %sext251 = shl i64 %.pre250, 32
  %16 = ashr exact i64 %sext251, 32
  %wide.trip.count = zext i32 %trials to i64
  br label %pfor.detach
; CHECK: {{^pfor.detach.lr.ph:}}
; CHECK: call fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE.outline_pfor.detach.ls1(i64 0,

pfor.cond.cleanup.loopexit:                       ; preds = %pfor.inc51
  br label %pfor.cond.cleanup

pfor.cond.cleanup:                                ; preds = %pfor.cond.cleanup.loopexit, %_ZNSt6vectorIS_ISt5tupleIJidiEESaIS1_EESaIS3_EEC2EmRKS4_.exit, %if.end.i
  %17 = phi i64 [ %15, %_ZNSt6vectorIS_ISt5tupleIJidiEESaIS1_EESaIS3_EEC2EmRKS4_.exit ], [ 0, %if.end.i ], [ %15, %pfor.cond.cleanup.loopexit ]
  %__cur.0.lcssa.i.i.i.i.i261 = phi %"class.std::vector.0"* [ %__cur.0.lcssa.i.i.i.i.i, %_ZNSt6vectorIS_ISt5tupleIJidiEESaIS1_EESaIS3_EEC2EmRKS4_.exit ], [ null, %if.end.i ], [ %__cur.0.lcssa.i.i.i.i.i, %pfor.cond.cleanup.loopexit ]
  %cond.i.i.i.i212259 = phi %"class.std::vector.0"* [ %cond.i.i.i.i212, %_ZNSt6vectorIS_ISt5tupleIJidiEESaIS1_EESaIS3_EEC2EmRKS4_.exit ], [ null, %if.end.i ], [ %cond.i.i.i.i212, %pfor.cond.cleanup.loopexit ]
  %18 = phi i8* [ %14, %_ZNSt6vectorIS_ISt5tupleIJidiEESaIS1_EESaIS3_EEC2EmRKS4_.exit ], [ null, %if.end.i ], [ %14, %pfor.cond.cleanup.loopexit ]
  %19 = phi i64 [ %13, %_ZNSt6vectorIS_ISt5tupleIJidiEESaIS1_EESaIS3_EEC2EmRKS4_.exit ], [ 0, %if.end.i ], [ %13, %pfor.cond.cleanup.loopexit ]
  sync within %syncreg, label %sync.continue53

pfor.detach:                                      ; preds = %pfor.inc51, %pfor.detach.lr.ph
  %indvars.iv246 = phi i64 [ 0, %pfor.detach.lr.ph ], [ %indvars.iv.next247, %pfor.inc51 ]
  detach within %syncreg, label %pfor.body, label %pfor.inc51

pfor.body:                                        ; preds = %pfor.detach
  %worker_matches_count.sroa.11 = alloca i32*, align 8
  %syncreg10 = call token @llvm.syncregion.start()
  %ref.tmp49 = alloca %"class.std::tuple", align 8
  %call5 = call i64 @clock() #7
  br i1 %cmp.i.i.i.i131, label %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i163.thread, label %cond.true.i.i.i.i133

_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i163.thread: ; preds = %pfor.body
  %worker_matches_count.sroa.11.0..sroa_cast189215 = bitcast i32** %worker_matches_count.sroa.11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %worker_matches_count.sroa.11.0..sroa_cast189215)
  store i32* %add.ptr.i.i.i161219, i32** %worker_matches_count.sroa.11, align 8
  br label %_ZNSt6vectorIiSaIiEEC2EmRKS0_.exit166

cond.true.i.i.i.i133:                             ; preds = %pfor.body
  br i1 %cmp.i.i.i.i.i.i132223, label %if.then.i.i.i.i.i.i134, label %for.body.lr.ph.i.i.i.i.i.i.i164

if.then.i.i.i.i.i.i134:                           ; preds = %cond.true.i.i.i.i133
  call void @_ZSt17__throw_bad_allocv() #8
  unreachable

for.body.lr.ph.i.i.i.i.i.i.i164:                  ; preds = %cond.true.i.i.i.i133
  %call2.i.i.i.i.i.i136 = call i8* @_Znwm(i64 %mul.i.i.i.i.i.i135) #7
  %20 = bitcast i8* %call2.i.i.i.i.i.i136 to i32*
  call void @llvm.memset.p0i8.i64(i8* nonnull %call2.i.i.i.i.i.i136, i8 0, i64 %mul.i.i.i.i.i.i135, i32 4, i1 false) #7
  %worker_matches_count.sroa.11.0..sroa_cast189 = bitcast i32** %worker_matches_count.sroa.11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %worker_matches_count.sroa.11.0..sroa_cast189)
  %call2.i.i.i.i.i.i156 = call i8* @_Znwm(i64 %mul.i.i.i.i.i.i135) #7
  %21 = bitcast i8* %call2.i.i.i.i.i.i156 to i32*
  %22 = ptrtoint i8* %call2.i.i.i.i.i.i156 to i64
  %add.ptr.i.i.i161 = getelementptr i32, i32* %21, i64 %conv6
  store i32* %add.ptr.i.i.i161, i32** %worker_matches_count.sroa.11, align 8
  call void @llvm.memset.p0i8.i64(i8* nonnull %call2.i.i.i.i.i.i156, i8 0, i64 %mul.i.i.i.i.i.i135, i32 4, i1 false) #7
  br label %_ZNSt6vectorIiSaIiEEC2EmRKS0_.exit166

_ZNSt6vectorIiSaIiEEC2EmRKS0_.exit166:            ; preds = %for.body.lr.ph.i.i.i.i.i.i.i164, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i163.thread
  %23 = phi i64 [ %22, %for.body.lr.ph.i.i.i.i.i.i.i164 ], [ 0, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i163.thread ]
  %24 = phi i8* [ %call2.i.i.i.i.i.i156, %for.body.lr.ph.i.i.i.i.i.i.i164 ], [ null, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i163.thread ]
  %cond.i.i.i.i158222 = phi i32* [ %21, %for.body.lr.ph.i.i.i.i.i.i.i164 ], [ null, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i163.thread ]
  %25 = phi i8* [ %call2.i.i.i.i.i.i136, %for.body.lr.ph.i.i.i.i.i.i.i164 ], [ null, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i163.thread ]
  %cond.i.i.i.i137214217221 = phi i32* [ %20, %for.body.lr.ph.i.i.i.i.i.i.i164 ], [ null, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i163.thread ]
  %worker_matches_count.sroa.11.0..sroa_cast189218220 = phi i8* [ %worker_matches_count.sroa.11.0..sroa_cast189, %for.body.lr.ph.i.i.i.i.i.i.i164 ], [ %worker_matches_count.sroa.11.0..sroa_cast189215, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i163.thread ]
  %__first.addr.0.lcssa.i.i.i.i.i.i.i165 = phi i32* [ %add.ptr.i.i.i161, %for.body.lr.ph.i.i.i.i.i.i.i164 ], [ null, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i163.thread ]
  %26 = ptrtoint i32* %__first.addr.0.lcssa.i.i.i.i.i.i.i165 to i64
  br i1 %cmp19230, label %pfor.detach21.preheader, label %pfor.cond.cleanup20

pfor.detach21.preheader:                          ; preds = %_ZNSt6vectorIiSaIiEEC2EmRKS0_.exit166
  br label %pfor.detach21

pfor.cond.cleanup20.loopexit:                     ; preds = %pfor.inc
  br label %pfor.cond.cleanup20

pfor.cond.cleanup20:                              ; preds = %pfor.cond.cleanup20.loopexit, %_ZNSt6vectorIiSaIiEEC2EmRKS0_.exit166
  sync within %syncreg10, label %sync.continue

pfor.detach21:                                    ; preds = %pfor.inc, %pfor.detach21.preheader
  %indvars.iv242 = phi i64 [ %indvars.iv.next243, %pfor.inc ], [ 0, %pfor.detach21.preheader ]
  detach within %syncreg10, label %pfor.body25, label %pfor.inc

pfor.body25:                                      ; preds = %pfor.detach21
  %27 = trunc i64 %indvars.iv242 to i32
  %call26 = call i32 @_Z15get_valid_movesi(i32 %27) #7
  %add.ptr.i168 = getelementptr inbounds i32, i32* %cond.i.i.i.i137214217221, i64 %indvars.iv242
  store i32 %call26, i32* %add.ptr.i168, align 4, !tbaa !14
  %call29 = call i32 @_Z17get_matches_counti(i32 %27) #7
  %add.ptr.i174 = getelementptr inbounds i32, i32* %cond.i.i.i.i158222, i64 %indvars.iv242
  store i32 %call29, i32* %add.ptr.i174, align 4, !tbaa !14
  reattach within %syncreg10, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.body25, %pfor.detach21
  %indvars.iv.next243 = add nuw nsw i64 %indvars.iv242, 1
  %exitcond = icmp eq i64 %indvars.iv.next243, %wide.trip.count
  br i1 %exitcond, label %pfor.cond.cleanup20.loopexit, label %pfor.detach21, !llvm.loop !16

sync.continue:                                    ; preds = %pfor.cond.cleanup20
  %sub.ptr.sub.i179 = sub i64 %26, %23
  %sub.ptr.div.i180 = ashr exact i64 %sub.ptr.sub.i179, 2
  %cmp35232 = icmp eq i64 %sub.ptr.sub.i179, 0
  br i1 %cmp35232, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %sync.continue
  %28 = icmp ugt i64 %sub.ptr.div.i180, 1
  %umax = select i1 %28, i64 %sub.ptr.div.i180, i64 1
  %min.iters.check = icmp ult i64 %umax, 8
  br i1 %min.iters.check, label %for.body.preheader, label %vector.ph

for.body.preheader:                               ; preds = %middle.block, %for.body.lr.ph
  %indvars.iv244.ph = phi i64 [ 0, %for.body.lr.ph ], [ %n.vec, %middle.block ]
  %matches_count.0234.ph = phi i32 [ 0, %for.body.lr.ph ], [ %70, %middle.block ]
  %valid_moves.0233.ph = phi i32 [ 0, %for.body.lr.ph ], [ %69, %middle.block ]
  br label %for.body

vector.ph:                                        ; preds = %for.body.lr.ph
  %n.vec = and i64 %umax, -8
  %29 = add nsw i64 %n.vec, -8
  %30 = lshr exact i64 %29, 3
  %31 = add nuw nsw i64 %30, 1
  %xtraiter = and i64 %31, 1
  %32 = icmp eq i64 %29, 0
  br i1 %32, label %middle.block.unr-lcssa, label %vector.ph.new

vector.ph.new:                                    ; preds = %vector.ph
  %unroll_iter = sub nsw i64 %31, %xtraiter
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph.new
  %index = phi i64 [ 0, %vector.ph.new ], [ %index.next.1, %vector.body ]
  %vec.phi = phi <4 x i32> [ zeroinitializer, %vector.ph.new ], [ %49, %vector.body ]
  %vec.phi263 = phi <4 x i32> [ zeroinitializer, %vector.ph.new ], [ %50, %vector.body ]
  %vec.phi264 = phi <4 x i32> [ zeroinitializer, %vector.ph.new ], [ %55, %vector.body ]
  %vec.phi265 = phi <4 x i32> [ zeroinitializer, %vector.ph.new ], [ %56, %vector.body ]
  %niter = phi i64 [ %unroll_iter, %vector.ph.new ], [ %niter.nsub.1, %vector.body ]
  %33 = getelementptr inbounds i32, i32* %cond.i.i.i.i158222, i64 %index
  %34 = bitcast i32* %33 to <4 x i32>*
  %wide.load = load <4 x i32>, <4 x i32>* %34, align 4, !tbaa !14
  %35 = getelementptr i32, i32* %33, i64 4
  %36 = bitcast i32* %35 to <4 x i32>*
  %wide.load266 = load <4 x i32>, <4 x i32>* %36, align 4, !tbaa !14
  %37 = add nsw <4 x i32> %wide.load, %vec.phi
  %38 = add nsw <4 x i32> %wide.load266, %vec.phi263
  %39 = getelementptr inbounds i32, i32* %cond.i.i.i.i137214217221, i64 %index
  %40 = bitcast i32* %39 to <4 x i32>*
  %wide.load267 = load <4 x i32>, <4 x i32>* %40, align 4, !tbaa !14
  %41 = getelementptr i32, i32* %39, i64 4
  %42 = bitcast i32* %41 to <4 x i32>*
  %wide.load268 = load <4 x i32>, <4 x i32>* %42, align 4, !tbaa !14
  %43 = add nsw <4 x i32> %wide.load267, %vec.phi264
  %44 = add nsw <4 x i32> %wide.load268, %vec.phi265
  %index.next = or i64 %index, 8
  %45 = getelementptr inbounds i32, i32* %cond.i.i.i.i158222, i64 %index.next
  %46 = bitcast i32* %45 to <4 x i32>*
  %wide.load.1 = load <4 x i32>, <4 x i32>* %46, align 4, !tbaa !14
  %47 = getelementptr i32, i32* %45, i64 4
  %48 = bitcast i32* %47 to <4 x i32>*
  %wide.load266.1 = load <4 x i32>, <4 x i32>* %48, align 4, !tbaa !14
  %49 = add nsw <4 x i32> %wide.load.1, %37
  %50 = add nsw <4 x i32> %wide.load266.1, %38
  %51 = getelementptr inbounds i32, i32* %cond.i.i.i.i137214217221, i64 %index.next
  %52 = bitcast i32* %51 to <4 x i32>*
  %wide.load267.1 = load <4 x i32>, <4 x i32>* %52, align 4, !tbaa !14
  %53 = getelementptr i32, i32* %51, i64 4
  %54 = bitcast i32* %53 to <4 x i32>*
  %wide.load268.1 = load <4 x i32>, <4 x i32>* %54, align 4, !tbaa !14
  %55 = add nsw <4 x i32> %wide.load267.1, %43
  %56 = add nsw <4 x i32> %wide.load268.1, %44
  %index.next.1 = add i64 %index, 16
  %niter.nsub.1 = add i64 %niter, -2
  %niter.ncmp.1 = icmp eq i64 %niter.nsub.1, 0
  br i1 %niter.ncmp.1, label %middle.block.unr-lcssa.loopexit, label %vector.body, !llvm.loop !18

middle.block.unr-lcssa.loopexit:                  ; preds = %vector.body
  %.lcssa4 = phi <4 x i32> [ %49, %vector.body ]
  %.lcssa3 = phi <4 x i32> [ %50, %vector.body ]
  %.lcssa2 = phi <4 x i32> [ %55, %vector.body ]
  %.lcssa1 = phi <4 x i32> [ %56, %vector.body ]
  %index.next.1.lcssa = phi i64 [ %index.next.1, %vector.body ]
  br label %middle.block.unr-lcssa

middle.block.unr-lcssa:                           ; preds = %middle.block.unr-lcssa.loopexit, %vector.ph
  %.lcssa280.ph = phi <4 x i32> [ undef, %vector.ph ], [ %.lcssa4, %middle.block.unr-lcssa.loopexit ]
  %.lcssa279.ph = phi <4 x i32> [ undef, %vector.ph ], [ %.lcssa3, %middle.block.unr-lcssa.loopexit ]
  %.lcssa278.ph = phi <4 x i32> [ undef, %vector.ph ], [ %.lcssa2, %middle.block.unr-lcssa.loopexit ]
  %.lcssa.ph = phi <4 x i32> [ undef, %vector.ph ], [ %.lcssa1, %middle.block.unr-lcssa.loopexit ]
  %index.unr = phi i64 [ 0, %vector.ph ], [ %index.next.1.lcssa, %middle.block.unr-lcssa.loopexit ]
  %vec.phi.unr = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %.lcssa4, %middle.block.unr-lcssa.loopexit ]
  %vec.phi263.unr = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %.lcssa3, %middle.block.unr-lcssa.loopexit ]
  %vec.phi264.unr = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %.lcssa2, %middle.block.unr-lcssa.loopexit ]
  %vec.phi265.unr = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %.lcssa1, %middle.block.unr-lcssa.loopexit ]
  %lcmp.mod = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod, label %middle.block, label %vector.body.epil

vector.body.epil:                                 ; preds = %middle.block.unr-lcssa
  %57 = getelementptr inbounds i32, i32* %cond.i.i.i.i158222, i64 %index.unr
  %58 = getelementptr inbounds i32, i32* %cond.i.i.i.i137214217221, i64 %index.unr
  %59 = getelementptr i32, i32* %58, i64 4
  %60 = bitcast i32* %59 to <4 x i32>*
  %wide.load268.epil = load <4 x i32>, <4 x i32>* %60, align 4, !tbaa !14
  %61 = add nsw <4 x i32> %wide.load268.epil, %vec.phi265.unr
  %62 = bitcast i32* %58 to <4 x i32>*
  %wide.load267.epil = load <4 x i32>, <4 x i32>* %62, align 4, !tbaa !14
  %63 = add nsw <4 x i32> %wide.load267.epil, %vec.phi264.unr
  %64 = getelementptr i32, i32* %57, i64 4
  %65 = bitcast i32* %64 to <4 x i32>*
  %wide.load266.epil = load <4 x i32>, <4 x i32>* %65, align 4, !tbaa !14
  %66 = add nsw <4 x i32> %wide.load266.epil, %vec.phi263.unr
  %67 = bitcast i32* %57 to <4 x i32>*
  %wide.load.epil = load <4 x i32>, <4 x i32>* %67, align 4, !tbaa !14
  %68 = add nsw <4 x i32> %wide.load.epil, %vec.phi.unr
  br label %middle.block

middle.block:                                     ; preds = %vector.body.epil, %middle.block.unr-lcssa
  %.lcssa280 = phi <4 x i32> [ %.lcssa280.ph, %middle.block.unr-lcssa ], [ %68, %vector.body.epil ]
  %.lcssa279 = phi <4 x i32> [ %.lcssa279.ph, %middle.block.unr-lcssa ], [ %66, %vector.body.epil ]
  %.lcssa278 = phi <4 x i32> [ %.lcssa278.ph, %middle.block.unr-lcssa ], [ %63, %vector.body.epil ]
  %.lcssa = phi <4 x i32> [ %.lcssa.ph, %middle.block.unr-lcssa ], [ %61, %vector.body.epil ]
  %bin.rdx272 = add <4 x i32> %.lcssa, %.lcssa278
  %rdx.shuf273 = shufflevector <4 x i32> %bin.rdx272, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx274 = add <4 x i32> %bin.rdx272, %rdx.shuf273
  %rdx.shuf275 = shufflevector <4 x i32> %bin.rdx274, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx276 = add <4 x i32> %bin.rdx274, %rdx.shuf275
  %69 = extractelement <4 x i32> %bin.rdx276, i32 0
  %bin.rdx = add <4 x i32> %.lcssa279, %.lcssa280
  %rdx.shuf = shufflevector <4 x i32> %bin.rdx, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx269 = add <4 x i32> %bin.rdx, %rdx.shuf
  %rdx.shuf270 = shufflevector <4 x i32> %bin.rdx269, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx271 = add <4 x i32> %bin.rdx269, %rdx.shuf270
  %70 = extractelement <4 x i32> %bin.rdx271, i32 0
  %cmp.n = icmp eq i64 %umax, %n.vec
  br i1 %cmp.n, label %for.cond.cleanup, label %for.body.preheader

for.cond.cleanup.loopexit:                        ; preds = %for.body
  %add38.lcssa = phi i32 [ %add38, %for.body ]
  %add41.lcssa = phi i32 [ %add41, %for.body ]
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %middle.block, %sync.continue
  %valid_moves.0.lcssa = phi i32 [ 0, %sync.continue ], [ %69, %middle.block ], [ %add41.lcssa, %for.cond.cleanup.loopexit ]
  %matches_count.0.lcssa = phi i32 [ 0, %sync.continue ], [ %70, %middle.block ], [ %add38.lcssa, %for.cond.cleanup.loopexit ]
  %call43 = call i64 @clock() #7
  %sub44 = sub nsw i64 %call43, %call5
  %conv45 = sitofp i64 %sub44 to double
  %div46 = fdiv double %conv45, 1.000000e+06
  %71 = bitcast %"class.std::tuple"* %ref.tmp49 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %71) #7
  %_M_head_impl.i.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %ref.tmp49, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store i32 %matches_count.0.lcssa, i32* %_M_head_impl.i.i.i.i.i.i, align 8, !tbaa !20, !alias.scope !22
  %72 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %ref.tmp49, i64 0, i32 0, i32 0, i32 1, i32 0
  store double %div46, double* %72, align 8, !tbaa !25, !alias.scope !22
  %73 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %ref.tmp49, i64 0, i32 0, i32 1, i32 0
  store i32 %valid_moves.0.lcssa, i32* %73, align 8, !tbaa !28, !alias.scope !22
  %_M_finish.i.i175 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %cond.i.i.i.i212, i64 %indvars.iv246, i32 0, i32 0, i32 1
  %74 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_finish.i.i175, align 8, !tbaa !12
  %_M_end_of_storage.i.i176 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %cond.i.i.i.i212, i64 %indvars.iv246, i32 0, i32 0, i32 2
  %75 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_end_of_storage.i.i176, align 8, !tbaa !13
  %cmp.i.i = icmp eq %"class.std::tuple"* %74, %75
  br i1 %cmp.i.i, label %if.else.i.i, label %if.then.i.i177

if.then.i.i177:                                   ; preds = %for.cond.cleanup
  %_M_head_impl.i.i6.i.i.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %74, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store i32 %matches_count.0.lcssa, i32* %_M_head_impl.i.i6.i.i.i.i.i.i.i, align 4, !tbaa !20
  %76 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %74, i64 0, i32 0, i32 0, i32 1, i32 0
  store double %div46, double* %76, align 8, !tbaa !25
  %77 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %74, i64 0, i32 0, i32 1, i32 0
  %78 = load i32, i32* %73, align 8, !tbaa !14
  store i32 %78, i32* %77, align 4, !tbaa !28
  %incdec.ptr.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %74, i64 1
  store %"class.std::tuple"* %incdec.ptr.i.i, %"class.std::tuple"** %_M_finish.i.i175, align 8, !tbaa !12
  br label %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE9push_backEOS1_.exit

if.else.i.i:                                      ; preds = %for.cond.cleanup
  %add.ptr.i182 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %cond.i.i.i.i212, i64 %indvars.iv246
  call void @_ZNSt6vectorISt5tupleIJidiEESaIS1_EE17_M_realloc_insertIJS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_(%"class.std::vector.0"* nonnull %add.ptr.i182, %"class.std::tuple"* %74, %"class.std::tuple"* nonnull dereferenceable(24) %ref.tmp49) #7
  br label %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE9push_backEOS1_.exit

_ZNSt6vectorISt5tupleIJidiEESaIS1_EE9push_backEOS1_.exit: ; preds = %if.else.i.i, %if.then.i.i177
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %71) #7
  %tobool.i.i.i170 = icmp eq i32* %cond.i.i.i.i158222, null
  br i1 %tobool.i.i.i170, label %_ZNSt6vectorIiSaIiEED2Ev.exit172, label %if.then.i.i.i171

if.then.i.i.i171:                                 ; preds = %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE9push_backEOS1_.exit
  call void @_ZdlPv(i8* %24) #7
  br label %_ZNSt6vectorIiSaIiEED2Ev.exit172

_ZNSt6vectorIiSaIiEED2Ev.exit172:                 ; preds = %if.then.i.i.i171, %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE9push_backEOS1_.exit
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %worker_matches_count.sroa.11.0..sroa_cast189218220)
  %tobool.i.i.i149 = icmp eq i32* %cond.i.i.i.i137214217221, null
  br i1 %tobool.i.i.i149, label %_ZNSt6vectorIiSaIiEED2Ev.exit, label %if.then.i.i.i150

if.then.i.i.i150:                                 ; preds = %_ZNSt6vectorIiSaIiEED2Ev.exit172
  call void @_ZdlPv(i8* %25) #7
  br label %_ZNSt6vectorIiSaIiEED2Ev.exit

_ZNSt6vectorIiSaIiEED2Ev.exit:                    ; preds = %if.then.i.i.i150, %_ZNSt6vectorIiSaIiEED2Ev.exit172
  reattach within %syncreg, label %pfor.inc51

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv244 = phi i64 [ %indvars.iv.next245, %for.body ], [ %indvars.iv244.ph, %for.body.preheader ]
  %matches_count.0234 = phi i32 [ %add38, %for.body ], [ %matches_count.0234.ph, %for.body.preheader ]
  %valid_moves.0233 = phi i32 [ %add41, %for.body ], [ %valid_moves.0233.ph, %for.body.preheader ]
  %add.ptr.i148 = getelementptr inbounds i32, i32* %cond.i.i.i.i158222, i64 %indvars.iv244
  %79 = load i32, i32* %add.ptr.i148, align 4, !tbaa !14
  %add38 = add nsw i32 %79, %matches_count.0234
  %add.ptr.i146 = getelementptr inbounds i32, i32* %cond.i.i.i.i137214217221, i64 %indvars.iv244
  %80 = load i32, i32* %add.ptr.i146, align 4, !tbaa !14
  %add41 = add nsw i32 %80, %valid_moves.0233
  %indvars.iv.next245 = add nuw i64 %indvars.iv244, 1
  %cmp35 = icmp ugt i64 %sub.ptr.div.i180, %indvars.iv.next245
  br i1 %cmp35, label %for.body, label %for.cond.cleanup.loopexit, !llvm.loop !30

pfor.inc51:                                       ; preds = %_ZNSt6vectorIiSaIiEED2Ev.exit, %pfor.detach
  %indvars.iv.next247 = add nuw nsw i64 %indvars.iv246, 1
  %exitcond5 = icmp ne i64 %indvars.iv.next247, %16
  br i1 %exitcond5, label %pfor.detach, label %pfor.cond.cleanup.loopexit, !llvm.loop !32

sync.continue53:                                  ; preds = %pfor.cond.cleanup
  %sub.ptr.sub.i143 = sub i64 %17, %19
  %sub.ptr.div.i144 = sdiv exact i64 %sub.ptr.sub.i143, 24
  %81 = icmp eq i64 %sub.ptr.sub.i143, 0
  br i1 %81, label %for.cond.cleanup60, label %for.body61.lr.ph

for.body61.lr.ph:                                 ; preds = %sync.continue53
  %_M_finish.i = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %agg.result, i64 0, i32 0, i32 0, i32 1
  br label %for.body61

for.cond.cleanup60.loopexit:                      ; preds = %for.cond.cleanup68
  br label %for.cond.cleanup60

for.cond.cleanup60:                               ; preds = %for.cond.cleanup60.loopexit, %sync.continue53
  %cmp3.i.i.i.i = icmp eq %"class.std::vector.0"* %cond.i.i.i.i212259, %__cur.0.lcssa.i.i.i.i.i261
  br i1 %cmp3.i.i.i.i, label %_ZSt8_DestroyIPSt6vectorISt5tupleIJidiEESaIS2_EES4_EvT_S6_RSaIT0_E.exit.i, label %for.body.i.i.i.i.preheader

for.body.i.i.i.i.preheader:                       ; preds = %for.cond.cleanup60
  br label %for.body.i.i.i.i

for.body.i.i.i.i:                                 ; preds = %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i, %for.body.i.i.i.i.preheader
  %__first.addr.04.i.i.i.i = phi %"class.std::vector.0"* [ %incdec.ptr.i.i.i.i, %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i ], [ %cond.i.i.i.i212259, %for.body.i.i.i.i.preheader ]
  %_M_start.i.i.i.i.i.i.i = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %__first.addr.04.i.i.i.i, i64 0, i32 0, i32 0, i32 0
  %82 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_start.i.i.i.i.i.i.i, align 8, !tbaa !9
  %tobool.i.i.i.i.i.i.i.i = icmp eq %"class.std::tuple"* %82, null
  br i1 %tobool.i.i.i.i.i.i.i.i, label %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i, label %if.then.i.i.i.i.i.i.i.i

if.then.i.i.i.i.i.i.i.i:                          ; preds = %for.body.i.i.i.i
  %83 = bitcast %"class.std::tuple"* %82 to i8*
  call void @_ZdlPv(i8* %83) #7
  br label %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i

_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i: ; preds = %if.then.i.i.i.i.i.i.i.i, %for.body.i.i.i.i
  %incdec.ptr.i.i.i.i = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %__first.addr.04.i.i.i.i, i64 1
  %cmp.i.i.i.i = icmp eq %"class.std::vector.0"* %incdec.ptr.i.i.i.i, %__cur.0.lcssa.i.i.i.i.i261
  br i1 %cmp.i.i.i.i, label %_ZSt8_DestroyIPSt6vectorISt5tupleIJidiEESaIS2_EES4_EvT_S6_RSaIT0_E.exit.i.loopexit, label %for.body.i.i.i.i

_ZSt8_DestroyIPSt6vectorISt5tupleIJidiEESaIS2_EES4_EvT_S6_RSaIT0_E.exit.i.loopexit: ; preds = %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i
  br label %_ZSt8_DestroyIPSt6vectorISt5tupleIJidiEESaIS2_EES4_EvT_S6_RSaIT0_E.exit.i

_ZSt8_DestroyIPSt6vectorISt5tupleIJidiEESaIS2_EES4_EvT_S6_RSaIT0_E.exit.i: ; preds = %_ZSt8_DestroyIPSt6vectorISt5tupleIJidiEESaIS2_EES4_EvT_S6_RSaIT0_E.exit.i.loopexit, %for.cond.cleanup60
  %tobool.i.i.i = icmp eq %"class.std::vector.0"* %cond.i.i.i.i212259, null
  br i1 %tobool.i.i.i, label %_ZNSt6vectorIS_ISt5tupleIJidiEESaIS1_EESaIS3_EED2Ev.exit, label %if.then.i.i.i

if.then.i.i.i:                                    ; preds = %_ZSt8_DestroyIPSt6vectorISt5tupleIJidiEESaIS2_EES4_EvT_S6_RSaIT0_E.exit.i
  call void @_ZdlPv(i8* %18) #7
  br label %_ZNSt6vectorIS_ISt5tupleIJidiEESaIS1_EESaIS3_EED2Ev.exit

_ZNSt6vectorIS_ISt5tupleIJidiEESaIS1_EESaIS3_EED2Ev.exit: ; preds = %if.then.i.i.i, %_ZSt8_DestroyIPSt6vectorISt5tupleIJidiEESaIS2_EES4_EvT_S6_RSaIT0_E.exit.i
  ret void

for.body61:                                       ; preds = %for.cond.cleanup68, %for.body61.lr.ph
  %indvars.iv240 = phi i64 [ 0, %for.body61.lr.ph ], [ %indvars.iv.next241, %for.cond.cleanup68 ]
  %add.ptr.i119 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %cond.i.i.i.i212259, i64 %indvars.iv240
  %_M_finish.i115 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %cond.i.i.i.i212259, i64 %indvars.iv240, i32 0, i32 0, i32 1
  %84 = bitcast %"class.std::tuple"** %_M_finish.i115 to i64*
  %85 = load i64, i64* %84, align 8, !tbaa !12
  %86 = bitcast %"class.std::vector.0"* %add.ptr.i119 to i64*
  %87 = load i64, i64* %86, align 8, !tbaa !9
  %88 = icmp eq i64 %85, %87
  br i1 %88, label %for.cond.cleanup68, label %for.body69.preheader

for.body69.preheader:                             ; preds = %for.body61
  br label %for.body69

for.cond.cleanup68.loopexit:                      ; preds = %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE9push_backERKS1_.exit
  br label %for.cond.cleanup68

for.cond.cleanup68:                               ; preds = %for.cond.cleanup68.loopexit, %for.body61
  %indvars.iv.next241 = add nuw i64 %indvars.iv240, 1
  %cmp59 = icmp ugt i64 %sub.ptr.div.i144, %indvars.iv.next241
  br i1 %cmp59, label %for.body61, label %for.cond.cleanup60.loopexit

for.body69:                                       ; preds = %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE9push_backERKS1_.exit, %for.body69.preheader
  %.in = phi i64 [ %96, %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE9push_backERKS1_.exit ], [ %87, %for.body69.preheader ]
  %indvars.iv = phi i64 [ %indvars.iv.next, %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE9push_backERKS1_.exit ], [ 0, %for.body69.preheader ]
  %89 = inttoptr i64 %.in to %"class.std::tuple"*
  %add.ptr.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %89, i64 %indvars.iv
  %90 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_finish.i, align 8, !tbaa !12
  %91 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_end_of_storage.i.i, align 8, !tbaa !13
  %cmp.i = icmp eq %"class.std::tuple"* %90, %91
  br i1 %cmp.i, label %if.else.i, label %if.then.i

if.then.i:                                        ; preds = %for.body69
  %92 = bitcast %"class.std::tuple"* %90 to i8*
  %93 = bitcast %"class.std::tuple"* %add.ptr.i to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %92, i8* nonnull %93, i64 24, i32 8, i1 false) #7
  %94 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_finish.i, align 8, !tbaa !12
  %incdec.ptr.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %94, i64 1
  store %"class.std::tuple"* %incdec.ptr.i, %"class.std::tuple"** %_M_finish.i, align 8, !tbaa !12
  br label %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE9push_backERKS1_.exit

if.else.i:                                        ; preds = %for.body69
  call void @_ZNSt6vectorISt5tupleIJidiEESaIS1_EE17_M_realloc_insertIJRKS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_(%"class.std::vector.0"* nonnull %agg.result, %"class.std::tuple"* %90, %"class.std::tuple"* nonnull dereferenceable(24) %add.ptr.i) #7
  br label %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE9push_backERKS1_.exit

_ZNSt6vectorISt5tupleIJidiEESaIS1_EE9push_backERKS1_.exit: ; preds = %if.else.i, %if.then.i
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %95 = load i64, i64* %84, align 8, !tbaa !12
  %96 = load i64, i64* %86, align 8, !tbaa !9
  %sub.ptr.sub.i116 = sub i64 %95, %96
  %sub.ptr.div.i117 = sdiv exact i64 %sub.ptr.sub.i116, 24
  %cmp67 = icmp ugt i64 %sub.ptr.div.i117, %indvars.iv.next
  br i1 %cmp67, label %for.body69, label %for.cond.cleanup68.loopexit
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #1

; Function Attrs: nounwind
declare i64 @clock() local_unnamed_addr #2

declare i32 @_Z15get_valid_movesi(i32) local_unnamed_addr #3

declare i32 @_Z17get_matches_counti(i32) local_unnamed_addr #3

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(i8*) local_unnamed_addr #4

; Function Attrs: noreturn
declare void @_ZSt20__throw_length_errorPKc(i8*) local_unnamed_addr #5

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #1

; Function Attrs: noreturn
declare void @_ZSt17__throw_bad_allocv() local_unnamed_addr #5

; Function Attrs: nobuiltin
declare noalias nonnull i8* @_Znwm(i64) local_unnamed_addr #6

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZNSt6vectorISt5tupleIJidiEESaIS1_EE17_M_realloc_insertIJS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_(%"class.std::vector.0"* %this, %"class.std::tuple"* %__position.coerce, %"class.std::tuple"* dereferenceable(24) %__args) local_unnamed_addr #0 comdat align 2 {
entry:
  %0 = ptrtoint %"class.std::tuple"* %__position.coerce to i64
  %_M_finish.i20.i = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %this, i64 0, i32 0, i32 0, i32 1
  %1 = bitcast %"class.std::tuple"** %_M_finish.i20.i to i64*
  %2 = load i64, i64* %1, align 8, !tbaa !12
  %3 = bitcast %"class.std::vector.0"* %this to i64*
  %4 = load i64, i64* %3, align 8, !tbaa !33
  %sub.ptr.sub.i21.i = sub i64 %2, %4
  %sub.ptr.div.i22.i = sdiv exact i64 %sub.ptr.sub.i21.i, 24
  %5 = icmp eq i64 %sub.ptr.sub.i21.i, 0
  %.sroa.speculated.i = select i1 %5, i64 1, i64 %sub.ptr.div.i22.i
  %add.i = add nsw i64 %.sroa.speculated.i, %sub.ptr.div.i22.i
  %cmp7.i = icmp ult i64 %add.i, %sub.ptr.div.i22.i
  %cmp9.i = icmp ugt i64 %add.i, 768614336404564650
  %or.cond.i = or i1 %cmp7.i, %cmp9.i
  %cond.i = select i1 %or.cond.i, i64 768614336404564650, i64 %add.i
  %sub.ptr.sub.i = sub i64 %0, %4
  %sub.ptr.div.i = sdiv exact i64 %sub.ptr.sub.i, 24
  %cmp.i.i.i = icmp ugt i64 %cond.i, 768614336404564650
  br i1 %cmp.i.i.i, label %if.then.i.i.i, label %_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i

if.then.i.i.i:                                    ; preds = %entry
  tail call void @_ZSt17__throw_bad_allocv() #8
  unreachable

_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i: ; preds = %entry
  %6 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %this, i64 0, i32 0, i32 0, i32 0
  %mul.i.i.i = mul i64 %cond.i, 24
  %call2.i.i.i = tail call i8* @_Znwm(i64 %mul.i.i.i) #7
  %7 = bitcast i8* %call2.i.i.i to %"class.std::tuple"*
  %_M_head_impl.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__args, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %_M_head_impl.i.i6.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %7, i64 %sub.ptr.div.i, i32 0, i32 0, i32 0, i32 0, i32 0
  %8 = load i32, i32* %_M_head_impl.i.i.i.i.i.i.i.i, align 4, !tbaa !14
  store i32 %8, i32* %_M_head_impl.i.i6.i.i.i.i.i, align 4, !tbaa !20
  %9 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %7, i64 %sub.ptr.div.i, i32 0, i32 0, i32 1
  %10 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__args, i64 0, i32 0, i32 0, i32 1, i32 0
  %11 = bitcast double* %10 to i64*
  %12 = load i64, i64* %11, align 8, !tbaa !34
  %13 = bitcast %"struct.std::_Head_base.7"* %9 to i64*
  store i64 %12, i64* %13, align 8, !tbaa !25
  %14 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__args, i64 0, i32 0, i32 1, i32 0
  %15 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %7, i64 %sub.ptr.div.i, i32 0, i32 1, i32 0
  %16 = load i32, i32* %14, align 4, !tbaa !14
  store i32 %16, i32* %15, align 4, !tbaa !28
  %17 = load %"class.std::tuple"*, %"class.std::tuple"** %6, align 8, !tbaa !9
  %cmp.i.i10.i.i.i.i44 = icmp eq %"class.std::tuple"* %17, %__position.coerce
  br i1 %cmp.i.i10.i.i.i.i44, label %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit55, label %for.body.i.i.i.i53.preheader

for.body.i.i.i.i53.preheader:                     ; preds = %_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i
  br label %for.body.i.i.i.i53

for.body.i.i.i.i53:                               ; preds = %for.body.i.i.i.i53, %for.body.i.i.i.i53.preheader
  %__cur.012.i.i.i.i46 = phi %"class.std::tuple"* [ %incdec.ptr.i.i.i.i51, %for.body.i.i.i.i53 ], [ %7, %for.body.i.i.i.i53.preheader ]
  %__first.sroa.0.011.i.i.i.i47 = phi %"class.std::tuple"* [ %incdec.ptr.i.i.i.i.i50, %for.body.i.i.i.i53 ], [ %17, %for.body.i.i.i.i53.preheader ]
  %_M_head_impl.i.i.i.i.i.i.i.i.i.i.i48 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.011.i.i.i.i47, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %_M_head_impl.i.i6.i.i.i.i.i.i.i.i49 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.012.i.i.i.i46, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %18 = load i32, i32* %_M_head_impl.i.i.i.i.i.i.i.i.i.i.i48, align 4, !tbaa !14
  store i32 %18, i32* %_M_head_impl.i.i6.i.i.i.i.i.i.i.i49, align 4, !tbaa !20
  %19 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.012.i.i.i.i46, i64 0, i32 0, i32 0, i32 1
  %20 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.011.i.i.i.i47, i64 0, i32 0, i32 0, i32 1, i32 0
  %21 = bitcast double* %20 to i64*
  %22 = load i64, i64* %21, align 8, !tbaa !34
  %23 = bitcast %"struct.std::_Head_base.7"* %19 to i64*
  store i64 %22, i64* %23, align 8, !tbaa !25
  %24 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.011.i.i.i.i47, i64 0, i32 0, i32 1, i32 0
  %25 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.012.i.i.i.i46, i64 0, i32 0, i32 1, i32 0
  %26 = load i32, i32* %24, align 4, !tbaa !14
  store i32 %26, i32* %25, align 4, !tbaa !28
  %incdec.ptr.i.i.i.i.i50 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.011.i.i.i.i47, i64 1
  %incdec.ptr.i.i.i.i51 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.012.i.i.i.i46, i64 1
  %cmp.i.i.i.i.i.i52 = icmp eq %"class.std::tuple"* %incdec.ptr.i.i.i.i.i50, %__position.coerce
  br i1 %cmp.i.i.i.i.i.i52, label %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit55.loopexit, label %for.body.i.i.i.i53

_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit55.loopexit: ; preds = %for.body.i.i.i.i53
  %incdec.ptr.i.i.i.i51.lcssa = phi %"class.std::tuple"* [ %incdec.ptr.i.i.i.i51, %for.body.i.i.i.i53 ]
  br label %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit55

_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit55: ; preds = %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit55.loopexit, %_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i
  %__cur.0.lcssa.i.i.i.i54 = phi %"class.std::tuple"* [ %7, %_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i ], [ %incdec.ptr.i.i.i.i51.lcssa, %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit55.loopexit ]
  %incdec.ptr = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.0.lcssa.i.i.i.i54, i64 1
  %27 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_finish.i20.i, align 8, !tbaa !12
  %cmp.i.i10.i.i.i.i = icmp eq %"class.std::tuple"* %27, %__position.coerce
  br i1 %cmp.i.i10.i.i.i.i, label %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit, label %for.body.i.i.i.i.preheader

for.body.i.i.i.i.preheader:                       ; preds = %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit55
  br label %for.body.i.i.i.i

for.body.i.i.i.i:                                 ; preds = %for.body.i.i.i.i, %for.body.i.i.i.i.preheader
  %__cur.012.i.i.i.i = phi %"class.std::tuple"* [ %incdec.ptr.i.i.i.i, %for.body.i.i.i.i ], [ %incdec.ptr, %for.body.i.i.i.i.preheader ]
  %__first.sroa.0.011.i.i.i.i = phi %"class.std::tuple"* [ %incdec.ptr.i.i.i.i.i, %for.body.i.i.i.i ], [ %__position.coerce, %for.body.i.i.i.i.preheader ]
  %_M_head_impl.i.i.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.011.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %_M_head_impl.i.i6.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.012.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %28 = load i32, i32* %_M_head_impl.i.i.i.i.i.i.i.i.i.i.i, align 4, !tbaa !14
  store i32 %28, i32* %_M_head_impl.i.i6.i.i.i.i.i.i.i.i, align 4, !tbaa !20
  %29 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.012.i.i.i.i, i64 0, i32 0, i32 0, i32 1
  %30 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.011.i.i.i.i, i64 0, i32 0, i32 0, i32 1, i32 0
  %31 = bitcast double* %30 to i64*
  %32 = load i64, i64* %31, align 8, !tbaa !34
  %33 = bitcast %"struct.std::_Head_base.7"* %29 to i64*
  store i64 %32, i64* %33, align 8, !tbaa !25
  %34 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.011.i.i.i.i, i64 0, i32 0, i32 1, i32 0
  %35 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.012.i.i.i.i, i64 0, i32 0, i32 1, i32 0
  %36 = load i32, i32* %34, align 4, !tbaa !14
  store i32 %36, i32* %35, align 4, !tbaa !28
  %incdec.ptr.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.011.i.i.i.i, i64 1
  %incdec.ptr.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.012.i.i.i.i, i64 1
  %cmp.i.i.i.i.i.i = icmp eq %"class.std::tuple"* %incdec.ptr.i.i.i.i.i, %27
  br i1 %cmp.i.i.i.i.i.i, label %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit.loopexit, label %for.body.i.i.i.i

_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit.loopexit: ; preds = %for.body.i.i.i.i
  %incdec.ptr.i.i.i.i.lcssa = phi %"class.std::tuple"* [ %incdec.ptr.i.i.i.i, %for.body.i.i.i.i ]
  br label %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit

_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit: ; preds = %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit.loopexit, %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit55
  %__cur.0.lcssa.i.i.i.i = phi %"class.std::tuple"* [ %incdec.ptr, %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit55 ], [ %incdec.ptr.i.i.i.i.lcssa, %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit.loopexit ]
  %_M_end_of_storage = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %this, i64 0, i32 0, i32 0, i32 2
  %tobool.i = icmp eq %"class.std::tuple"* %17, null
  br i1 %tobool.i, label %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit, label %if.then.i43

if.then.i43:                                      ; preds = %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit
  %37 = bitcast %"class.std::tuple"* %17 to i8*
  tail call void @_ZdlPv(i8* %37) #7
  br label %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit

_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit: ; preds = %if.then.i43, %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit
  %38 = bitcast %"class.std::vector.0"* %this to i8**
  store i8* %call2.i.i.i, i8** %38, align 8, !tbaa !9
  store %"class.std::tuple"* %__cur.0.lcssa.i.i.i.i, %"class.std::tuple"** %_M_finish.i20.i, align 8, !tbaa !12
  %add.ptr29 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %7, i64 %cond.i
  store %"class.std::tuple"* %add.ptr29, %"class.std::tuple"** %_M_end_of_storage, align 8, !tbaa !13
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZNSt6vectorISt5tupleIJidiEESaIS1_EE17_M_realloc_insertIJRKS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_(%"class.std::vector.0"* %this, %"class.std::tuple"* %__position.coerce, %"class.std::tuple"* dereferenceable(24) %__args) local_unnamed_addr #0 comdat align 2 {
entry:
  %0 = ptrtoint %"class.std::tuple"* %__position.coerce to i64
  %_M_finish.i20.i = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %this, i64 0, i32 0, i32 0, i32 1
  %1 = bitcast %"class.std::tuple"** %_M_finish.i20.i to i64*
  %2 = load i64, i64* %1, align 8, !tbaa !12
  %3 = bitcast %"class.std::vector.0"* %this to i64*
  %4 = load i64, i64* %3, align 8, !tbaa !33
  %sub.ptr.sub.i21.i = sub i64 %2, %4
  %sub.ptr.div.i22.i = sdiv exact i64 %sub.ptr.sub.i21.i, 24
  %5 = icmp eq i64 %sub.ptr.sub.i21.i, 0
  %.sroa.speculated.i = select i1 %5, i64 1, i64 %sub.ptr.div.i22.i
  %add.i = add nsw i64 %.sroa.speculated.i, %sub.ptr.div.i22.i
  %cmp7.i = icmp ult i64 %add.i, %sub.ptr.div.i22.i
  %cmp9.i = icmp ugt i64 %add.i, 768614336404564650
  %or.cond.i = or i1 %cmp7.i, %cmp9.i
  %cond.i = select i1 %or.cond.i, i64 768614336404564650, i64 %add.i
  %6 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %this, i64 0, i32 0, i32 0, i32 0
  %sub.ptr.sub.i = sub i64 %0, %4
  %sub.ptr.div.i = sdiv exact i64 %sub.ptr.sub.i, 24
  %cmp.i57 = icmp eq i64 %cond.i, 0
  %7 = inttoptr i64 %4 to %"class.std::tuple"*
  br i1 %cmp.i57, label %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE11_M_allocateEm.exit, label %cond.true.i

cond.true.i:                                      ; preds = %entry
  %cmp.i.i.i = icmp ugt i64 %cond.i, 768614336404564650
  br i1 %cmp.i.i.i, label %if.then.i.i.i, label %_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i

if.then.i.i.i:                                    ; preds = %cond.true.i
  tail call void @_ZSt17__throw_bad_allocv() #8
  unreachable

_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i: ; preds = %cond.true.i
  %mul.i.i.i = mul i64 %cond.i, 24
  %call2.i.i.i = tail call i8* @_Znwm(i64 %mul.i.i.i) #7
  %8 = bitcast i8* %call2.i.i.i to %"class.std::tuple"*
  %.pre = load %"class.std::tuple"*, %"class.std::tuple"** %6, align 8, !tbaa !9
  br label %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE11_M_allocateEm.exit

_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE11_M_allocateEm.exit: ; preds = %_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i, %entry
  %9 = phi %"class.std::tuple"* [ %.pre, %_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i ], [ %7, %entry ]
  %cond.i58 = phi %"class.std::tuple"* [ %8, %_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i ], [ null, %entry ]
  %add.ptr = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %cond.i58, i64 %sub.ptr.div.i
  %10 = bitcast %"class.std::tuple"* %add.ptr to i8*
  %11 = bitcast %"class.std::tuple"* %__args to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %10, i8* nonnull %11, i64 24, i32 8, i1 false) #7
  %cmp.i.i10.i.i.i.i44 = icmp eq %"class.std::tuple"* %9, %__position.coerce
  br i1 %cmp.i.i10.i.i.i.i44, label %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit55, label %for.body.i.i.i.i53.preheader

for.body.i.i.i.i53.preheader:                     ; preds = %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE11_M_allocateEm.exit
  br label %for.body.i.i.i.i53

for.body.i.i.i.i53:                               ; preds = %for.body.i.i.i.i53, %for.body.i.i.i.i53.preheader
  %__cur.012.i.i.i.i46 = phi %"class.std::tuple"* [ %incdec.ptr.i.i.i.i51, %for.body.i.i.i.i53 ], [ %cond.i58, %for.body.i.i.i.i53.preheader ]
  %__first.sroa.0.011.i.i.i.i47 = phi %"class.std::tuple"* [ %incdec.ptr.i.i.i.i.i50, %for.body.i.i.i.i53 ], [ %9, %for.body.i.i.i.i53.preheader ]
  %_M_head_impl.i.i.i.i.i.i.i.i.i.i.i48 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.011.i.i.i.i47, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %_M_head_impl.i.i6.i.i.i.i.i.i.i.i49 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.012.i.i.i.i46, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %12 = load i32, i32* %_M_head_impl.i.i.i.i.i.i.i.i.i.i.i48, align 4, !tbaa !14
  store i32 %12, i32* %_M_head_impl.i.i6.i.i.i.i.i.i.i.i49, align 4, !tbaa !20
  %13 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.012.i.i.i.i46, i64 0, i32 0, i32 0, i32 1
  %14 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.011.i.i.i.i47, i64 0, i32 0, i32 0, i32 1, i32 0
  %15 = bitcast double* %14 to i64*
  %16 = load i64, i64* %15, align 8, !tbaa !34
  %17 = bitcast %"struct.std::_Head_base.7"* %13 to i64*
  store i64 %16, i64* %17, align 8, !tbaa !25
  %18 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.011.i.i.i.i47, i64 0, i32 0, i32 1, i32 0
  %19 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.012.i.i.i.i46, i64 0, i32 0, i32 1, i32 0
  %20 = load i32, i32* %18, align 4, !tbaa !14
  store i32 %20, i32* %19, align 4, !tbaa !28
  %incdec.ptr.i.i.i.i.i50 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.011.i.i.i.i47, i64 1
  %incdec.ptr.i.i.i.i51 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.012.i.i.i.i46, i64 1
  %cmp.i.i.i.i.i.i52 = icmp eq %"class.std::tuple"* %incdec.ptr.i.i.i.i.i50, %__position.coerce
  br i1 %cmp.i.i.i.i.i.i52, label %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit55.loopexit, label %for.body.i.i.i.i53

_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit55.loopexit: ; preds = %for.body.i.i.i.i53
  %incdec.ptr.i.i.i.i51.lcssa = phi %"class.std::tuple"* [ %incdec.ptr.i.i.i.i51, %for.body.i.i.i.i53 ]
  br label %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit55

_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit55: ; preds = %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit55.loopexit, %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE11_M_allocateEm.exit
  %__cur.0.lcssa.i.i.i.i54 = phi %"class.std::tuple"* [ %cond.i58, %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE11_M_allocateEm.exit ], [ %incdec.ptr.i.i.i.i51.lcssa, %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit55.loopexit ]
  %incdec.ptr = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.0.lcssa.i.i.i.i54, i64 1
  %21 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_finish.i20.i, align 8, !tbaa !12
  %cmp.i.i10.i.i.i.i = icmp eq %"class.std::tuple"* %21, %__position.coerce
  br i1 %cmp.i.i10.i.i.i.i, label %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit, label %for.body.i.i.i.i.preheader

for.body.i.i.i.i.preheader:                       ; preds = %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit55
  br label %for.body.i.i.i.i

for.body.i.i.i.i:                                 ; preds = %for.body.i.i.i.i, %for.body.i.i.i.i.preheader
  %__cur.012.i.i.i.i = phi %"class.std::tuple"* [ %incdec.ptr.i.i.i.i, %for.body.i.i.i.i ], [ %incdec.ptr, %for.body.i.i.i.i.preheader ]
  %__first.sroa.0.011.i.i.i.i = phi %"class.std::tuple"* [ %incdec.ptr.i.i.i.i.i, %for.body.i.i.i.i ], [ %__position.coerce, %for.body.i.i.i.i.preheader ]
  %_M_head_impl.i.i.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.011.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %_M_head_impl.i.i6.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.012.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %22 = load i32, i32* %_M_head_impl.i.i.i.i.i.i.i.i.i.i.i, align 4, !tbaa !14
  store i32 %22, i32* %_M_head_impl.i.i6.i.i.i.i.i.i.i.i, align 4, !tbaa !20
  %23 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.012.i.i.i.i, i64 0, i32 0, i32 0, i32 1
  %24 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.011.i.i.i.i, i64 0, i32 0, i32 0, i32 1, i32 0
  %25 = bitcast double* %24 to i64*
  %26 = load i64, i64* %25, align 8, !tbaa !34
  %27 = bitcast %"struct.std::_Head_base.7"* %23 to i64*
  store i64 %26, i64* %27, align 8, !tbaa !25
  %28 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.011.i.i.i.i, i64 0, i32 0, i32 1, i32 0
  %29 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.012.i.i.i.i, i64 0, i32 0, i32 1, i32 0
  %30 = load i32, i32* %28, align 4, !tbaa !14
  store i32 %30, i32* %29, align 4, !tbaa !28
  %incdec.ptr.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.011.i.i.i.i, i64 1
  %incdec.ptr.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.012.i.i.i.i, i64 1
  %cmp.i.i.i.i.i.i = icmp eq %"class.std::tuple"* %incdec.ptr.i.i.i.i.i, %21
  br i1 %cmp.i.i.i.i.i.i, label %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit.loopexit, label %for.body.i.i.i.i

_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit.loopexit: ; preds = %for.body.i.i.i.i
  %incdec.ptr.i.i.i.i.lcssa = phi %"class.std::tuple"* [ %incdec.ptr.i.i.i.i, %for.body.i.i.i.i ]
  br label %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit

_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit: ; preds = %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit.loopexit, %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit55
  %__cur.0.lcssa.i.i.i.i = phi %"class.std::tuple"* [ %incdec.ptr, %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit55 ], [ %incdec.ptr.i.i.i.i.lcssa, %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit.loopexit ]
  %_M_end_of_storage = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %this, i64 0, i32 0, i32 0, i32 2
  %tobool.i = icmp eq %"class.std::tuple"* %9, null
  br i1 %tobool.i, label %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit, label %if.then.i43

if.then.i43:                                      ; preds = %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit
  %31 = bitcast %"class.std::tuple"* %9 to i8*
  tail call void @_ZdlPv(i8* %31) #7
  br label %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit

_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit: ; preds = %if.then.i43, %_ZSt34__uninitialized_move_if_noexcept_aIPSt5tupleIJidiEES2_SaIS1_EET0_T_S5_S4_RT1_.exit
  store %"class.std::tuple"* %cond.i58, %"class.std::tuple"** %6, align 8, !tbaa !9
  store %"class.std::tuple"* %__cur.0.lcssa.i.i.i.i, %"class.std::tuple"** %_M_finish.i20.i, align 8, !tbaa !12
  %add.ptr29 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %cond.i58, i64 %cond.i
  store %"class.std::tuple"* %add.ptr29, %"class.std::tuple"** %_M_end_of_storage, align 8, !tbaa !13
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #1

; CHECK-LABEL: define internal fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE.outline_pfor.detach21.ls2(i64
; CHECK: %[[SYNCREG:.+]] = call token @llvm.syncregion.start()
; CHECK: detach within %[[SYNCREG]], label %.split, label %{{.+}}
; CHECK: {{^.split}}:
; CHECK-NEXT: call fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE.outline_pfor.detach21.ls2(i64

; CHECK-LABEL: define internal fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE.outline_pfor.detach.ls1(i64
; CHECK: %[[SYNCREG:.+]] = tail call token @llvm.syncregion.start()
; CHECK: detach within %[[SYNCREG]], label %.split, label %{{.+}}
; CHECK: {{^.split:}}
; CHECK-NEXT: call fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE.outline_pfor.detach.ls1(i64
; CHECK: {{^pfor.detach21.preheader.ls1:}}
; CHECK: call fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE.outline_pfor.detach21.ls2(i64 0,

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nobuiltin nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noreturn "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nobuiltin "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nounwind }
attributes #8 = { noreturn nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.0 (git@github.com:wsmoses/Tapir-Clang.git e0c101088ce269d4bb0fcbc91d328db51fa809ae) (git@github.com:wsmoses/Tapir-LLVM.git 108f1043ddbc9e235a885e734a031d213eeabe5a)"}
!2 = !{!3, !5, i64 8}
!3 = !{!"_ZTSSt12_Vector_baseI6paramsSaIS0_EE", !4, i64 0}
!4 = !{!"_ZTSNSt12_Vector_baseI6paramsSaIS0_EE12_Vector_implE", !5, i64 0, !5, i64 8, !5, i64 16}
!5 = !{!"any pointer", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
!8 = !{!3, !5, i64 0}
!9 = !{!10, !5, i64 0}
!10 = !{!"_ZTSSt12_Vector_baseISt5tupleIJidiEESaIS1_EE", !11, i64 0}
!11 = !{!"_ZTSNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE12_Vector_implE", !5, i64 0, !5, i64 8, !5, i64 16}
!12 = !{!10, !5, i64 8}
!13 = !{!10, !5, i64 16}
!14 = !{!15, !15, i64 0}
!15 = !{!"int", !6, i64 0}
!16 = distinct !{!16, !17}
!17 = !{!"tapir.loop.spawn.strategy", i32 1}
!18 = distinct !{!18, !19}
!19 = !{!"llvm.loop.isvectorized", i32 1}
!20 = !{!21, !15, i64 0}
!21 = !{!"_ZTSSt10_Head_baseILm2EiLb0EE", !15, i64 0}
!22 = !{!23}
!23 = distinct !{!23, !24, !"_ZSt10make_tupleIJRiRdS0_EESt5tupleIJDpNSt17__decay_and_stripIT_E6__typeEEEDpOS4_: %agg.result"}
!24 = distinct !{!24, !"_ZSt10make_tupleIJRiRdS0_EESt5tupleIJDpNSt17__decay_and_stripIT_E6__typeEEEDpOS4_"}
!25 = !{!26, !27, i64 0}
!26 = !{!"_ZTSSt10_Head_baseILm1EdLb0EE", !27, i64 0}
!27 = !{!"double", !6, i64 0}
!28 = !{!29, !15, i64 0}
!29 = !{!"_ZTSSt10_Head_baseILm0EiLb0EE", !15, i64 0}
!30 = distinct !{!30, !31, !19}
!31 = !{!"llvm.loop.unroll.runtime.disable"}
!32 = distinct !{!32, !17}
!33 = !{!5, !5, i64 0}
!34 = !{!27, !27, i64 0}
