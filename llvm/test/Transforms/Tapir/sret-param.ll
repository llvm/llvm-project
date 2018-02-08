; Test to verify that LoopSpawning creates a helper function that places any
; sret parameters at the begin of the argument list.
;
; Credit to Tim Kaler for producing the source code that inspired this test
; case.
;
; RUN: opt < %s -loop-spawning -ls-tapir-target=cilk -simplifycfg -S | FileCheck %s --check-prefix=LS
; RUN: opt < %s -tapir2target -tapir-target=cilk -simplifycfg -S | FileCheck %s --check-prefix=TT

; ModuleID = 'sret-test.cpp'
source_filename = "sret-test.cpp"
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

$_ZNSt6vectorISt5tupleIJidiEESaIS1_EE19_M_emplace_back_auxIJS1_EEEvDpOT_ = comdat any

$_ZNSt6vectorISt5tupleIJidiEESaIS1_EE19_M_emplace_back_auxIJRKS1_EEEvDpOT_ = comdat any

@.str = private unnamed_addr constant [16 x i8] c"vector::reserve\00", align 1

; Function Attrs: uwtable
define void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE(%"class.std::vector.0"* noalias sret %agg.result, i32 %trials, double %threshold, %"class.std::vector"* nocapture readonly dereferenceable(24) %ps) local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %0 = bitcast %"class.std::vector.0"* %agg.result to i8*
  tail call void @llvm.memset.p0i8.i64(i8* %0, i8 0, i64 24, i32 8, i1 false) #7
  %_M_finish.i = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %ps, i64 0, i32 0, i32 0, i32 1
  %1 = bitcast %struct.params** %_M_finish.i to i64*
  %2 = load i64, i64* %1, align 8, !tbaa !2
  %3 = bitcast %"class.std::vector"* %ps to i64*
  %4 = load i64, i64* %3, align 8, !tbaa !8
  %sub.ptr.sub.i = sub i64 %2, %4
  %sub.ptr.div.i = sdiv exact i64 %sub.ptr.sub.i, 24
  %cmp.i = icmp ugt i64 %sub.ptr.div.i, 768614336404564650
  br i1 %cmp.i, label %if.then.i, label %if.end.i

if.then.i:                                        ; preds = %entry
  invoke void @_ZSt20__throw_length_errorPKc(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str, i64 0, i64 0)) #8
          to label %.noexc unwind label %lpad

.noexc:                                           ; preds = %if.then.i
  unreachable

if.end.i:                                         ; preds = %entry
  %_M_end_of_storage.i.i = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %agg.result, i64 0, i32 0, i32 0, i32 2
  %5 = icmp eq i64 %sub.ptr.sub.i, 0
  br i1 %5, label %invoke.cont, label %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE11_M_allocateEm.exit.i.i

_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE11_M_allocateEm.exit.i.i: ; preds = %if.end.i
  %_M_finish.i.i = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %agg.result, i64 0, i32 0, i32 0, i32 1
  %call2.i.i.i.i.i166 = invoke i8* @_Znwm(i64 %sub.ptr.sub.i)
          to label %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE20_M_allocate_and_copyISt13move_iteratorIPS1_EEES6_mT_S8_.exit.i unwind label %lpad

_ZNSt6vectorISt5tupleIJidiEESaIS1_EE20_M_allocate_and_copyISt13move_iteratorIPS1_EEES6_mT_S8_.exit.i: ; preds = %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE11_M_allocateEm.exit.i.i
  %_M_start.i165 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %agg.result, i64 0, i32 0, i32 0, i32 0
  %6 = bitcast i8* %call2.i.i.i.i.i166 to %"class.std::tuple"*
  %7 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_start.i165, align 8, !tbaa !9
  %tobool.i.i = icmp eq %"class.std::tuple"* %7, null
  br i1 %tobool.i.i, label %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit.i, label %if.then.i.i

if.then.i.i:                                      ; preds = %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE20_M_allocate_and_copyISt13move_iteratorIPS1_EEES6_mT_S8_.exit.i
  %8 = bitcast %"class.std::tuple"* %7 to i8*
  tail call void @_ZdlPv(i8* %8) #7
  br label %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit.i

_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit.i: ; preds = %if.then.i.i, %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE20_M_allocate_and_copyISt13move_iteratorIPS1_EEES6_mT_S8_.exit.i
  %9 = bitcast %"class.std::vector.0"* %agg.result to i8**
  store i8* %call2.i.i.i.i.i166, i8** %9, align 8, !tbaa !9
  %10 = bitcast %"class.std::tuple"** %_M_finish.i.i to i8**
  store i8* %call2.i.i.i.i.i166, i8** %10, align 8, !tbaa !12
  %add.ptr30.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %6, i64 %sub.ptr.div.i
  store %"class.std::tuple"* %add.ptr30.i, %"class.std::tuple"** %_M_end_of_storage.i.i, align 8, !tbaa !13
  %.pre = load i64, i64* %1, align 8, !tbaa !2
  %.pre397 = load i64, i64* %3, align 8, !tbaa !8
  br label %invoke.cont

invoke.cont:                                      ; preds = %if.end.i, %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit.i
  %11 = phi i64 [ %.pre397, %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit.i ], [ %4, %if.end.i ]
  %12 = phi i64 [ %.pre, %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit.i ], [ %2, %if.end.i ]
  %sub.ptr.sub.i168 = sub i64 %12, %11
  %sub.ptr.div.i169 = sdiv exact i64 %sub.ptr.sub.i168, 24
  %conv = trunc i64 %sub.ptr.div.i169 to i32
  %sext = shl i64 %sub.ptr.div.i169, 32
  %conv2 = ashr exact i64 %sext, 32
  %cmp.i.i.i.i170 = icmp eq i64 %conv2, 0
  br i1 %cmp.i.i.i.i170, label %invoke.cont4, label %cond.true.i.i.i.i

cond.true.i.i.i.i:                                ; preds = %invoke.cont
  %cmp.i.i.i.i.i.i = icmp ugt i64 %conv2, 768614336404564650
  br i1 %cmp.i.i.i.i.i.i, label %if.then.i.i.i.i.i.i, label %_ZNSt16allocator_traitsISaISt6vectorISt5tupleIJidiEESaIS2_EEEE8allocateERS5_m.exit.i.i.i.i

if.then.i.i.i.i.i.i:                              ; preds = %cond.true.i.i.i.i
  invoke void @_ZSt17__throw_bad_allocv() #8
          to label %.noexc173 unwind label %lpad3

.noexc173:                                        ; preds = %if.then.i.i.i.i.i.i
  unreachable

_ZNSt16allocator_traitsISaISt6vectorISt5tupleIJidiEESaIS2_EEEE8allocateERS5_m.exit.i.i.i.i: ; preds = %cond.true.i.i.i.i
  %mul.i.i.i.i.i.i = mul nsw i64 %conv2, 24
  %call2.i.i.i.i3.i.i174 = invoke i8* @_Znwm(i64 %mul.i.i.i.i.i.i)
          to label %for.body.lr.ph.i.i.i.i.i171 unwind label %lpad3

for.body.lr.ph.i.i.i.i.i171:                      ; preds = %_ZNSt16allocator_traitsISaISt6vectorISt5tupleIJidiEESaIS2_EEEE8allocateERS5_m.exit.i.i.i.i
  %13 = bitcast i8* %call2.i.i.i.i3.i.i174 to %"class.std::vector.0"*
  %14 = ptrtoint i8* %call2.i.i.i.i3.i.i174 to i64
  %add.ptr.i.i.i = getelementptr %"class.std::vector.0", %"class.std::vector.0"* %13, i64 %conv2
  tail call void @llvm.memset.p0i8.i64(i8* nonnull %call2.i.i.i.i3.i.i174, i8 0, i64 %mul.i.i.i.i.i.i, i32 8, i1 false)
  br label %invoke.cont4

invoke.cont4:                                     ; preds = %invoke.cont, %for.body.lr.ph.i.i.i.i.i171
  %15 = phi i64 [ %14, %for.body.lr.ph.i.i.i.i.i171 ], [ 0, %invoke.cont ]
  %16 = phi i8* [ %call2.i.i.i.i3.i.i174, %for.body.lr.ph.i.i.i.i.i171 ], [ null, %invoke.cont ]
  %cond.i.i.i.i326 = phi %"class.std::vector.0"* [ %13, %for.body.lr.ph.i.i.i.i.i171 ], [ null, %invoke.cont ]
  %__cur.0.lcssa.i.i.i.i.i = phi %"class.std::vector.0"* [ %add.ptr.i.i.i, %for.body.lr.ph.i.i.i.i.i171 ], [ null, %invoke.cont ]
  %17 = ptrtoint %"class.std::vector.0"* %__cur.0.lcssa.i.i.i.i.i to i64
  %cmp375 = icmp sgt i32 %conv, 0
  br i1 %cmp375, label %pfor.detach.lr.ph, label %pfor.cond.cleanup

pfor.detach.lr.ph:                                ; preds = %invoke.cont4
  %conv8 = sext i32 %trials to i64
  %cmp.i.i.i.i180 = icmp eq i32 %trials, 0
  %add.ptr.i.i.i230340 = getelementptr i32, i32* null, i64 %conv8
  %cmp27367 = icmp sgt i32 %trials, 0
  %cmp.i.i.i.i.i.i181348 = icmp slt i32 %trials, 0
  %mul.i.i.i.i.i.i184 = shl nsw i64 %conv8, 2
  %sext398 = shl i64 %sub.ptr.div.i169, 32
  %18 = ashr exact i64 %sext398, 32
  br label %pfor.detach
; LS: pfor.detach.lr.ph:
; LS: call fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE_pfor.detach.ls(%"class.std::vector.0"* %agg.result,
; TT: pfor.detach.split:
; TT-NEXT: call fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE_pfor.body.cilk(%"class.std::vector.0"* %agg.result,

pfor.cond.cleanup:                                ; preds = %pfor.inc78, %invoke.cont4
  sync within %syncreg, label %pfor.end.continue

pfor.end.continue:                                ; preds = %pfor.cond.cleanup
  %sub.ptr.sub.i241 = sub i64 %17, %15
  %sub.ptr.div.i242 = sdiv exact i64 %sub.ptr.sub.i241, 24
  %19 = icmp eq i64 %sub.ptr.sub.i241, 0
  br i1 %19, label %for.cond.cleanup89, label %for.body90.lr.ph

for.body90.lr.ph:                                 ; preds = %pfor.end.continue
  %_M_finish.i175 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %agg.result, i64 0, i32 0, i32 0, i32 1
  br label %for.body90

lpad:                                             ; preds = %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE11_M_allocateEm.exit.i.i, %if.then.i
  %20 = landingpad { i8*, i32 }
          cleanup
  %21 = extractvalue { i8*, i32 } %20, 0
  %22 = extractvalue { i8*, i32 } %20, 1
  br label %ehcleanup116

lpad3:                                            ; preds = %_ZNSt16allocator_traitsISaISt6vectorISt5tupleIJidiEESaIS2_EEEE8allocateERS5_m.exit.i.i.i.i, %if.then.i.i.i.i.i.i
  %23 = landingpad { i8*, i32 }
          cleanup
  %24 = extractvalue { i8*, i32 } %23, 0
  %25 = extractvalue { i8*, i32 } %23, 1
  br label %ehcleanup116

pfor.detach:                                      ; preds = %pfor.detach.lr.ph, %pfor.inc78
  %indvars.iv395 = phi i64 [ 0, %pfor.detach.lr.ph ], [ %indvars.iv.next396, %pfor.inc78 ]
  detach within %syncreg, label %pfor.body, label %pfor.inc78

pfor.body:                                        ; preds = %pfor.detach
  %worker_matches_count.sroa.13 = alloca i32*, align 8
  %syncreg18 = call token @llvm.syncregion.start()
  %ref.tmp63 = alloca %"class.std::tuple", align 8
  %call7 = call i64 @clock() #7
  br i1 %cmp.i.i.i.i180, label %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i232.thread, label %cond.true.i.i.i.i182

_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i232.thread: ; preds = %pfor.body
  %worker_matches_count.sroa.13.0..sroa_cast289334 = bitcast i32** %worker_matches_count.sroa.13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %worker_matches_count.sroa.13.0..sroa_cast289334)
  store i32* %add.ptr.i.i.i230340, i32** %worker_matches_count.sroa.13, align 8
  br label %invoke.cont17

cond.true.i.i.i.i182:                             ; preds = %pfor.body
  br i1 %cmp.i.i.i.i.i.i181348, label %if.then.i.i.i.i.i.i183, label %_ZNSt16allocator_traitsISaIiEE8allocateERS0_m.exit.i.i.i.i

if.then.i.i.i.i.i.i183:                           ; preds = %cond.true.i.i.i.i182
  invoke void @_ZSt17__throw_bad_allocv() #8
          to label %.noexc191 unwind label %lpad10.loopexit.split-lp.loopexit.split-lp

.noexc191:                                        ; preds = %if.then.i.i.i.i.i.i183
  unreachable

_ZNSt16allocator_traitsISaIiEE8allocateERS0_m.exit.i.i.i.i: ; preds = %cond.true.i.i.i.i182
  %call2.i.i.i.i3.i.i193 = invoke i8* @_Znwm(i64 %mul.i.i.i.i.i.i184)
          to label %_ZNSt16allocator_traitsISaIiEE8allocateERS0_m.exit.i.i.i.i226 unwind label %lpad10.loopexit.split-lp.loopexit

_ZNSt16allocator_traitsISaIiEE8allocateERS0_m.exit.i.i.i.i226: ; preds = %_ZNSt16allocator_traitsISaIiEE8allocateERS0_m.exit.i.i.i.i
  call void @llvm.memset.p0i8.i64(i8* nonnull %call2.i.i.i.i3.i.i193, i8 0, i64 %mul.i.i.i.i.i.i184, i32 4, i1 false)
  %worker_matches_count.sroa.13.0..sroa_cast289 = bitcast i32** %worker_matches_count.sroa.13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %worker_matches_count.sroa.13.0..sroa_cast289)
  store i32* null, i32** %worker_matches_count.sroa.13, align 8
  %call2.i.i.i.i3.i.i238 = invoke i8* @_Znwm(i64 %mul.i.i.i.i.i.i184)
          to label %for.body.lr.ph.i.i.i.i.i.i.i233 unwind label %ehcleanup67.thread

for.body.lr.ph.i.i.i.i.i.i.i233:                  ; preds = %_ZNSt16allocator_traitsISaIiEE8allocateERS0_m.exit.i.i.i.i226
  %26 = bitcast i8* %call2.i.i.i.i3.i.i193 to i32*
  %27 = bitcast i8* %call2.i.i.i.i3.i.i238 to i32*
  %28 = ptrtoint i8* %call2.i.i.i.i3.i.i238 to i64
  %add.ptr.i.i.i230 = getelementptr i32, i32* %27, i64 %conv8
  store i32* %add.ptr.i.i.i230, i32** %worker_matches_count.sroa.13, align 8
  call void @llvm.memset.p0i8.i64(i8* nonnull %call2.i.i.i.i3.i.i238, i8 0, i64 %mul.i.i.i.i.i.i184, i32 4, i1 false)
  br label %invoke.cont17

invoke.cont17:                                    ; preds = %for.body.lr.ph.i.i.i.i.i.i.i233, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i232.thread
  %29 = phi i64 [ %28, %for.body.lr.ph.i.i.i.i.i.i.i233 ], [ 0, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i232.thread ]
  %30 = phi i8* [ %call2.i.i.i.i3.i.i238, %for.body.lr.ph.i.i.i.i.i.i.i233 ], [ null, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i232.thread ]
  %cond.i.i.i.i227343 = phi i32* [ %27, %for.body.lr.ph.i.i.i.i.i.i.i233 ], [ null, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i232.thread ]
  %31 = phi i8* [ %call2.i.i.i.i3.i.i193, %for.body.lr.ph.i.i.i.i.i.i.i233 ], [ null, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i232.thread ]
  %cond.i.i.i.i185332336342 = phi i32* [ %26, %for.body.lr.ph.i.i.i.i.i.i.i233 ], [ null, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i232.thread ]
  %worker_matches_count.sroa.13.0..sroa_cast289337341 = phi i8* [ %worker_matches_count.sroa.13.0..sroa_cast289, %for.body.lr.ph.i.i.i.i.i.i.i233 ], [ %worker_matches_count.sroa.13.0..sroa_cast289334, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i232.thread ]
  %__first.addr.0.lcssa.i.i.i.i.i.i.i234 = phi i32* [ %add.ptr.i.i.i230, %for.body.lr.ph.i.i.i.i.i.i.i233 ], [ null, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i232.thread ]
  %32 = ptrtoint i32* %__first.addr.0.lcssa.i.i.i.i.i.i.i234 to i64
  br i1 %cmp27367, label %pfor.detach30.preheader, label %pfor.cond.cleanup28

pfor.detach30.preheader:                          ; preds = %invoke.cont17
  br label %pfor.detach30

pfor.cond.cleanup28:                              ; preds = %pfor.inc, %invoke.cont17
  sync within %syncreg18, label %pfor.end.continue29

pfor.end.continue29:                              ; preds = %pfor.cond.cleanup28
  %sub.ptr.sub.i262 = sub i64 %32, %29
  %sub.ptr.div.i263 = ashr exact i64 %sub.ptr.sub.i262, 2
  %cmp49369 = icmp eq i64 %sub.ptr.div.i263, 0
  br i1 %cmp49369, label %invoke.cont65, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %pfor.end.continue29
  %min.iters.check = icmp ult i64 %sub.ptr.div.i263, 8
  br i1 %min.iters.check, label %for.body.preheader, label %vector.ph

for.body.preheader:                               ; preds = %middle.block, %for.body.lr.ph
  %indvars.iv393.ph = phi i64 [ 0, %for.body.lr.ph ], [ %n.vec, %middle.block ]
  %matches_count.0371.ph = phi i32 [ 0, %for.body.lr.ph ], [ %74, %middle.block ]
  %valid_moves.0370.ph = phi i32 [ 0, %for.body.lr.ph ], [ %73, %middle.block ]
  br label %for.body

vector.ph:                                        ; preds = %for.body.lr.ph
  %n.vec = and i64 %sub.ptr.div.i263, -8
  %33 = add nsw i64 %n.vec, -8
  %34 = lshr exact i64 %33, 3
  %35 = add nuw nsw i64 %34, 1
  %xtraiter = and i64 %35, 1
  %36 = icmp eq i64 %34, 0
  br i1 %36, label %middle.block.unr-lcssa, label %vector.ph.new

vector.ph.new:                                    ; preds = %vector.ph
  %unroll_iter = sub nsw i64 %35, %xtraiter
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph.new
  %index = phi i64 [ 0, %vector.ph.new ], [ %index.next.1, %vector.body ]
  %vec.phi = phi <4 x i32> [ zeroinitializer, %vector.ph.new ], [ %53, %vector.body ]
  %vec.phi420 = phi <4 x i32> [ zeroinitializer, %vector.ph.new ], [ %54, %vector.body ]
  %vec.phi421 = phi <4 x i32> [ zeroinitializer, %vector.ph.new ], [ %59, %vector.body ]
  %vec.phi422 = phi <4 x i32> [ zeroinitializer, %vector.ph.new ], [ %60, %vector.body ]
  %niter = phi i64 [ %unroll_iter, %vector.ph.new ], [ %niter.nsub.1, %vector.body ]
  %37 = getelementptr inbounds i32, i32* %cond.i.i.i.i227343, i64 %index
  %38 = bitcast i32* %37 to <4 x i32>*
  %wide.load = load <4 x i32>, <4 x i32>* %38, align 4, !tbaa !14
  %39 = getelementptr i32, i32* %37, i64 4
  %40 = bitcast i32* %39 to <4 x i32>*
  %wide.load423 = load <4 x i32>, <4 x i32>* %40, align 4, !tbaa !14
  %41 = add nsw <4 x i32> %wide.load, %vec.phi
  %42 = add nsw <4 x i32> %wide.load423, %vec.phi420
  %43 = getelementptr inbounds i32, i32* %cond.i.i.i.i185332336342, i64 %index
  %44 = bitcast i32* %43 to <4 x i32>*
  %wide.load424 = load <4 x i32>, <4 x i32>* %44, align 4, !tbaa !14
  %45 = getelementptr i32, i32* %43, i64 4
  %46 = bitcast i32* %45 to <4 x i32>*
  %wide.load425 = load <4 x i32>, <4 x i32>* %46, align 4, !tbaa !14
  %47 = add nsw <4 x i32> %wide.load424, %vec.phi421
  %48 = add nsw <4 x i32> %wide.load425, %vec.phi422
  %index.next = or i64 %index, 8
  %49 = getelementptr inbounds i32, i32* %cond.i.i.i.i227343, i64 %index.next
  %50 = bitcast i32* %49 to <4 x i32>*
  %wide.load.1 = load <4 x i32>, <4 x i32>* %50, align 4, !tbaa !14
  %51 = getelementptr i32, i32* %49, i64 4
  %52 = bitcast i32* %51 to <4 x i32>*
  %wide.load423.1 = load <4 x i32>, <4 x i32>* %52, align 4, !tbaa !14
  %53 = add nsw <4 x i32> %wide.load.1, %41
  %54 = add nsw <4 x i32> %wide.load423.1, %42
  %55 = getelementptr inbounds i32, i32* %cond.i.i.i.i185332336342, i64 %index.next
  %56 = bitcast i32* %55 to <4 x i32>*
  %wide.load424.1 = load <4 x i32>, <4 x i32>* %56, align 4, !tbaa !14
  %57 = getelementptr i32, i32* %55, i64 4
  %58 = bitcast i32* %57 to <4 x i32>*
  %wide.load425.1 = load <4 x i32>, <4 x i32>* %58, align 4, !tbaa !14
  %59 = add nsw <4 x i32> %wide.load424.1, %47
  %60 = add nsw <4 x i32> %wide.load425.1, %48
  %index.next.1 = add nsw i64 %index, 16
  %niter.nsub.1 = add i64 %niter, -2
  %niter.ncmp.1 = icmp eq i64 %niter.nsub.1, 0
  br i1 %niter.ncmp.1, label %middle.block.unr-lcssa, label %vector.body, !llvm.loop !16

middle.block.unr-lcssa:                           ; preds = %vector.body, %vector.ph
  %.lcssa437.ph = phi <4 x i32> [ undef, %vector.ph ], [ %53, %vector.body ]
  %.lcssa436.ph = phi <4 x i32> [ undef, %vector.ph ], [ %54, %vector.body ]
  %.lcssa435.ph = phi <4 x i32> [ undef, %vector.ph ], [ %59, %vector.body ]
  %.lcssa.ph = phi <4 x i32> [ undef, %vector.ph ], [ %60, %vector.body ]
  %index.unr = phi i64 [ 0, %vector.ph ], [ %index.next.1, %vector.body ]
  %vec.phi.unr = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %53, %vector.body ]
  %vec.phi420.unr = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %54, %vector.body ]
  %vec.phi421.unr = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %59, %vector.body ]
  %vec.phi422.unr = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %60, %vector.body ]
  %lcmp.mod = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod, label %middle.block, label %vector.body.epil

vector.body.epil:                                 ; preds = %middle.block.unr-lcssa
  %61 = getelementptr inbounds i32, i32* %cond.i.i.i.i227343, i64 %index.unr
  %62 = getelementptr inbounds i32, i32* %cond.i.i.i.i185332336342, i64 %index.unr
  %63 = getelementptr i32, i32* %62, i64 4
  %64 = bitcast i32* %63 to <4 x i32>*
  %wide.load425.epil = load <4 x i32>, <4 x i32>* %64, align 4, !tbaa !14
  %65 = add nsw <4 x i32> %wide.load425.epil, %vec.phi422.unr
  %66 = bitcast i32* %62 to <4 x i32>*
  %wide.load424.epil = load <4 x i32>, <4 x i32>* %66, align 4, !tbaa !14
  %67 = add nsw <4 x i32> %wide.load424.epil, %vec.phi421.unr
  %68 = getelementptr i32, i32* %61, i64 4
  %69 = bitcast i32* %68 to <4 x i32>*
  %wide.load423.epil = load <4 x i32>, <4 x i32>* %69, align 4, !tbaa !14
  %70 = add nsw <4 x i32> %wide.load423.epil, %vec.phi420.unr
  %71 = bitcast i32* %61 to <4 x i32>*
  %wide.load.epil = load <4 x i32>, <4 x i32>* %71, align 4, !tbaa !14
  %72 = add nsw <4 x i32> %wide.load.epil, %vec.phi.unr
  br label %middle.block

middle.block:                                     ; preds = %middle.block.unr-lcssa, %vector.body.epil
  %.lcssa437 = phi <4 x i32> [ %.lcssa437.ph, %middle.block.unr-lcssa ], [ %72, %vector.body.epil ]
  %.lcssa436 = phi <4 x i32> [ %.lcssa436.ph, %middle.block.unr-lcssa ], [ %70, %vector.body.epil ]
  %.lcssa435 = phi <4 x i32> [ %.lcssa435.ph, %middle.block.unr-lcssa ], [ %67, %vector.body.epil ]
  %.lcssa = phi <4 x i32> [ %.lcssa.ph, %middle.block.unr-lcssa ], [ %65, %vector.body.epil ]
  %bin.rdx429 = add <4 x i32> %.lcssa, %.lcssa435
  %rdx.shuf430 = shufflevector <4 x i32> %bin.rdx429, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx431 = add <4 x i32> %bin.rdx429, %rdx.shuf430
  %rdx.shuf432 = shufflevector <4 x i32> %bin.rdx431, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx433 = add <4 x i32> %bin.rdx431, %rdx.shuf432
  %73 = extractelement <4 x i32> %bin.rdx433, i32 0
  %bin.rdx = add <4 x i32> %.lcssa436, %.lcssa437
  %rdx.shuf = shufflevector <4 x i32> %bin.rdx, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx426 = add <4 x i32> %bin.rdx, %rdx.shuf
  %rdx.shuf427 = shufflevector <4 x i32> %bin.rdx426, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx428 = add <4 x i32> %bin.rdx426, %rdx.shuf427
  %74 = extractelement <4 x i32> %bin.rdx428, i32 0
  %cmp.n = icmp eq i64 %sub.ptr.div.i263, %n.vec
  br i1 %cmp.n, label %invoke.cont65, label %for.body.preheader

lpad10.loopexit.split-lp.loopexit:                ; preds = %_ZNSt16allocator_traitsISaIiEE8allocateERS0_m.exit.i.i.i.i
  %lpad.loopexit = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup113

lpad10.loopexit.split-lp.loopexit.split-lp:       ; preds = %if.then.i.i.i.i.i.i183
  %lpad.loopexit.split-lp378 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup113

ehcleanup67.thread:                               ; preds = %_ZNSt16allocator_traitsISaIiEE8allocateERS0_m.exit.i.i.i.i226
  %75 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %worker_matches_count.sroa.13.0..sroa_cast289)
  br label %if.then.i.i.i246

pfor.detach30:                                    ; preds = %pfor.detach30.preheader, %pfor.inc
  %indvars.iv391 = phi i64 [ %indvars.iv.next392, %pfor.inc ], [ 0, %pfor.detach30.preheader ]
  detach within %syncreg18, label %pfor.body34, label %pfor.inc

pfor.body34:                                      ; preds = %pfor.detach30
  %76 = trunc i64 %indvars.iv391 to i32
  %call39 = invoke i32 @_Z15get_valid_movesi(i32 %76)
          to label %invoke.cont38 unwind label %lpad35

invoke.cont38:                                    ; preds = %pfor.body34
  %add.ptr.i248 = getelementptr inbounds i32, i32* %cond.i.i.i.i185332336342, i64 %indvars.iv391
  store i32 %call39, i32* %add.ptr.i248, align 4, !tbaa !14
  %call43 = invoke i32 @_Z17get_matches_counti(i32 %76)
          to label %invoke.cont42 unwind label %lpad35

invoke.cont42:                                    ; preds = %invoke.cont38
  %add.ptr.i255 = getelementptr inbounds i32, i32* %cond.i.i.i.i227343, i64 %indvars.iv391
  store i32 %call43, i32* %add.ptr.i255, align 4, !tbaa !14
  reattach within %syncreg18, label %pfor.inc

pfor.inc:                                         ; preds = %invoke.cont42, %pfor.detach30
  %indvars.iv.next392 = add nuw nsw i64 %indvars.iv391, 1
  %cmp27 = icmp slt i64 %indvars.iv.next392, %conv8
  br i1 %cmp27, label %pfor.detach30, label %pfor.cond.cleanup28, !llvm.loop !19

lpad35:                                           ; preds = %invoke.cont38, %pfor.body34
  %77 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv393 = phi i64 [ %indvars.iv.next394, %for.body ], [ %indvars.iv393.ph, %for.body.preheader ]
  %matches_count.0371 = phi i32 [ %add52, %for.body ], [ %matches_count.0371.ph, %for.body.preheader ]
  %valid_moves.0370 = phi i32 [ %add55, %for.body ], [ %valid_moves.0370.ph, %for.body.preheader ]
  %add.ptr.i278 = getelementptr inbounds i32, i32* %cond.i.i.i.i227343, i64 %indvars.iv393
  %78 = load i32, i32* %add.ptr.i278, align 4, !tbaa !14
  %add52 = add nsw i32 %78, %matches_count.0371
  %add.ptr.i276 = getelementptr inbounds i32, i32* %cond.i.i.i.i185332336342, i64 %indvars.iv393
  %79 = load i32, i32* %add.ptr.i276, align 4, !tbaa !14
  %add55 = add nsw i32 %79, %valid_moves.0370
  %indvars.iv.next394 = add nuw nsw i64 %indvars.iv393, 1
  %exitcond = icmp eq i64 %indvars.iv.next394, %sub.ptr.div.i263
  br i1 %exitcond, label %invoke.cont65, label %for.body, !llvm.loop !21

invoke.cont65:                                    ; preds = %for.body, %middle.block, %pfor.end.continue29
  %valid_moves.0.lcssa = phi i32 [ 0, %pfor.end.continue29 ], [ %73, %middle.block ], [ %add55, %for.body ]
  %matches_count.0.lcssa = phi i32 [ 0, %pfor.end.continue29 ], [ %74, %middle.block ], [ %add52, %for.body ]
  %call57 = call i64 @clock() #7
  %sub58 = sub nsw i64 %call57, %call7
  %conv59 = sitofp i64 %sub58 to double
  %div60 = fdiv double %conv59, 1.000000e+06
  %80 = bitcast %"class.std::tuple"* %ref.tmp63 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %80) #7
  %_M_head_impl.i.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %ref.tmp63, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store i32 %matches_count.0.lcssa, i32* %_M_head_impl.i.i.i.i.i.i, align 8, !tbaa !23, !alias.scope !25
  %81 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %ref.tmp63, i64 0, i32 0, i32 0, i32 1, i32 0
  store double %div60, double* %81, align 8, !tbaa !28, !alias.scope !25
  %82 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %ref.tmp63, i64 0, i32 0, i32 1, i32 0
  store i32 %valid_moves.0.lcssa, i32* %82, align 8, !tbaa !31, !alias.scope !25
  %_M_finish.i.i271 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %cond.i.i.i.i326, i64 %indvars.iv395, i32 0, i32 0, i32 1
  %83 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_finish.i.i271, align 8, !tbaa !12
  %_M_end_of_storage.i.i272 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %cond.i.i.i.i326, i64 %indvars.iv395, i32 0, i32 0, i32 2
  %84 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_end_of_storage.i.i272, align 8, !tbaa !13
  %cmp.i.i = icmp eq %"class.std::tuple"* %83, %84
  br i1 %cmp.i.i, label %if.else.i.i, label %if.then.i.i273

if.then.i.i273:                                   ; preds = %invoke.cont65
  %_M_head_impl.i.i6.i.i.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %83, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store i32 %matches_count.0.lcssa, i32* %_M_head_impl.i.i6.i.i.i.i.i.i.i, align 4, !tbaa !23
  %85 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %83, i64 0, i32 0, i32 0, i32 1, i32 0
  store double %div60, double* %85, align 8, !tbaa !28
  %86 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %83, i64 0, i32 0, i32 1, i32 0
  %87 = load i32, i32* %82, align 8, !tbaa !14
  store i32 %87, i32* %86, align 4, !tbaa !31
  %incdec.ptr.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %83, i64 1
  store %"class.std::tuple"* %incdec.ptr.i.i, %"class.std::tuple"** %_M_finish.i.i271, align 8, !tbaa !12
  br label %invoke.cont66

if.else.i.i:                                      ; preds = %invoke.cont65
  %add.ptr.i270 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %cond.i.i.i.i326, i64 %indvars.iv395
  invoke void @_ZNSt6vectorISt5tupleIJidiEESaIS1_EE19_M_emplace_back_auxIJS1_EEEvDpOT_(%"class.std::vector.0"* nonnull %add.ptr.i270, %"class.std::tuple"* nonnull dereferenceable(24) %ref.tmp63)
          to label %invoke.cont66 unwind label %lpad64

invoke.cont66:                                    ; preds = %if.then.i.i273, %if.else.i.i
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %80) #7
  %tobool.i.i.i265 = icmp eq i32* %cond.i.i.i.i227343, null
  br i1 %tobool.i.i.i265, label %_ZNSt6vectorIiSaIiEED2Ev.exit268, label %if.then.i.i.i267

if.then.i.i.i267:                                 ; preds = %invoke.cont66
  call void @_ZdlPv(i8* %30) #7
  br label %_ZNSt6vectorIiSaIiEED2Ev.exit268

_ZNSt6vectorIiSaIiEED2Ev.exit268:                 ; preds = %invoke.cont66, %if.then.i.i.i267
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %worker_matches_count.sroa.13.0..sroa_cast289337341)
  %tobool.i.i.i257 = icmp eq i32* %cond.i.i.i.i185332336342, null
  br i1 %tobool.i.i.i257, label %_ZNSt6vectorIiSaIiEED2Ev.exit260, label %if.then.i.i.i259

if.then.i.i.i259:                                 ; preds = %_ZNSt6vectorIiSaIiEED2Ev.exit268
  call void @_ZdlPv(i8* %31) #7
  br label %_ZNSt6vectorIiSaIiEED2Ev.exit260

_ZNSt6vectorIiSaIiEED2Ev.exit260:                 ; preds = %_ZNSt6vectorIiSaIiEED2Ev.exit268, %if.then.i.i.i259
  reattach within %syncreg, label %pfor.inc78

pfor.inc78:                                       ; preds = %_ZNSt6vectorIiSaIiEED2Ev.exit260, %pfor.detach
  %indvars.iv.next396 = add nuw nsw i64 %indvars.iv395, 1
  %cmp = icmp slt i64 %indvars.iv.next396, %18
  br i1 %cmp, label %pfor.detach, label %pfor.cond.cleanup, !llvm.loop !33

lpad64:                                           ; preds = %if.else.i.i
  %88 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %80) #7
  br label %ehcleanup

ehcleanup:                                        ; preds = %lpad64, %lpad35
  %tobool.i.i.i250 = icmp eq i32* %cond.i.i.i.i227343, null
  br i1 %tobool.i.i.i250, label %ehcleanup67, label %if.then.i.i.i252

if.then.i.i.i252:                                 ; preds = %ehcleanup
  call void @_ZdlPv(i8* %30) #7
  br label %ehcleanup67

ehcleanup67:                                      ; preds = %if.then.i.i.i252, %ehcleanup
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %worker_matches_count.sroa.13.0..sroa_cast289337341)
  %tobool.i.i.i244 = icmp eq i32* %cond.i.i.i.i185332336342, null
  br i1 %tobool.i.i.i244, label %ehcleanup113, label %if.then.i.i.i246

if.then.i.i.i246:                                 ; preds = %ehcleanup67.thread, %ehcleanup67
  %89 = phi i8* [ %call2.i.i.i.i3.i.i193, %ehcleanup67.thread ], [ %31, %ehcleanup67 ]
  call void @_ZdlPv(i8* %89) #7
  br label %ehcleanup113

for.cond.cleanup89:                               ; preds = %for.cond.cleanup97, %pfor.end.continue
  %cmp3.i.i.i.i205 = icmp eq %"class.std::vector.0"* %cond.i.i.i.i326, %__cur.0.lcssa.i.i.i.i.i
  br i1 %cmp3.i.i.i.i205, label %invoke.cont.i218, label %for.body.i.i.i.i210.preheader

for.body.i.i.i.i210.preheader:                    ; preds = %for.cond.cleanup89
  br label %for.body.i.i.i.i210

for.body.i.i.i.i210:                              ; preds = %for.body.i.i.i.i210.preheader, %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i214
  %__first.addr.04.i.i.i.i207 = phi %"class.std::vector.0"* [ %incdec.ptr.i.i.i.i212, %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i214 ], [ %cond.i.i.i.i326, %for.body.i.i.i.i210.preheader ]
  %_M_start.i.i.i.i.i.i.i208 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %__first.addr.04.i.i.i.i207, i64 0, i32 0, i32 0, i32 0
  %90 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_start.i.i.i.i.i.i.i208, align 8, !tbaa !9
  %tobool.i.i.i.i.i.i.i.i209 = icmp eq %"class.std::tuple"* %90, null
  br i1 %tobool.i.i.i.i.i.i.i.i209, label %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i214, label %if.then.i.i.i.i.i.i.i.i211

if.then.i.i.i.i.i.i.i.i211:                       ; preds = %for.body.i.i.i.i210
  %91 = bitcast %"class.std::tuple"* %90 to i8*
  call void @_ZdlPv(i8* %91) #7
  br label %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i214

_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i214: ; preds = %if.then.i.i.i.i.i.i.i.i211, %for.body.i.i.i.i210
  %incdec.ptr.i.i.i.i212 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %__first.addr.04.i.i.i.i207, i64 1
  %cmp.i.i.i.i213 = icmp eq %"class.std::vector.0"* %incdec.ptr.i.i.i.i212, %__cur.0.lcssa.i.i.i.i.i
  br i1 %cmp.i.i.i.i213, label %invoke.cont.i218, label %for.body.i.i.i.i210

invoke.cont.i218:                                 ; preds = %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i214, %for.cond.cleanup89
  %tobool.i.i.i217 = icmp eq %"class.std::vector.0"* %cond.i.i.i.i326, null
  br i1 %tobool.i.i.i217, label %_ZNSt6vectorIS_ISt5tupleIJidiEESaIS1_EESaIS3_EED2Ev.exit220, label %if.then.i.i.i219

if.then.i.i.i219:                                 ; preds = %invoke.cont.i218
  call void @_ZdlPv(i8* %16) #7
  br label %_ZNSt6vectorIS_ISt5tupleIJidiEESaIS1_EESaIS3_EED2Ev.exit220

_ZNSt6vectorIS_ISt5tupleIJidiEESaIS1_EESaIS3_EED2Ev.exit220: ; preds = %invoke.cont.i218, %if.then.i.i.i219
  ret void

for.body90:                                       ; preds = %for.body90.lr.ph, %for.cond.cleanup97
  %indvars.iv389 = phi i64 [ 0, %for.body90.lr.ph ], [ %indvars.iv.next390, %for.cond.cleanup97 ]
  %add.ptr.i202 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %cond.i.i.i.i326, i64 %indvars.iv389
  %_M_finish.i198 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %cond.i.i.i.i326, i64 %indvars.iv389, i32 0, i32 0, i32 1
  %92 = bitcast %"class.std::tuple"** %_M_finish.i198 to i64*
  %93 = load i64, i64* %92, align 8, !tbaa !12
  %94 = bitcast %"class.std::vector.0"* %add.ptr.i202 to i64*
  %95 = load i64, i64* %94, align 8, !tbaa !9
  %96 = icmp eq i64 %93, %95
  br i1 %96, label %for.cond.cleanup97, label %for.body98.preheader

for.body98.preheader:                             ; preds = %for.body90
  br label %for.body98

for.cond.cleanup97:                               ; preds = %for.inc105, %for.body90
  %indvars.iv.next390 = add nuw i64 %indvars.iv389, 1
  %cmp88 = icmp ugt i64 %sub.ptr.div.i242, %indvars.iv.next390
  br i1 %cmp88, label %for.body90, label %for.cond.cleanup89

for.body98:                                       ; preds = %for.body98.preheader, %for.inc105
  %.in = phi i64 [ %104, %for.inc105 ], [ %95, %for.body98.preheader ]
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc105 ], [ 0, %for.body98.preheader ]
  %97 = inttoptr i64 %.in to %"class.std::tuple"*
  %add.ptr.i195 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %97, i64 %indvars.iv
  %98 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_finish.i175, align 8, !tbaa !12
  %99 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_end_of_storage.i.i, align 8, !tbaa !13
  %cmp.i176 = icmp eq %"class.std::tuple"* %98, %99
  br i1 %cmp.i176, label %if.else.i, label %if.then.i177

if.then.i177:                                     ; preds = %for.body98
  %100 = bitcast %"class.std::tuple"* %98 to i8*
  %101 = bitcast %"class.std::tuple"* %add.ptr.i195 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %100, i8* nonnull %101, i64 24, i32 8, i1 false) #7
  %102 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_finish.i175, align 8, !tbaa !12
  %incdec.ptr.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %102, i64 1
  store %"class.std::tuple"* %incdec.ptr.i, %"class.std::tuple"** %_M_finish.i175, align 8, !tbaa !12
  br label %for.inc105

if.else.i:                                        ; preds = %for.body98
  invoke void @_ZNSt6vectorISt5tupleIJidiEESaIS1_EE19_M_emplace_back_auxIJRKS1_EEEvDpOT_(%"class.std::vector.0"* nonnull %agg.result, %"class.std::tuple"* nonnull dereferenceable(24) %add.ptr.i195)
          to label %for.inc105 unwind label %lpad103

for.inc105:                                       ; preds = %if.then.i177, %if.else.i
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %103 = load i64, i64* %92, align 8, !tbaa !12
  %104 = load i64, i64* %94, align 8, !tbaa !9
  %sub.ptr.sub.i199 = sub i64 %103, %104
  %sub.ptr.div.i200 = sdiv exact i64 %sub.ptr.sub.i199, 24
  %cmp96 = icmp ugt i64 %sub.ptr.div.i200, %indvars.iv.next
  br i1 %cmp96, label %for.body98, label %for.cond.cleanup97

lpad103:                                          ; preds = %if.else.i
  %105 = landingpad { i8*, i32 }
          cleanup
  %106 = extractvalue { i8*, i32 } %105, 0
  %107 = extractvalue { i8*, i32 } %105, 1
  br label %ehcleanup113

ehcleanup113:                                     ; preds = %lpad10.loopexit.split-lp.loopexit, %lpad10.loopexit.split-lp.loopexit.split-lp, %ehcleanup67, %if.then.i.i.i246, %lpad103
  %ehselector.slot.0 = phi i32 [ %107, %lpad103 ], [ undef, %if.then.i.i.i246 ], [ undef, %ehcleanup67 ], [ undef, %lpad10.loopexit.split-lp.loopexit.split-lp ], [ undef, %lpad10.loopexit.split-lp.loopexit ]
  %exn.slot.0 = phi i8* [ %106, %lpad103 ], [ undef, %if.then.i.i.i246 ], [ undef, %ehcleanup67 ], [ undef, %lpad10.loopexit.split-lp.loopexit.split-lp ], [ undef, %lpad10.loopexit.split-lp.loopexit ]
  %cmp3.i.i.i.i = icmp eq %"class.std::vector.0"* %cond.i.i.i.i326, %__cur.0.lcssa.i.i.i.i.i
  br i1 %cmp3.i.i.i.i, label %invoke.cont.i, label %for.body.i.i.i.i.preheader

for.body.i.i.i.i.preheader:                       ; preds = %ehcleanup113
  br label %for.body.i.i.i.i

for.body.i.i.i.i:                                 ; preds = %for.body.i.i.i.i.preheader, %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i
  %__first.addr.04.i.i.i.i = phi %"class.std::vector.0"* [ %incdec.ptr.i.i.i.i, %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i ], [ %cond.i.i.i.i326, %for.body.i.i.i.i.preheader ]
  %_M_start.i.i.i.i.i.i.i = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %__first.addr.04.i.i.i.i, i64 0, i32 0, i32 0, i32 0
  %108 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_start.i.i.i.i.i.i.i, align 8, !tbaa !9
  %tobool.i.i.i.i.i.i.i.i = icmp eq %"class.std::tuple"* %108, null
  br i1 %tobool.i.i.i.i.i.i.i.i, label %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i, label %if.then.i.i.i.i.i.i.i.i

if.then.i.i.i.i.i.i.i.i:                          ; preds = %for.body.i.i.i.i
  %109 = bitcast %"class.std::tuple"* %108 to i8*
  call void @_ZdlPv(i8* %109) #7
  br label %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i

_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i: ; preds = %if.then.i.i.i.i.i.i.i.i, %for.body.i.i.i.i
  %incdec.ptr.i.i.i.i = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %__first.addr.04.i.i.i.i, i64 1
  %cmp.i.i.i.i = icmp eq %"class.std::vector.0"* %incdec.ptr.i.i.i.i, %__cur.0.lcssa.i.i.i.i.i
  br i1 %cmp.i.i.i.i, label %invoke.cont.i, label %for.body.i.i.i.i

invoke.cont.i:                                    ; preds = %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i, %ehcleanup113
  %tobool.i.i.i163 = icmp eq %"class.std::vector.0"* %cond.i.i.i.i326, null
  br i1 %tobool.i.i.i163, label %ehcleanup116, label %if.then.i.i.i164

if.then.i.i.i164:                                 ; preds = %invoke.cont.i
  call void @_ZdlPv(i8* %16) #7
  br label %ehcleanup116

ehcleanup116:                                     ; preds = %lpad3, %invoke.cont.i, %if.then.i.i.i164, %lpad
  %ehselector.slot.2 = phi i32 [ %22, %lpad ], [ %25, %lpad3 ], [ %ehselector.slot.0, %invoke.cont.i ], [ %ehselector.slot.0, %if.then.i.i.i164 ]
  %exn.slot.2 = phi i8* [ %21, %lpad ], [ %24, %lpad3 ], [ %exn.slot.0, %invoke.cont.i ], [ %exn.slot.0, %if.then.i.i.i164 ]
  %_M_start.i.i = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %agg.result, i64 0, i32 0, i32 0, i32 0
  %110 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_start.i.i, align 8, !tbaa !9
  %tobool.i.i.i = icmp eq %"class.std::tuple"* %110, null
  br i1 %tobool.i.i.i, label %_ZNSt6vectorISt5tupleIJidiEESaIS1_EED2Ev.exit, label %if.then.i.i.i

if.then.i.i.i:                                    ; preds = %ehcleanup116
  %111 = bitcast %"class.std::tuple"* %110 to i8*
  call void @_ZdlPv(i8* %111) #7
  br label %_ZNSt6vectorISt5tupleIJidiEESaIS1_EED2Ev.exit

_ZNSt6vectorISt5tupleIJidiEESaIS1_EED2Ev.exit:    ; preds = %ehcleanup116, %if.then.i.i.i
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn.slot.2, 0
  %lpad.val117 = insertvalue { i8*, i32 } %lpad.val, i32 %ehselector.slot.2, 1
  resume { i8*, i32 } %lpad.val117
}

declare i32 @__gxx_personality_v0(...)

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

; Function Attrs: uwtable
define linkonce_odr void @_ZNSt6vectorISt5tupleIJidiEESaIS1_EE19_M_emplace_back_auxIJS1_EEEvDpOT_(%"class.std::vector.0"* %this, %"class.std::tuple"* dereferenceable(24) %__args) local_unnamed_addr #0 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %_M_finish.i20.i = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %this, i64 0, i32 0, i32 0, i32 1
  %0 = bitcast %"class.std::tuple"** %_M_finish.i20.i to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !12
  %2 = bitcast %"class.std::vector.0"* %this to i64*
  %3 = load i64, i64* %2, align 8, !tbaa !9
  %sub.ptr.sub.i21.i = sub i64 %1, %3
  %sub.ptr.div.i22.i = sdiv exact i64 %sub.ptr.sub.i21.i, 24
  %4 = icmp eq i64 %sub.ptr.sub.i21.i, 0
  %.sroa.speculated.i = select i1 %4, i64 1, i64 %sub.ptr.div.i22.i
  %add.i = add nsw i64 %.sroa.speculated.i, %sub.ptr.div.i22.i
  %cmp7.i = icmp ult i64 %add.i, %sub.ptr.div.i22.i
  %cmp9.i = icmp ugt i64 %add.i, 768614336404564650
  %or.cond.i = or i1 %cmp7.i, %cmp9.i
  %cond.i = select i1 %or.cond.i, i64 768614336404564650, i64 %add.i
  %cmp.i.i.i = icmp ugt i64 %cond.i, 768614336404564650
  br i1 %cmp.i.i.i, label %if.then.i.i.i, label %_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i

if.then.i.i.i:                                    ; preds = %entry
  tail call void @_ZSt17__throw_bad_allocv() #8
  unreachable

_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i: ; preds = %entry
  %mul.i.i.i = mul i64 %cond.i, 24
  %call2.i.i.i = tail call i8* @_Znwm(i64 %mul.i.i.i)
  %5 = bitcast i8* %call2.i.i.i to %"class.std::tuple"*
  %6 = load i64, i64* %0, align 8, !tbaa !12
  %7 = load i64, i64* %2, align 8, !tbaa !9
  %sub.ptr.sub.i = sub i64 %6, %7
  %sub.ptr.div.i = sdiv exact i64 %sub.ptr.sub.i, 24
  %_M_head_impl.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__args, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %_M_head_impl.i.i6.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %5, i64 %sub.ptr.div.i, i32 0, i32 0, i32 0, i32 0, i32 0
  %8 = load i32, i32* %_M_head_impl.i.i.i.i.i.i.i.i, align 4, !tbaa !14
  store i32 %8, i32* %_M_head_impl.i.i6.i.i.i.i.i, align 4, !tbaa !23
  %9 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %5, i64 %sub.ptr.div.i, i32 0, i32 0, i32 1
  %10 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__args, i64 0, i32 0, i32 0, i32 1, i32 0
  %11 = bitcast double* %10 to i64*
  %12 = load i64, i64* %11, align 8, !tbaa !34
  %13 = bitcast %"struct.std::_Head_base.7"* %9 to i64*
  store i64 %12, i64* %13, align 8, !tbaa !28
  %14 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__args, i64 0, i32 0, i32 1, i32 0
  %15 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %5, i64 %sub.ptr.div.i, i32 0, i32 1, i32 0
  %16 = load i32, i32* %14, align 4, !tbaa !14
  store i32 %16, i32* %15, align 4, !tbaa !31
  %17 = inttoptr i64 %7 to %"class.std::tuple"*
  %18 = inttoptr i64 %6 to %"class.std::tuple"*
  %cmp.i.i21.i.i.i.i = icmp eq %"class.std::tuple"* %17, %18
  br i1 %cmp.i.i21.i.i.i.i, label %invoke.cont8, label %for.body.i.i.i.i.preheader

for.body.i.i.i.i.preheader:                       ; preds = %_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i
  br label %for.body.i.i.i.i

for.body.i.i.i.i:                                 ; preds = %for.body.i.i.i.i.preheader, %for.body.i.i.i.i
  %__cur.023.i.i.i.i = phi %"class.std::tuple"* [ %incdec.ptr.i.i.i.i, %for.body.i.i.i.i ], [ %5, %for.body.i.i.i.i.preheader ]
  %__first.sroa.0.022.i.i.i.i = phi %"class.std::tuple"* [ %incdec.ptr.i.i.i.i.i, %for.body.i.i.i.i ], [ %17, %for.body.i.i.i.i.preheader ]
  %_M_head_impl.i.i.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.022.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %_M_head_impl.i.i6.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.023.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %19 = load i32, i32* %_M_head_impl.i.i.i.i.i.i.i.i.i.i.i, align 4, !tbaa !14
  store i32 %19, i32* %_M_head_impl.i.i6.i.i.i.i.i.i.i.i, align 4, !tbaa !23
  %20 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.023.i.i.i.i, i64 0, i32 0, i32 0, i32 1
  %21 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.022.i.i.i.i, i64 0, i32 0, i32 0, i32 1, i32 0
  %22 = bitcast double* %21 to i64*
  %23 = load i64, i64* %22, align 8, !tbaa !34
  %24 = bitcast %"struct.std::_Head_base.7"* %20 to i64*
  store i64 %23, i64* %24, align 8, !tbaa !28
  %25 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.022.i.i.i.i, i64 0, i32 0, i32 1, i32 0
  %26 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.023.i.i.i.i, i64 0, i32 0, i32 1, i32 0
  %27 = load i32, i32* %25, align 4, !tbaa !14
  store i32 %27, i32* %26, align 4, !tbaa !31
  %incdec.ptr.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.022.i.i.i.i, i64 1
  %incdec.ptr.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.023.i.i.i.i, i64 1
  %cmp.i.i.i.i.i.i = icmp eq %"class.std::tuple"* %incdec.ptr.i.i.i.i.i, %18
  br i1 %cmp.i.i.i.i.i.i, label %invoke.cont8, label %for.body.i.i.i.i

invoke.cont8:                                     ; preds = %for.body.i.i.i.i, %_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i
  %__cur.0.lcssa.i.i.i.i = phi %"class.std::tuple"* [ %5, %_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i ], [ %incdec.ptr.i.i.i.i, %for.body.i.i.i.i ]
  %incdec.ptr = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.0.lcssa.i.i.i.i, i64 1
  %_M_end_of_storage = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %this, i64 0, i32 0, i32 0, i32 2
  %tobool.i62 = icmp eq i64 %7, 0
  br i1 %tobool.i62, label %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit64, label %if.then.i63

if.then.i63:                                      ; preds = %invoke.cont8
  %28 = inttoptr i64 %7 to i8*
  tail call void @_ZdlPv(i8* %28) #7
  br label %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit64

_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit64: ; preds = %invoke.cont8, %if.then.i63
  %29 = bitcast %"class.std::vector.0"* %this to i8**
  store i8* %call2.i.i.i, i8** %29, align 8, !tbaa !9
  store %"class.std::tuple"* %incdec.ptr, %"class.std::tuple"** %_M_finish.i20.i, align 8, !tbaa !12
  %add.ptr33 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %5, i64 %cond.i
  store %"class.std::tuple"* %add.ptr33, %"class.std::tuple"** %_M_end_of_storage, align 8, !tbaa !13
  ret void
}

; Function Attrs: uwtable
define linkonce_odr void @_ZNSt6vectorISt5tupleIJidiEESaIS1_EE19_M_emplace_back_auxIJRKS1_EEEvDpOT_(%"class.std::vector.0"* %this, %"class.std::tuple"* dereferenceable(24) %__args) local_unnamed_addr #0 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %_M_finish.i20.i = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %this, i64 0, i32 0, i32 0, i32 1
  %0 = bitcast %"class.std::tuple"** %_M_finish.i20.i to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !12
  %2 = bitcast %"class.std::vector.0"* %this to i64*
  %3 = load i64, i64* %2, align 8, !tbaa !9
  %sub.ptr.sub.i21.i = sub i64 %1, %3
  %sub.ptr.div.i22.i = sdiv exact i64 %sub.ptr.sub.i21.i, 24
  %4 = icmp eq i64 %sub.ptr.sub.i21.i, 0
  %.sroa.speculated.i = select i1 %4, i64 1, i64 %sub.ptr.div.i22.i
  %add.i = add nsw i64 %.sroa.speculated.i, %sub.ptr.div.i22.i
  %cmp7.i = icmp ult i64 %add.i, %sub.ptr.div.i22.i
  %cmp9.i = icmp ugt i64 %add.i, 768614336404564650
  %or.cond.i = or i1 %cmp7.i, %cmp9.i
  %cond.i = select i1 %or.cond.i, i64 768614336404564650, i64 %add.i
  %cmp.i56 = icmp eq i64 %cond.i, 0
  br i1 %cmp.i56, label %invoke.cont, label %cond.true.i

cond.true.i:                                      ; preds = %entry
  %cmp.i.i.i = icmp ugt i64 %cond.i, 768614336404564650
  br i1 %cmp.i.i.i, label %if.then.i.i.i, label %_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i

if.then.i.i.i:                                    ; preds = %cond.true.i
  tail call void @_ZSt17__throw_bad_allocv() #8
  unreachable

_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i: ; preds = %cond.true.i
  %mul.i.i.i = mul i64 %cond.i, 24
  %call2.i.i.i = tail call i8* @_Znwm(i64 %mul.i.i.i)
  %5 = bitcast i8* %call2.i.i.i to %"class.std::tuple"*
  %.pre = load i64, i64* %0, align 8, !tbaa !12
  %.pre68 = load i64, i64* %2, align 8, !tbaa !9
  br label %invoke.cont

invoke.cont:                                      ; preds = %_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i, %entry
  %.in = phi i64 [ %.pre, %_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i ], [ %1, %entry ]
  %.in69 = phi i64 [ %.pre68, %_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i ], [ %3, %entry ]
  %cond.i57 = phi %"class.std::tuple"* [ %5, %_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i ], [ null, %entry ]
  %6 = inttoptr i64 %.in69 to %"class.std::tuple"*
  %7 = inttoptr i64 %.in to %"class.std::tuple"*
  %sub.ptr.sub.i = sub i64 %.in, %.in69
  %sub.ptr.div.i = sdiv exact i64 %sub.ptr.sub.i, 24
  %add.ptr = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %cond.i57, i64 %sub.ptr.div.i
  %8 = bitcast %"class.std::tuple"* %add.ptr to i8*
  %9 = bitcast %"class.std::tuple"* %__args to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %8, i8* nonnull %9, i64 24, i32 8, i1 false) #7
  %_M_start = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %this, i64 0, i32 0, i32 0, i32 0
  %cmp.i.i21.i.i.i.i = icmp eq %"class.std::tuple"* %6, %7
  br i1 %cmp.i.i21.i.i.i.i, label %invoke.cont8, label %for.body.i.i.i.i.preheader

for.body.i.i.i.i.preheader:                       ; preds = %invoke.cont
  br label %for.body.i.i.i.i

for.body.i.i.i.i:                                 ; preds = %for.body.i.i.i.i.preheader, %for.body.i.i.i.i
  %__cur.023.i.i.i.i = phi %"class.std::tuple"* [ %incdec.ptr.i.i.i.i, %for.body.i.i.i.i ], [ %cond.i57, %for.body.i.i.i.i.preheader ]
  %__first.sroa.0.022.i.i.i.i = phi %"class.std::tuple"* [ %incdec.ptr.i.i.i.i.i, %for.body.i.i.i.i ], [ %6, %for.body.i.i.i.i.preheader ]
  %_M_head_impl.i.i.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.022.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %_M_head_impl.i.i6.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.023.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %10 = load i32, i32* %_M_head_impl.i.i.i.i.i.i.i.i.i.i.i, align 4, !tbaa !14
  store i32 %10, i32* %_M_head_impl.i.i6.i.i.i.i.i.i.i.i, align 4, !tbaa !23
  %11 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.023.i.i.i.i, i64 0, i32 0, i32 0, i32 1
  %12 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.022.i.i.i.i, i64 0, i32 0, i32 0, i32 1, i32 0
  %13 = bitcast double* %12 to i64*
  %14 = load i64, i64* %13, align 8, !tbaa !34
  %15 = bitcast %"struct.std::_Head_base.7"* %11 to i64*
  store i64 %14, i64* %15, align 8, !tbaa !28
  %16 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.022.i.i.i.i, i64 0, i32 0, i32 1, i32 0
  %17 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.023.i.i.i.i, i64 0, i32 0, i32 1, i32 0
  %18 = load i32, i32* %16, align 4, !tbaa !14
  store i32 %18, i32* %17, align 4, !tbaa !31
  %incdec.ptr.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.022.i.i.i.i, i64 1
  %incdec.ptr.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.023.i.i.i.i, i64 1
  %cmp.i.i.i.i.i.i = icmp eq %"class.std::tuple"* %incdec.ptr.i.i.i.i.i, %7
  br i1 %cmp.i.i.i.i.i.i, label %invoke.cont8, label %for.body.i.i.i.i

invoke.cont8:                                     ; preds = %for.body.i.i.i.i, %invoke.cont
  %__cur.0.lcssa.i.i.i.i = phi %"class.std::tuple"* [ %cond.i57, %invoke.cont ], [ %incdec.ptr.i.i.i.i, %for.body.i.i.i.i ]
  %incdec.ptr = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.0.lcssa.i.i.i.i, i64 1
  %_M_end_of_storage = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %this, i64 0, i32 0, i32 0, i32 2
  %tobool.i62 = icmp eq i64 %.in69, 0
  br i1 %tobool.i62, label %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit64, label %if.then.i63

if.then.i63:                                      ; preds = %invoke.cont8
  %19 = inttoptr i64 %.in69 to i8*
  tail call void @_ZdlPv(i8* %19) #7
  br label %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit64

_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit64: ; preds = %invoke.cont8, %if.then.i63
  store %"class.std::tuple"* %cond.i57, %"class.std::tuple"** %_M_start, align 8, !tbaa !9
  store %"class.std::tuple"* %incdec.ptr, %"class.std::tuple"** %_M_finish.i20.i, align 8, !tbaa !12
  %add.ptr33 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %cond.i57, i64 %cond.i
  store %"class.std::tuple"* %add.ptr33, %"class.std::tuple"** %_M_end_of_storage, align 8, !tbaa !13
  ret void
}

; LS-LABEL: define internal fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE_pfor.detach.ls(%"class.std::vector.0"* noalias sret align 8 %agg.result.ls, i64 %indvars.iv395.start.ls, i64 %end.ls, i64 %.ls,
; LS: {{^.split:}}
; LS-NEXT: call fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE_pfor.detach.ls(%"class.std::vector.0"* %agg.result.ls, i64 %indvars.iv395.ls.dac, i64 %miditer, i64 %.ls,

; LS: {{^pfor.detach30.preheader.ls:}}
; LS: call fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE_pfor.detach.ls_pfor.detach30.ls.ls(%"class.std::vector.0"* %agg.result.ls, i64 0,

; LS-LABEL: define internal fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE_pfor.detach.ls_pfor.detach30.ls.ls(%"class.std::vector.0"* noalias sret align 8 %agg.result.ls.ls, i64 %indvars.iv391.ls.start.ls, i64 %end.ls, i64 %.ls,
; LS: {{^.split:}}
; LS: call fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE_pfor.detach.ls_pfor.detach30.ls.ls(%"class.std::vector.0"* %agg.result.ls.ls, i64 %indvars.iv391.ls.ls.dac, i64 %miditer, i64 %.ls,

; TT-LABEL: define internal fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE_pfor.body.cilk(%"class.std::vector.0"* noalias sret align 8 %agg.result.cilk,
; TT: {{^pfor.detach30.cilk.split:}}
; TT-NEXT: call fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE_pfor.body.cilk_pfor.body34.cilk.cilk(%"class.std::vector.0"* %agg.result.cilk,

; TT-LABEL: define internal fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE_pfor.body.cilk_pfor.body34.cilk.cilk(%"class.std::vector.0"* noalias sret align 8 %agg.result.cilk.cilk,

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #1

attributes #0 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nobuiltin nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noreturn "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nobuiltin "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nounwind }
attributes #8 = { noreturn }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (git@github.com:wsmoses/Tapir-Clang.git 245c29d5cb99796c4107fd83f9bbe668c130b275) (git@github.com:wsmoses/Tapir-LLVM.git 7352407d063c8bac796926ca618e14d8eca87735)"}
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
!16 = distinct !{!16, !17, !18}
!17 = !{!"llvm.loop.vectorize.width", i32 1}
!18 = !{!"llvm.loop.interleave.count", i32 1}
!19 = distinct !{!19, !20}
!20 = !{!"tapir.loop.spawn.strategy", i32 1}
!21 = distinct !{!21, !22, !17, !18}
!22 = !{!"llvm.loop.unroll.runtime.disable"}
!23 = !{!24, !15, i64 0}
!24 = !{!"_ZTSSt10_Head_baseILm2EiLb0EE", !15, i64 0}
!25 = !{!26}
!26 = distinct !{!26, !27, !"_ZSt10make_tupleIJRiRdS0_EESt5tupleIJDpNSt17__decay_and_stripIT_E6__typeEEEDpOS4_: %agg.result"}
!27 = distinct !{!27, !"_ZSt10make_tupleIJRiRdS0_EESt5tupleIJDpNSt17__decay_and_stripIT_E6__typeEEEDpOS4_"}
!28 = !{!29, !30, i64 0}
!29 = !{!"_ZTSSt10_Head_baseILm1EdLb0EE", !30, i64 0}
!30 = !{!"double", !6, i64 0}
!31 = !{!32, !15, i64 0}
!32 = !{!"_ZTSSt10_Head_baseILm0EiLb0EE", !15, i64 0}
!33 = distinct !{!33, !20}
!34 = !{!30, !30, i64 0}
