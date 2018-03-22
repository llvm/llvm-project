; Test to verify that LoopSpawning creates a helper function that places any
; sret parameters at the begin of the argument list.
;
; Credit to Tim Kaler for producing the source code that inspired this test
; case.
;
; RUN: opt < %s -loop-spawning -simplifycfg -S | FileCheck %s --check-prefix=LS
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

$_ZNSt6vectorISt5tupleIJidiEESaIS1_EE17_M_realloc_insertIJS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_ = comdat any

@.str = private unnamed_addr constant [16 x i8] c"vector::reserve\00", align 1

; Function Attrs: uwtable
define void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE(%"class.std::vector.0"* noalias sret %agg.result, i32 %trials, double %threshold, %"class.std::vector"* nocapture readonly dereferenceable(24) %ps) local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %0 = bitcast %"class.std::vector.0"* %agg.result to i8*
  tail call void @llvm.memset.p0i8.i64(i8* %0, i8 0, i64 24, i32 8, i1 false) #8
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
  invoke void @_ZSt20__throw_length_errorPKc(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str, i64 0, i64 0)) #9
          to label %.noexc unwind label %lpad

.noexc:                                           ; preds = %if.then.i
  unreachable

if.end.i:                                         ; preds = %entry
  %_M_end_of_storage.i.i = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %agg.result, i64 0, i32 0, i32 0, i32 2
  %5 = icmp eq i64 %sub.ptr.sub.i, 0
  br i1 %5, label %pfor.cond.cleanup, label %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE11_M_allocateEm.exit.i.i

_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE11_M_allocateEm.exit.i.i: ; preds = %if.end.i
  %_M_finish.i.i = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %agg.result, i64 0, i32 0, i32 0, i32 1
  %call2.i.i.i.i.i139 = invoke i8* @_Znwm(i64 %sub.ptr.sub.i)
          to label %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE20_M_allocate_and_copyISt13move_iteratorIPS1_EEES6_mT_S8_.exit.i unwind label %lpad

_ZNSt6vectorISt5tupleIJidiEESaIS1_EE20_M_allocate_and_copyISt13move_iteratorIPS1_EEES6_mT_S8_.exit.i: ; preds = %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE11_M_allocateEm.exit.i.i
  %_M_start.i138 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %agg.result, i64 0, i32 0, i32 0, i32 0
  %6 = bitcast i8* %call2.i.i.i.i.i139 to %"class.std::tuple"*
  %7 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_start.i138, align 8, !tbaa !9
  %tobool.i.i = icmp eq %"class.std::tuple"* %7, null
  br i1 %tobool.i.i, label %invoke.cont, label %if.then.i.i

if.then.i.i:                                      ; preds = %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE20_M_allocate_and_copyISt13move_iteratorIPS1_EEES6_mT_S8_.exit.i
  %8 = bitcast %"class.std::tuple"* %7 to i8*
  tail call void @_ZdlPv(i8* %8) #8
  br label %invoke.cont

invoke.cont:                                      ; preds = %_ZNSt6vectorISt5tupleIJidiEESaIS1_EE20_M_allocate_and_copyISt13move_iteratorIPS1_EEES6_mT_S8_.exit.i, %if.then.i.i
  %9 = bitcast %"class.std::vector.0"* %agg.result to i8**
  store i8* %call2.i.i.i.i.i139, i8** %9, align 8, !tbaa !9
  %10 = bitcast %"class.std::tuple"** %_M_finish.i.i to i8**
  store i8* %call2.i.i.i.i.i139, i8** %10, align 8, !tbaa !12
  %add.ptr30.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %6, i64 %sub.ptr.div.i
  store %"class.std::tuple"* %add.ptr30.i, %"class.std::tuple"** %_M_end_of_storage.i.i, align 8, !tbaa !13
  %.pre = load i64, i64* %1, align 8, !tbaa !2
  %.pre358 = load i64, i64* %3, align 8, !tbaa !8
  %.pre359 = sub i64 %.pre, %.pre358
  %.pre360 = sdiv exact i64 %.pre359, 24
  %conv = trunc i64 %.pre360 to i32
  %sext = shl i64 %.pre360, 32
  %conv2 = ashr exact i64 %sext, 32
  %cmp.i.i.i.i143 = icmp eq i64 %sext, 0
  br i1 %cmp.i.i.i.i143, label %invoke.cont4, label %cond.true.i.i.i.i

cond.true.i.i.i.i:                                ; preds = %invoke.cont
  %cmp.i.i.i.i.i.i = icmp ugt i64 %conv2, 768614336404564650
  br i1 %cmp.i.i.i.i.i.i, label %if.then.i.i.i.i.i.i, label %_ZNSt16allocator_traitsISaISt6vectorISt5tupleIJidiEESaIS2_EEEE8allocateERS5_m.exit.i.i.i.i

if.then.i.i.i.i.i.i:                              ; preds = %cond.true.i.i.i.i
  invoke void @_ZSt17__throw_bad_allocv() #9
          to label %.noexc146 unwind label %lpad3

.noexc146:                                        ; preds = %if.then.i.i.i.i.i.i
  unreachable

_ZNSt16allocator_traitsISaISt6vectorISt5tupleIJidiEESaIS2_EEEE8allocateERS5_m.exit.i.i.i.i: ; preds = %cond.true.i.i.i.i
  %mul.i.i.i.i.i.i = mul nsw i64 %conv2, 24
  %call2.i.i.i.i3.i.i147 = invoke i8* @_Znwm(i64 %mul.i.i.i.i.i.i)
          to label %for.body.lr.ph.i.i.i.i.i144 unwind label %lpad3

for.body.lr.ph.i.i.i.i.i144:                      ; preds = %_ZNSt16allocator_traitsISaISt6vectorISt5tupleIJidiEESaIS2_EEEE8allocateERS5_m.exit.i.i.i.i
  %11 = bitcast i8* %call2.i.i.i.i3.i.i147 to %"class.std::vector.0"*
  %add.ptr.i.i.i = getelementptr %"class.std::vector.0", %"class.std::vector.0"* %11, i64 %conv2
  tail call void @llvm.memset.p0i8.i64(i8* nonnull %call2.i.i.i.i3.i.i147, i8 0, i64 %mul.i.i.i.i.i.i, i32 8, i1 false)
  br label %invoke.cont4

invoke.cont4:                                     ; preds = %invoke.cont, %for.body.lr.ph.i.i.i.i.i144
  %12 = phi i8* [ %call2.i.i.i.i3.i.i147, %for.body.lr.ph.i.i.i.i.i144 ], [ null, %invoke.cont ]
  %cond.i.i.i.i275 = phi %"class.std::vector.0"* [ %11, %for.body.lr.ph.i.i.i.i.i144 ], [ null, %invoke.cont ]
  %__cur.0.lcssa.i.i.i.i.i = phi %"class.std::vector.0"* [ %add.ptr.i.i.i, %for.body.lr.ph.i.i.i.i.i144 ], [ null, %invoke.cont ]
  %cmp327 = icmp sgt i32 %conv, 0
  br i1 %cmp327, label %pfor.detach.lr.ph, label %pfor.cond.cleanup

pfor.detach.lr.ph:                                ; preds = %invoke.cont4
  %conv8 = sext i32 %trials to i64
  %cmp.i.i.i.i166 = icmp eq i32 %trials, 0
  %add.ptr.i.i.i189285 = getelementptr i32, i32* null, i64 %conv8
  %cmp27320 = icmp sgt i32 %trials, 0
  %cmp.i.i.i.i.i.i167295 = icmp slt i32 %trials, 0
  %mul.i.i.i.i.i.i170 = shl nsw i64 %conv8, 2
  %_M_finish.i.i225 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %agg.result, i64 0, i32 0, i32 0, i32 1
  br label %pfor.detach
; LS: pfor.detach.lr.ph:
; LS: invoke fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE_pfor.detach.ls(%"class.std::vector.0"* %agg.result, [[IVTYPE:i[0-9]+]] 0, [[IVTYPE]] %{{.+}}, [[IVTYPE]] %{{.+}},
; LS-NEXT: to label %{{.+}} unwind label %lpad78.loopexit
; TT: pfor.detach.split:
; TT-NEXT: invoke fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE_pfor.body.cilk(%"class.std::vector.0"* %agg.result,

pfor.cond.cleanup:                                ; preds = %pfor.inc76, %if.end.i, %invoke.cont4
  %__cur.0.lcssa.i.i.i.i.i368 = phi %"class.std::vector.0"* [ %__cur.0.lcssa.i.i.i.i.i, %invoke.cont4 ], [ null, %if.end.i ], [ %__cur.0.lcssa.i.i.i.i.i, %pfor.inc76 ]
  %cond.i.i.i.i275367 = phi %"class.std::vector.0"* [ %cond.i.i.i.i275, %invoke.cont4 ], [ null, %if.end.i ], [ %cond.i.i.i.i275, %pfor.inc76 ]
  %13 = phi i8* [ %12, %invoke.cont4 ], [ null, %if.end.i ], [ %12, %pfor.inc76 ]
  sync within %syncreg, label %sync.continue85

lpad:                                             ; preds = %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE11_M_allocateEm.exit.i.i, %if.then.i
  %14 = landingpad { i8*, i32 }
          cleanup
  %15 = extractvalue { i8*, i32 } %14, 0
  %16 = extractvalue { i8*, i32 } %14, 1
  br label %ehcleanup95

lpad3:                                            ; preds = %_ZNSt16allocator_traitsISaISt6vectorISt5tupleIJidiEESaIS2_EEEE8allocateERS5_m.exit.i.i.i.i, %if.then.i.i.i.i.i.i
  %17 = landingpad { i8*, i32 }
          cleanup
  %18 = extractvalue { i8*, i32 } %17, 0
  %19 = extractvalue { i8*, i32 } %17, 1
  br label %ehcleanup95

pfor.detach:                                      ; preds = %pfor.detach.lr.ph, %pfor.inc76
  %__begin.0328 = phi i32 [ 0, %pfor.detach.lr.ph ], [ %inc77, %pfor.inc76 ]
  detach within %syncreg, label %pfor.body, label %pfor.inc76 unwind label %lpad78.loopexit

pfor.body:                                        ; preds = %pfor.detach
  %worker_matches_count.sroa.13 = alloca i32*, align 8
  %syncreg18 = call token @llvm.syncregion.start()
  %ref.tmp63 = alloca %"class.std::tuple", align 8
  %call7 = call i64 @clock() #8
  br i1 %cmp.i.i.i.i166, label %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i191.thread, label %cond.true.i.i.i.i168

_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i191.thread: ; preds = %pfor.body
  %worker_matches_count.sroa.13.0..sroa_cast243279 = bitcast i32** %worker_matches_count.sroa.13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %worker_matches_count.sroa.13.0..sroa_cast243279)
  store i32* %add.ptr.i.i.i189285, i32** %worker_matches_count.sroa.13, align 8
  br label %invoke.cont17

cond.true.i.i.i.i168:                             ; preds = %pfor.body
  br i1 %cmp.i.i.i.i.i.i167295, label %if.then.i.i.i.i.i.i169, label %_ZNSt16allocator_traitsISaIiEE8allocateERS0_m.exit.i.i.i.i

if.then.i.i.i.i.i.i169:                           ; preds = %cond.true.i.i.i.i168
  invoke void @_ZSt17__throw_bad_allocv() #9
          to label %.noexc177 unwind label %lpad10.loopexit.split-lp

.noexc177:                                        ; preds = %if.then.i.i.i.i.i.i169
  unreachable

_ZNSt16allocator_traitsISaIiEE8allocateERS0_m.exit.i.i.i.i: ; preds = %cond.true.i.i.i.i168
  %call2.i.i.i.i3.i.i179 = invoke i8* @_Znwm(i64 %mul.i.i.i.i.i.i170)
          to label %_ZNSt16allocator_traitsISaIiEE8allocateERS0_m.exit.i.i.i.i185 unwind label %lpad10.loopexit

_ZNSt16allocator_traitsISaIiEE8allocateERS0_m.exit.i.i.i.i185: ; preds = %_ZNSt16allocator_traitsISaIiEE8allocateERS0_m.exit.i.i.i.i
  call void @llvm.memset.p0i8.i64(i8* nonnull %call2.i.i.i.i3.i.i179, i8 0, i64 %mul.i.i.i.i.i.i170, i32 4, i1 false)
  %worker_matches_count.sroa.13.0..sroa_cast243 = bitcast i32** %worker_matches_count.sroa.13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %worker_matches_count.sroa.13.0..sroa_cast243)
  store i32* null, i32** %worker_matches_count.sroa.13, align 8
  %call2.i.i.i.i3.i.i197 = invoke i8* @_Znwm(i64 %mul.i.i.i.i.i.i170)
          to label %for.body.lr.ph.i.i.i.i.i.i.i192 unwind label %ehcleanup67.thread

for.body.lr.ph.i.i.i.i.i.i.i192:                  ; preds = %_ZNSt16allocator_traitsISaIiEE8allocateERS0_m.exit.i.i.i.i185
  %20 = bitcast i8* %call2.i.i.i.i3.i.i179 to i32*
  %21 = bitcast i8* %call2.i.i.i.i3.i.i197 to i32*
  %22 = ptrtoint i8* %call2.i.i.i.i3.i.i197 to i64
  %add.ptr.i.i.i189 = getelementptr i32, i32* %21, i64 %conv8
  store i32* %add.ptr.i.i.i189, i32** %worker_matches_count.sroa.13, align 8
  call void @llvm.memset.p0i8.i64(i8* nonnull %call2.i.i.i.i3.i.i197, i8 0, i64 %mul.i.i.i.i.i.i170, i32 4, i1 false)
  br label %invoke.cont17

invoke.cont17:                                    ; preds = %for.body.lr.ph.i.i.i.i.i.i.i192, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i191.thread
  %23 = phi i64 [ %22, %for.body.lr.ph.i.i.i.i.i.i.i192 ], [ 0, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i191.thread ]
  %24 = phi i8* [ %call2.i.i.i.i3.i.i197, %for.body.lr.ph.i.i.i.i.i.i.i192 ], [ null, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i191.thread ]
  %cond.i.i.i.i186288 = phi i32* [ %21, %for.body.lr.ph.i.i.i.i.i.i.i192 ], [ null, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i191.thread ]
  %25 = phi i8* [ %call2.i.i.i.i3.i.i179, %for.body.lr.ph.i.i.i.i.i.i.i192 ], [ null, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i191.thread ]
  %cond.i.i.i.i171277281287 = phi i32* [ %20, %for.body.lr.ph.i.i.i.i.i.i.i192 ], [ null, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i191.thread ]
  %worker_matches_count.sroa.13.0..sroa_cast243282286 = phi i8* [ %worker_matches_count.sroa.13.0..sroa_cast243, %for.body.lr.ph.i.i.i.i.i.i.i192 ], [ %worker_matches_count.sroa.13.0..sroa_cast243279, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i191.thread ]
  %__first.addr.0.lcssa.i.i.i.i.i.i.i193 = phi i32* [ %add.ptr.i.i.i189, %for.body.lr.ph.i.i.i.i.i.i.i192 ], [ null, %_ZNSt12_Vector_baseIiSaIiEEC2EmRKS0_.exit.i191.thread ]
  %26 = ptrtoint i32* %__first.addr.0.lcssa.i.i.i.i.i.i.i193 to i64
  br i1 %cmp27320, label %pfor.detach29.preheader, label %pfor.cond.cleanup28

pfor.detach29.preheader:                          ; preds = %invoke.cont17
  br label %pfor.detach29

pfor.cond.cleanup28:                              ; preds = %pfor.inc, %invoke.cont17
  sync within %syncreg18, label %sync.continue

lpad10.loopexit:                                  ; preds = %_ZNSt16allocator_traitsISaIiEE8allocateERS0_m.exit.i.i.i.i
  %lpad.loopexit299 = landingpad { i8*, i32 }
          catch i8* null
  br label %lpad10

lpad10.loopexit.split-lp:                         ; preds = %if.then.i.i.i.i.i.i169
  %lpad.loopexit.split-lp300 = landingpad { i8*, i32 }
          catch i8* null
  br label %lpad10

lpad10:                                           ; preds = %lpad10.loopexit.split-lp, %lpad10.loopexit
  %lpad.phi301 = phi { i8*, i32 } [ %lpad.loopexit299, %lpad10.loopexit ], [ %lpad.loopexit.split-lp300, %lpad10.loopexit.split-lp ]
  %27 = extractvalue { i8*, i32 } %lpad.phi301, 0
  %28 = extractvalue { i8*, i32 } %lpad.phi301, 1
  br label %ehcleanup69

ehcleanup67.thread:                               ; preds = %_ZNSt16allocator_traitsISaIiEE8allocateERS0_m.exit.i.i.i.i185
  %29 = landingpad { i8*, i32 }
          catch i8* null
  %worker_matches_count.sroa.13.0..sroa_cast243.le = bitcast i32** %worker_matches_count.sroa.13 to i8*
  %30 = extractvalue { i8*, i32 } %29, 0
  %31 = extractvalue { i8*, i32 } %29, 1
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %worker_matches_count.sroa.13.0..sroa_cast243.le)
  br label %if.then.i.i.i202

pfor.detach29:                                    ; preds = %pfor.detach29.preheader, %pfor.inc
  %indvars.iv = phi i64 [ %indvars.iv.next, %pfor.inc ], [ 0, %pfor.detach29.preheader ]
  detach within %syncreg18, label %pfor.body33, label %pfor.inc unwind label %lpad45.loopexit

pfor.body33:                                      ; preds = %pfor.detach29
  %32 = trunc i64 %indvars.iv to i32
  %call38 = invoke i32 @_Z15get_valid_movesi(i32 %32)
          to label %invoke.cont37 unwind label %lpad34

invoke.cont37:                                    ; preds = %pfor.body33
  %add.ptr.i209 = getelementptr inbounds i32, i32* %cond.i.i.i.i171277281287, i64 %indvars.iv
  store i32 %call38, i32* %add.ptr.i209, align 4, !tbaa !14
  %call42 = invoke i32 @_Z17get_matches_counti(i32 %32)
          to label %invoke.cont41 unwind label %lpad34

invoke.cont41:                                    ; preds = %invoke.cont37
  %add.ptr.i216 = getelementptr inbounds i32, i32* %cond.i.i.i.i186288, i64 %indvars.iv
  store i32 %call42, i32* %add.ptr.i216, align 4, !tbaa !14
  reattach within %syncreg18, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.detach29, %invoke.cont41
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp27 = icmp slt i64 %indvars.iv.next, %conv8
  br i1 %cmp27, label %pfor.detach29, label %pfor.cond.cleanup28, !llvm.loop !16

lpad34:                                           ; preds = %invoke.cont37, %pfor.body33
  %33 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg18, { i8*, i32 } %33)
          to label %det.rethrow.unreachable unwind label %lpad45.loopexit.split-lp

det.rethrow.unreachable:                          ; preds = %lpad34
  unreachable

lpad45.loopexit:                                  ; preds = %pfor.detach29
  %lpad.loopexit = landingpad { i8*, i32 }
          catch i8* null
  br label %lpad45

lpad45.loopexit.split-lp:                         ; preds = %lpad34
  %lpad.loopexit.split-lp = landingpad { i8*, i32 }
          catch i8* null
  br label %lpad45

lpad45:                                           ; preds = %lpad45.loopexit.split-lp, %lpad45.loopexit
  %lpad.phi = phi { i8*, i32 } [ %lpad.loopexit, %lpad45.loopexit ], [ %lpad.loopexit.split-lp, %lpad45.loopexit.split-lp ]
  %34 = extractvalue { i8*, i32 } %lpad.phi, 0
  %35 = extractvalue { i8*, i32 } %lpad.phi, 1
  sync within %syncreg18, label %ehcleanup

sync.continue:                                    ; preds = %pfor.cond.cleanup28
  %sub.ptr.sub.i223 = sub i64 %26, %23
  %sub.ptr.div.i224 = ashr exact i64 %sub.ptr.sub.i223, 2
  %cmp51322 = icmp eq i64 %sub.ptr.sub.i223, 0
  br i1 %cmp51322, label %invoke.cont65, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %sync.continue
  %36 = icmp ugt i64 %sub.ptr.div.i224, 1
  %umax = select i1 %36, i64 %sub.ptr.div.i224, i64 1
  %min.iters.check = icmp ult i64 %umax, 8
  br i1 %min.iters.check, label %for.body.preheader, label %vector.ph

vector.ph:                                        ; preds = %for.body.lr.ph
  %n.vec = and i64 %umax, -8
  %37 = add nsw i64 %n.vec, -8
  %38 = lshr exact i64 %37, 3
  %39 = add nuw nsw i64 %38, 1
  %xtraiter = and i64 %39, 1
  %40 = icmp eq i64 %37, 0
  br i1 %40, label %middle.block.unr-lcssa, label %vector.ph.new

vector.ph.new:                                    ; preds = %vector.ph
  %unroll_iter = sub nsw i64 %39, %xtraiter
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph.new
  %index = phi i64 [ 0, %vector.ph.new ], [ %index.next.1, %vector.body ]
  %vec.phi = phi <4 x i32> [ zeroinitializer, %vector.ph.new ], [ %57, %vector.body ]
  %vec.phi415 = phi <4 x i32> [ zeroinitializer, %vector.ph.new ], [ %58, %vector.body ]
  %vec.phi416 = phi <4 x i32> [ zeroinitializer, %vector.ph.new ], [ %63, %vector.body ]
  %vec.phi417 = phi <4 x i32> [ zeroinitializer, %vector.ph.new ], [ %64, %vector.body ]
  %niter = phi i64 [ %unroll_iter, %vector.ph.new ], [ %niter.nsub.1, %vector.body ]
  %41 = getelementptr inbounds i32, i32* %cond.i.i.i.i186288, i64 %index
  %42 = bitcast i32* %41 to <4 x i32>*
  %wide.load = load <4 x i32>, <4 x i32>* %42, align 4, !tbaa !14
  %43 = getelementptr i32, i32* %41, i64 4
  %44 = bitcast i32* %43 to <4 x i32>*
  %wide.load418 = load <4 x i32>, <4 x i32>* %44, align 4, !tbaa !14
  %45 = add nsw <4 x i32> %wide.load, %vec.phi
  %46 = add nsw <4 x i32> %wide.load418, %vec.phi415
  %47 = getelementptr inbounds i32, i32* %cond.i.i.i.i171277281287, i64 %index
  %48 = bitcast i32* %47 to <4 x i32>*
  %wide.load419 = load <4 x i32>, <4 x i32>* %48, align 4, !tbaa !14
  %49 = getelementptr i32, i32* %47, i64 4
  %50 = bitcast i32* %49 to <4 x i32>*
  %wide.load420 = load <4 x i32>, <4 x i32>* %50, align 4, !tbaa !14
  %51 = add nsw <4 x i32> %wide.load419, %vec.phi416
  %52 = add nsw <4 x i32> %wide.load420, %vec.phi417
  %index.next = or i64 %index, 8
  %53 = getelementptr inbounds i32, i32* %cond.i.i.i.i186288, i64 %index.next
  %54 = bitcast i32* %53 to <4 x i32>*
  %wide.load.1 = load <4 x i32>, <4 x i32>* %54, align 4, !tbaa !14
  %55 = getelementptr i32, i32* %53, i64 4
  %56 = bitcast i32* %55 to <4 x i32>*
  %wide.load418.1 = load <4 x i32>, <4 x i32>* %56, align 4, !tbaa !14
  %57 = add nsw <4 x i32> %wide.load.1, %45
  %58 = add nsw <4 x i32> %wide.load418.1, %46
  %59 = getelementptr inbounds i32, i32* %cond.i.i.i.i171277281287, i64 %index.next
  %60 = bitcast i32* %59 to <4 x i32>*
  %wide.load419.1 = load <4 x i32>, <4 x i32>* %60, align 4, !tbaa !14
  %61 = getelementptr i32, i32* %59, i64 4
  %62 = bitcast i32* %61 to <4 x i32>*
  %wide.load420.1 = load <4 x i32>, <4 x i32>* %62, align 4, !tbaa !14
  %63 = add nsw <4 x i32> %wide.load419.1, %51
  %64 = add nsw <4 x i32> %wide.load420.1, %52
  %index.next.1 = add i64 %index, 16
  %niter.nsub.1 = add i64 %niter, -2
  %niter.ncmp.1 = icmp eq i64 %niter.nsub.1, 0
  br i1 %niter.ncmp.1, label %middle.block.unr-lcssa, label %vector.body, !llvm.loop !18

middle.block.unr-lcssa:                           ; preds = %vector.body, %vector.ph
  %.lcssa432.ph = phi <4 x i32> [ undef, %vector.ph ], [ %57, %vector.body ]
  %.lcssa431.ph = phi <4 x i32> [ undef, %vector.ph ], [ %58, %vector.body ]
  %.lcssa430.ph = phi <4 x i32> [ undef, %vector.ph ], [ %63, %vector.body ]
  %.lcssa.ph = phi <4 x i32> [ undef, %vector.ph ], [ %64, %vector.body ]
  %index.unr = phi i64 [ 0, %vector.ph ], [ %index.next.1, %vector.body ]
  %vec.phi.unr = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %57, %vector.body ]
  %vec.phi415.unr = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %58, %vector.body ]
  %vec.phi416.unr = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %63, %vector.body ]
  %vec.phi417.unr = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %64, %vector.body ]
  %lcmp.mod = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod, label %middle.block, label %vector.body.epil

vector.body.epil:                                 ; preds = %middle.block.unr-lcssa
  %65 = getelementptr inbounds i32, i32* %cond.i.i.i.i186288, i64 %index.unr
  %66 = getelementptr inbounds i32, i32* %cond.i.i.i.i171277281287, i64 %index.unr
  %67 = getelementptr i32, i32* %66, i64 4
  %68 = bitcast i32* %67 to <4 x i32>*
  %wide.load420.epil = load <4 x i32>, <4 x i32>* %68, align 4, !tbaa !14
  %69 = add nsw <4 x i32> %wide.load420.epil, %vec.phi417.unr
  %70 = bitcast i32* %66 to <4 x i32>*
  %wide.load419.epil = load <4 x i32>, <4 x i32>* %70, align 4, !tbaa !14
  %71 = add nsw <4 x i32> %wide.load419.epil, %vec.phi416.unr
  %72 = getelementptr i32, i32* %65, i64 4
  %73 = bitcast i32* %72 to <4 x i32>*
  %wide.load418.epil = load <4 x i32>, <4 x i32>* %73, align 4, !tbaa !14
  %74 = add nsw <4 x i32> %wide.load418.epil, %vec.phi415.unr
  %75 = bitcast i32* %65 to <4 x i32>*
  %wide.load.epil = load <4 x i32>, <4 x i32>* %75, align 4, !tbaa !14
  %76 = add nsw <4 x i32> %wide.load.epil, %vec.phi.unr
  br label %middle.block

middle.block:                                     ; preds = %middle.block.unr-lcssa, %vector.body.epil
  %.lcssa432 = phi <4 x i32> [ %.lcssa432.ph, %middle.block.unr-lcssa ], [ %76, %vector.body.epil ]
  %.lcssa431 = phi <4 x i32> [ %.lcssa431.ph, %middle.block.unr-lcssa ], [ %74, %vector.body.epil ]
  %.lcssa430 = phi <4 x i32> [ %.lcssa430.ph, %middle.block.unr-lcssa ], [ %71, %vector.body.epil ]
  %.lcssa = phi <4 x i32> [ %.lcssa.ph, %middle.block.unr-lcssa ], [ %69, %vector.body.epil ]
  %bin.rdx424 = add <4 x i32> %.lcssa, %.lcssa430
  %rdx.shuf425 = shufflevector <4 x i32> %bin.rdx424, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx426 = add <4 x i32> %bin.rdx424, %rdx.shuf425
  %rdx.shuf427 = shufflevector <4 x i32> %bin.rdx426, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx428 = add <4 x i32> %bin.rdx426, %rdx.shuf427
  %77 = extractelement <4 x i32> %bin.rdx428, i32 0
  %bin.rdx = add <4 x i32> %.lcssa431, %.lcssa432
  %rdx.shuf = shufflevector <4 x i32> %bin.rdx, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx421 = add <4 x i32> %bin.rdx, %rdx.shuf
  %rdx.shuf422 = shufflevector <4 x i32> %bin.rdx421, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx423 = add <4 x i32> %bin.rdx421, %rdx.shuf422
  %78 = extractelement <4 x i32> %bin.rdx423, i32 0
  %cmp.n = icmp eq i64 %umax, %n.vec
  br i1 %cmp.n, label %invoke.cont65, label %for.body.preheader

for.body.preheader:                               ; preds = %middle.block, %for.body.lr.ph
  %indvars.iv356.ph = phi i64 [ 0, %for.body.lr.ph ], [ %n.vec, %middle.block ]
  %matches_count.0324.ph = phi i32 [ 0, %for.body.lr.ph ], [ %78, %middle.block ]
  %valid_moves.0323.ph = phi i32 [ 0, %for.body.lr.ph ], [ %77, %middle.block ]
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv356 = phi i64 [ %indvars.iv.next357, %for.body ], [ %indvars.iv356.ph, %for.body.preheader ]
  %matches_count.0324 = phi i32 [ %add54, %for.body ], [ %matches_count.0324.ph, %for.body.preheader ]
  %valid_moves.0323 = phi i32 [ %add57, %for.body ], [ %valid_moves.0323.ph, %for.body.preheader ]
  %add.ptr.i232 = getelementptr inbounds i32, i32* %cond.i.i.i.i186288, i64 %indvars.iv356
  %79 = load i32, i32* %add.ptr.i232, align 4, !tbaa !14
  %add54 = add nsw i32 %79, %matches_count.0324
  %add.ptr.i230 = getelementptr inbounds i32, i32* %cond.i.i.i.i171277281287, i64 %indvars.iv356
  %80 = load i32, i32* %add.ptr.i230, align 4, !tbaa !14
  %add57 = add nsw i32 %80, %valid_moves.0323
  %indvars.iv.next357 = add nuw i64 %indvars.iv356, 1
  %cmp51 = icmp ugt i64 %sub.ptr.div.i224, %indvars.iv.next357
  br i1 %cmp51, label %for.body, label %invoke.cont65, !llvm.loop !20

invoke.cont65:                                    ; preds = %for.body, %middle.block, %sync.continue
  %valid_moves.0.lcssa = phi i32 [ 0, %sync.continue ], [ %77, %middle.block ], [ %add57, %for.body ]
  %matches_count.0.lcssa = phi i32 [ 0, %sync.continue ], [ %78, %middle.block ], [ %add54, %for.body ]
  %call59 = call i64 @clock() #8
  %sub60 = sub nsw i64 %call59, %call7
  %conv61 = sitofp i64 %sub60 to double
  %div62 = fdiv double %conv61, 1.000000e+06
  %81 = bitcast %"class.std::tuple"* %ref.tmp63 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %81) #8
  %_M_head_impl.i.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %ref.tmp63, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store i32 %matches_count.0.lcssa, i32* %_M_head_impl.i.i.i.i.i.i, align 8, !tbaa !22, !alias.scope !24
  %82 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %ref.tmp63, i64 0, i32 0, i32 0, i32 1, i32 0
  store double %div62, double* %82, align 8, !tbaa !27, !alias.scope !24
  %83 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %ref.tmp63, i64 0, i32 0, i32 1, i32 0
  store i32 %valid_moves.0.lcssa, i32* %83, align 8, !tbaa !30, !alias.scope !24
  %84 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_finish.i.i225, align 8, !tbaa !12
  %85 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_end_of_storage.i.i, align 8, !tbaa !13
  %cmp.i.i = icmp eq %"class.std::tuple"* %84, %85
  br i1 %cmp.i.i, label %if.else.i.i, label %if.then.i.i227

if.then.i.i227:                                   ; preds = %invoke.cont65
  %_M_head_impl.i.i6.i.i.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %84, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store i32 %matches_count.0.lcssa, i32* %_M_head_impl.i.i6.i.i.i.i.i.i.i, align 4, !tbaa !22
  %86 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %84, i64 0, i32 0, i32 0, i32 1, i32 0
  store double %div62, double* %86, align 8, !tbaa !27
  %87 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %84, i64 0, i32 0, i32 1, i32 0
  %88 = load i32, i32* %83, align 8, !tbaa !14
  store i32 %88, i32* %87, align 4, !tbaa !30
  %incdec.ptr.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %84, i64 1
  store %"class.std::tuple"* %incdec.ptr.i.i, %"class.std::tuple"** %_M_finish.i.i225, align 8, !tbaa !12
  br label %invoke.cont66

if.else.i.i:                                      ; preds = %invoke.cont65
  invoke void @_ZNSt6vectorISt5tupleIJidiEESaIS1_EE17_M_realloc_insertIJS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_(%"class.std::vector.0"* nonnull %agg.result, %"class.std::tuple"* %84, %"class.std::tuple"* nonnull dereferenceable(24) %ref.tmp63)
          to label %invoke.cont66 unwind label %lpad64

invoke.cont66:                                    ; preds = %if.then.i.i227, %if.else.i.i
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %81) #8
  %tobool.i.i.i218 = icmp eq i32* %cond.i.i.i.i186288, null
  br i1 %tobool.i.i.i218, label %_ZNSt6vectorIiSaIiEED2Ev.exit221, label %if.then.i.i.i220

if.then.i.i.i220:                                 ; preds = %invoke.cont66
  call void @_ZdlPv(i8* %24) #8
  br label %_ZNSt6vectorIiSaIiEED2Ev.exit221

_ZNSt6vectorIiSaIiEED2Ev.exit221:                 ; preds = %invoke.cont66, %if.then.i.i.i220
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %worker_matches_count.sroa.13.0..sroa_cast243282286)
  %tobool.i.i.i211 = icmp eq i32* %cond.i.i.i.i171277281287, null
  br i1 %tobool.i.i.i211, label %_ZNSt6vectorIiSaIiEED2Ev.exit214, label %if.then.i.i.i213

if.then.i.i.i213:                                 ; preds = %_ZNSt6vectorIiSaIiEED2Ev.exit221
  call void @_ZdlPv(i8* %25) #8
  br label %_ZNSt6vectorIiSaIiEED2Ev.exit214

_ZNSt6vectorIiSaIiEED2Ev.exit214:                 ; preds = %_ZNSt6vectorIiSaIiEED2Ev.exit221, %if.then.i.i.i213
  reattach within %syncreg, label %pfor.inc76

pfor.inc76:                                       ; preds = %pfor.detach, %_ZNSt6vectorIiSaIiEED2Ev.exit214
  %inc77 = add nuw nsw i32 %__begin.0328, 1
  %cmp = icmp slt i32 %inc77, %conv
  br i1 %cmp, label %pfor.detach, label %pfor.cond.cleanup, !llvm.loop !32

lpad64:                                           ; preds = %if.else.i.i
  %89 = landingpad { i8*, i32 }
          catch i8* null
  %90 = bitcast %"class.std::tuple"* %ref.tmp63 to i8*
  %91 = extractvalue { i8*, i32 } %89, 0
  %92 = extractvalue { i8*, i32 } %89, 1
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %90) #8
  br label %ehcleanup

ehcleanup:                                        ; preds = %lpad45, %lpad64
  %exn.slot11.0 = phi i8* [ %91, %lpad64 ], [ %34, %lpad45 ]
  %ehselector.slot12.0 = phi i32 [ %92, %lpad64 ], [ %35, %lpad45 ]
  %tobool.i.i.i204 = icmp eq i32* %cond.i.i.i.i186288, null
  br i1 %tobool.i.i.i204, label %ehcleanup67, label %if.then.i.i.i206

if.then.i.i.i206:                                 ; preds = %ehcleanup
  call void @_ZdlPv(i8* %24) #8
  br label %ehcleanup67

ehcleanup67:                                      ; preds = %if.then.i.i.i206, %ehcleanup
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %worker_matches_count.sroa.13.0..sroa_cast243282286)
  %tobool.i.i.i200 = icmp eq i32* %cond.i.i.i.i171277281287, null
  br i1 %tobool.i.i.i200, label %ehcleanup69, label %if.then.i.i.i202

if.then.i.i.i202:                                 ; preds = %ehcleanup67.thread, %ehcleanup67
  %ehselector.slot12.1293 = phi i32 [ %31, %ehcleanup67.thread ], [ %ehselector.slot12.0, %ehcleanup67 ]
  %exn.slot11.1291 = phi i8* [ %30, %ehcleanup67.thread ], [ %exn.slot11.0, %ehcleanup67 ]
  %93 = phi i8* [ %call2.i.i.i.i3.i.i179, %ehcleanup67.thread ], [ %25, %ehcleanup67 ]
  call void @_ZdlPv(i8* %93) #8
  br label %ehcleanup69

ehcleanup69:                                      ; preds = %if.then.i.i.i202, %ehcleanup67, %lpad10
  %exn.slot11.2 = phi i8* [ %27, %lpad10 ], [ %exn.slot11.0, %ehcleanup67 ], [ %exn.slot11.1291, %if.then.i.i.i202 ]
  %ehselector.slot12.2 = phi i32 [ %28, %lpad10 ], [ %ehselector.slot12.0, %ehcleanup67 ], [ %ehselector.slot12.1293, %if.then.i.i.i202 ]
  %lpad.val81 = insertvalue { i8*, i32 } undef, i8* %exn.slot11.2, 0
  %lpad.val82 = insertvalue { i8*, i32 } %lpad.val81, i32 %ehselector.slot12.2, 1
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg, { i8*, i32 } %lpad.val82)
          to label %det.rethrow.unreachable84 unwind label %lpad78.loopexit.split-lp

det.rethrow.unreachable84:                        ; preds = %ehcleanup69
  unreachable

lpad78.loopexit:                                  ; preds = %pfor.detach
  %lpad.loopexit296 = landingpad { i8*, i32 }
          cleanup
  br label %lpad78

lpad78.loopexit.split-lp:                         ; preds = %ehcleanup69
  %lpad.loopexit.split-lp297 = landingpad { i8*, i32 }
          cleanup
  br label %lpad78

lpad78:                                           ; preds = %lpad78.loopexit.split-lp, %lpad78.loopexit
  %lpad.phi298 = phi { i8*, i32 } [ %lpad.loopexit296, %lpad78.loopexit ], [ %lpad.loopexit.split-lp297, %lpad78.loopexit.split-lp ]
  %94 = extractvalue { i8*, i32 } %lpad.phi298, 0
  %95 = extractvalue { i8*, i32 } %lpad.phi298, 1
  sync within %syncreg, label %sync.continue87

sync.continue85:                                  ; preds = %pfor.cond.cleanup
  %cmp3.i.i.i.i150 = icmp eq %"class.std::vector.0"* %cond.i.i.i.i275367, %__cur.0.lcssa.i.i.i.i.i368
  br i1 %cmp3.i.i.i.i150, label %invoke.cont.i163, label %for.body.i.i.i.i155.preheader

for.body.i.i.i.i155.preheader:                    ; preds = %sync.continue85
  br label %for.body.i.i.i.i155

for.body.i.i.i.i155:                              ; preds = %for.body.i.i.i.i155.preheader, %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i159
  %__first.addr.04.i.i.i.i152 = phi %"class.std::vector.0"* [ %incdec.ptr.i.i.i.i157, %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i159 ], [ %cond.i.i.i.i275367, %for.body.i.i.i.i155.preheader ]
  %_M_start.i.i.i.i.i.i.i153 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %__first.addr.04.i.i.i.i152, i64 0, i32 0, i32 0, i32 0
  %96 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_start.i.i.i.i.i.i.i153, align 8, !tbaa !9
  %tobool.i.i.i.i.i.i.i.i154 = icmp eq %"class.std::tuple"* %96, null
  br i1 %tobool.i.i.i.i.i.i.i.i154, label %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i159, label %if.then.i.i.i.i.i.i.i.i156

if.then.i.i.i.i.i.i.i.i156:                       ; preds = %for.body.i.i.i.i155
  %97 = bitcast %"class.std::tuple"* %96 to i8*
  call void @_ZdlPv(i8* %97) #8
  br label %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i159

_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i159: ; preds = %if.then.i.i.i.i.i.i.i.i156, %for.body.i.i.i.i155
  %incdec.ptr.i.i.i.i157 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %__first.addr.04.i.i.i.i152, i64 1
  %cmp.i.i.i.i158 = icmp eq %"class.std::vector.0"* %incdec.ptr.i.i.i.i157, %__cur.0.lcssa.i.i.i.i.i368
  br i1 %cmp.i.i.i.i158, label %invoke.cont.i163, label %for.body.i.i.i.i155

invoke.cont.i163:                                 ; preds = %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i159, %sync.continue85
  %tobool.i.i.i162 = icmp eq %"class.std::vector.0"* %cond.i.i.i.i275367, null
  br i1 %tobool.i.i.i162, label %_ZNSt6vectorIS_ISt5tupleIJidiEESaIS1_EESaIS3_EED2Ev.exit165, label %if.then.i.i.i164

if.then.i.i.i164:                                 ; preds = %invoke.cont.i163
  call void @_ZdlPv(i8* %13) #8
  br label %_ZNSt6vectorIS_ISt5tupleIJidiEESaIS1_EESaIS3_EED2Ev.exit165

_ZNSt6vectorIS_ISt5tupleIJidiEESaIS1_EESaIS3_EED2Ev.exit165: ; preds = %invoke.cont.i163, %if.then.i.i.i164
  ret void

sync.continue87:                                  ; preds = %lpad78
  %cmp3.i.i.i.i = icmp eq %"class.std::vector.0"* %cond.i.i.i.i275, %__cur.0.lcssa.i.i.i.i.i
  br i1 %cmp3.i.i.i.i, label %invoke.cont.i, label %for.body.i.i.i.i.preheader

for.body.i.i.i.i.preheader:                       ; preds = %sync.continue87
  br label %for.body.i.i.i.i

for.body.i.i.i.i:                                 ; preds = %for.body.i.i.i.i.preheader, %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i
  %__first.addr.04.i.i.i.i = phi %"class.std::vector.0"* [ %incdec.ptr.i.i.i.i, %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i ], [ %cond.i.i.i.i275, %for.body.i.i.i.i.preheader ]
  %_M_start.i.i.i.i.i.i.i = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %__first.addr.04.i.i.i.i, i64 0, i32 0, i32 0, i32 0
  %98 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_start.i.i.i.i.i.i.i, align 8, !tbaa !9
  %tobool.i.i.i.i.i.i.i.i = icmp eq %"class.std::tuple"* %98, null
  br i1 %tobool.i.i.i.i.i.i.i.i, label %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i, label %if.then.i.i.i.i.i.i.i.i

if.then.i.i.i.i.i.i.i.i:                          ; preds = %for.body.i.i.i.i
  %99 = bitcast %"class.std::tuple"* %98 to i8*
  call void @_ZdlPv(i8* %99) #8
  br label %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i

_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i: ; preds = %if.then.i.i.i.i.i.i.i.i, %for.body.i.i.i.i
  %incdec.ptr.i.i.i.i = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %__first.addr.04.i.i.i.i, i64 1
  %cmp.i.i.i.i = icmp eq %"class.std::vector.0"* %incdec.ptr.i.i.i.i, %__cur.0.lcssa.i.i.i.i.i
  br i1 %cmp.i.i.i.i, label %invoke.cont.i, label %for.body.i.i.i.i

invoke.cont.i:                                    ; preds = %_ZSt8_DestroyISt6vectorISt5tupleIJidiEESaIS2_EEEvPT_.exit.i.i.i.i, %sync.continue87
  %tobool.i.i.i136 = icmp eq %"class.std::vector.0"* %cond.i.i.i.i275, null
  br i1 %tobool.i.i.i136, label %ehcleanup95, label %if.then.i.i.i137

if.then.i.i.i137:                                 ; preds = %invoke.cont.i
  call void @_ZdlPv(i8* %12) #8
  br label %ehcleanup95

ehcleanup95:                                      ; preds = %lpad3, %invoke.cont.i, %if.then.i.i.i137, %lpad
  %ehselector.slot.1 = phi i32 [ %16, %lpad ], [ %19, %lpad3 ], [ %95, %invoke.cont.i ], [ %95, %if.then.i.i.i137 ]
  %exn.slot.1 = phi i8* [ %15, %lpad ], [ %18, %lpad3 ], [ %94, %invoke.cont.i ], [ %94, %if.then.i.i.i137 ]
  %_M_start.i.i = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %agg.result, i64 0, i32 0, i32 0, i32 0
  %100 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_start.i.i, align 8, !tbaa !9
  %tobool.i.i.i = icmp eq %"class.std::tuple"* %100, null
  br i1 %tobool.i.i.i, label %_ZNSt6vectorISt5tupleIJidiEESaIS1_EED2Ev.exit, label %if.then.i.i.i

if.then.i.i.i:                                    ; preds = %ehcleanup95
  %101 = bitcast %"class.std::tuple"* %100 to i8*
  call void @_ZdlPv(i8* %101) #8
  br label %_ZNSt6vectorISt5tupleIJidiEESaIS1_EED2Ev.exit

_ZNSt6vectorISt5tupleIJidiEESaIS1_EED2Ev.exit:    ; preds = %ehcleanup95, %if.then.i.i.i
  %lpad.val98 = insertvalue { i8*, i32 } undef, i8* %exn.slot.1, 0
  %lpad.val99 = insertvalue { i8*, i32 } %lpad.val98, i32 %ehselector.slot.1, 1
  resume { i8*, i32 } %lpad.val99
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

; Function Attrs: argmemonly
declare void @llvm.detached.rethrow.sl_p0i8i32s(token, { i8*, i32 }) #4

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(i8*) local_unnamed_addr #5

; Function Attrs: noreturn
declare void @_ZSt20__throw_length_errorPKc(i8*) local_unnamed_addr #6

; Function Attrs: noreturn
declare void @_ZSt17__throw_bad_allocv() local_unnamed_addr #6

; Function Attrs: nobuiltin
declare noalias nonnull i8* @_Znwm(i64) local_unnamed_addr #7

; Function Attrs: uwtable
define linkonce_odr void @_ZNSt6vectorISt5tupleIJidiEESaIS1_EE17_M_realloc_insertIJS1_EEEvN9__gnu_cxx17__normal_iteratorIPS1_S3_EEDpOT_(%"class.std::vector.0"* %this, %"class.std::tuple"* %__position.coerce, %"class.std::tuple"* dereferenceable(24) %__args) local_unnamed_addr #0 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
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
  tail call void @_ZSt17__throw_bad_allocv() #9
  unreachable

_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i: ; preds = %entry
  %6 = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %this, i64 0, i32 0, i32 0, i32 0
  %mul.i.i.i = mul i64 %cond.i, 24
  %call2.i.i.i = tail call i8* @_Znwm(i64 %mul.i.i.i)
  %7 = bitcast i8* %call2.i.i.i to %"class.std::tuple"*
  %_M_head_impl.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__args, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %_M_head_impl.i.i6.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %7, i64 %sub.ptr.div.i, i32 0, i32 0, i32 0, i32 0, i32 0
  %8 = load i32, i32* %_M_head_impl.i.i.i.i.i.i.i.i, align 4, !tbaa !14
  store i32 %8, i32* %_M_head_impl.i.i6.i.i.i.i.i, align 4, !tbaa !22
  %9 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %7, i64 %sub.ptr.div.i, i32 0, i32 0, i32 1
  %10 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__args, i64 0, i32 0, i32 0, i32 1, i32 0
  %11 = bitcast double* %10 to i64*
  %12 = load i64, i64* %11, align 8, !tbaa !34
  %13 = bitcast %"struct.std::_Head_base.7"* %9 to i64*
  store i64 %12, i64* %13, align 8, !tbaa !27
  %14 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__args, i64 0, i32 0, i32 1, i32 0
  %15 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %7, i64 %sub.ptr.div.i, i32 0, i32 1, i32 0
  %16 = load i32, i32* %14, align 4, !tbaa !14
  store i32 %16, i32* %15, align 4, !tbaa !30
  %17 = load %"class.std::tuple"*, %"class.std::tuple"** %6, align 8, !tbaa !9
  %cmp.i.i21.i.i.i.i73 = icmp eq %"class.std::tuple"* %17, %__position.coerce
  br i1 %cmp.i.i21.i.i.i.i73, label %invoke.cont10, label %for.body.i.i.i.i82.preheader

for.body.i.i.i.i82.preheader:                     ; preds = %_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i
  br label %for.body.i.i.i.i82

for.body.i.i.i.i82:                               ; preds = %for.body.i.i.i.i82.preheader, %for.body.i.i.i.i82
  %__cur.023.i.i.i.i75 = phi %"class.std::tuple"* [ %incdec.ptr.i.i.i.i80, %for.body.i.i.i.i82 ], [ %7, %for.body.i.i.i.i82.preheader ]
  %__first.sroa.0.022.i.i.i.i76 = phi %"class.std::tuple"* [ %incdec.ptr.i.i.i.i.i79, %for.body.i.i.i.i82 ], [ %17, %for.body.i.i.i.i82.preheader ]
  %_M_head_impl.i.i.i.i.i.i.i.i.i.i.i77 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.022.i.i.i.i76, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %_M_head_impl.i.i6.i.i.i.i.i.i.i.i78 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.023.i.i.i.i75, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %18 = load i32, i32* %_M_head_impl.i.i.i.i.i.i.i.i.i.i.i77, align 4, !tbaa !14
  store i32 %18, i32* %_M_head_impl.i.i6.i.i.i.i.i.i.i.i78, align 4, !tbaa !22
  %19 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.023.i.i.i.i75, i64 0, i32 0, i32 0, i32 1
  %20 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.022.i.i.i.i76, i64 0, i32 0, i32 0, i32 1, i32 0
  %21 = bitcast double* %20 to i64*
  %22 = load i64, i64* %21, align 8, !tbaa !34
  %23 = bitcast %"struct.std::_Head_base.7"* %19 to i64*
  store i64 %22, i64* %23, align 8, !tbaa !27
  %24 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.022.i.i.i.i76, i64 0, i32 0, i32 1, i32 0
  %25 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.023.i.i.i.i75, i64 0, i32 0, i32 1, i32 0
  %26 = load i32, i32* %24, align 4, !tbaa !14
  store i32 %26, i32* %25, align 4, !tbaa !30
  %incdec.ptr.i.i.i.i.i79 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.022.i.i.i.i76, i64 1
  %incdec.ptr.i.i.i.i80 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.023.i.i.i.i75, i64 1
  %cmp.i.i.i.i.i.i81 = icmp eq %"class.std::tuple"* %incdec.ptr.i.i.i.i.i79, %__position.coerce
  br i1 %cmp.i.i.i.i.i.i81, label %invoke.cont10, label %for.body.i.i.i.i82

invoke.cont10:                                    ; preds = %for.body.i.i.i.i82, %_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i
  %__cur.0.lcssa.i.i.i.i83 = phi %"class.std::tuple"* [ %7, %_ZNSt16allocator_traitsISaISt5tupleIJidiEEEE8allocateERS2_m.exit.i ], [ %incdec.ptr.i.i.i.i80, %for.body.i.i.i.i82 ]
  %incdec.ptr = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.0.lcssa.i.i.i.i83, i64 1
  %27 = load %"class.std::tuple"*, %"class.std::tuple"** %_M_finish.i20.i, align 8, !tbaa !12
  %cmp.i.i21.i.i.i.i = icmp eq %"class.std::tuple"* %27, %__position.coerce
  br i1 %cmp.i.i21.i.i.i.i, label %invoke.cont15, label %for.body.i.i.i.i.preheader

for.body.i.i.i.i.preheader:                       ; preds = %invoke.cont10
  br label %for.body.i.i.i.i

for.body.i.i.i.i:                                 ; preds = %for.body.i.i.i.i.preheader, %for.body.i.i.i.i
  %__cur.023.i.i.i.i = phi %"class.std::tuple"* [ %incdec.ptr.i.i.i.i, %for.body.i.i.i.i ], [ %incdec.ptr, %for.body.i.i.i.i.preheader ]
  %__first.sroa.0.022.i.i.i.i = phi %"class.std::tuple"* [ %incdec.ptr.i.i.i.i.i, %for.body.i.i.i.i ], [ %__position.coerce, %for.body.i.i.i.i.preheader ]
  %_M_head_impl.i.i.i.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.022.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %_M_head_impl.i.i6.i.i.i.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.023.i.i.i.i, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %28 = load i32, i32* %_M_head_impl.i.i.i.i.i.i.i.i.i.i.i, align 4, !tbaa !14
  store i32 %28, i32* %_M_head_impl.i.i6.i.i.i.i.i.i.i.i, align 4, !tbaa !22
  %29 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.023.i.i.i.i, i64 0, i32 0, i32 0, i32 1
  %30 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.022.i.i.i.i, i64 0, i32 0, i32 0, i32 1, i32 0
  %31 = bitcast double* %30 to i64*
  %32 = load i64, i64* %31, align 8, !tbaa !34
  %33 = bitcast %"struct.std::_Head_base.7"* %29 to i64*
  store i64 %32, i64* %33, align 8, !tbaa !27
  %34 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.022.i.i.i.i, i64 0, i32 0, i32 1, i32 0
  %35 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.023.i.i.i.i, i64 0, i32 0, i32 1, i32 0
  %36 = load i32, i32* %34, align 4, !tbaa !14
  store i32 %36, i32* %35, align 4, !tbaa !30
  %incdec.ptr.i.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__first.sroa.0.022.i.i.i.i, i64 1
  %incdec.ptr.i.i.i.i = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %__cur.023.i.i.i.i, i64 1
  %cmp.i.i.i.i.i.i = icmp eq %"class.std::tuple"* %incdec.ptr.i.i.i.i.i, %27
  br i1 %cmp.i.i.i.i.i.i, label %invoke.cont15, label %for.body.i.i.i.i

invoke.cont15:                                    ; preds = %for.body.i.i.i.i, %invoke.cont10
  %__cur.0.lcssa.i.i.i.i = phi %"class.std::tuple"* [ %incdec.ptr, %invoke.cont10 ], [ %incdec.ptr.i.i.i.i, %for.body.i.i.i.i ]
  %_M_end_of_storage = getelementptr inbounds %"class.std::vector.0", %"class.std::vector.0"* %this, i64 0, i32 0, i32 0, i32 2
  %tobool.i69 = icmp eq %"class.std::tuple"* %17, null
  br i1 %tobool.i69, label %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit71, label %if.then.i70

if.then.i70:                                      ; preds = %invoke.cont15
  %37 = bitcast %"class.std::tuple"* %17 to i8*
  tail call void @_ZdlPv(i8* %37) #8
  br label %_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit71

_ZNSt12_Vector_baseISt5tupleIJidiEESaIS1_EE13_M_deallocateEPS1_m.exit71: ; preds = %invoke.cont15, %if.then.i70
  %38 = bitcast %"class.std::vector.0"* %this to i8**
  store i8* %call2.i.i.i, i8** %38, align 8, !tbaa !9
  store %"class.std::tuple"* %__cur.0.lcssa.i.i.i.i, %"class.std::tuple"** %_M_finish.i20.i, align 8, !tbaa !12
  %add.ptr39 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %7, i64 %cond.i
  store %"class.std::tuple"* %add.ptr39, %"class.std::tuple"** %_M_end_of_storage, align 8, !tbaa !13
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #1

; LS-LABEL: define internal fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE_pfor.detach.ls(%"class.std::vector.0"* noalias sret align 8 %agg.result.ls,
; LS: [[IVTYPE]] {{.+}}, [[IVTYPE]] {{.+}}, [[IVTYPE]] {{.*}}%[[GRAINSIZE:.+]],
; LS: {{^.split:}}
; LS-NEXT: invoke fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE_pfor.detach.ls(%"class.std::vector.0"* %agg.result.ls, [[IVTYPE]] {{.+}}, [[IVTYPE]] {{.+}}, [[IVTYPE]] {{.*}}[[GRAINSIZE]],

; TT-LABEL: define internal fastcc void @_Z14func_with_sretidRSt6vectorI6paramsSaIS0_EE_pfor.body.cilk(%"class.std::vector.0"* {{.*}}sret {{.*}}%agg.result.cilk,

attributes #0 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { argmemonly }
attributes #5 = { nobuiltin nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { noreturn "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nobuiltin "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { nounwind }
attributes #9 = { noreturn }

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
!20 = distinct !{!20, !21, !19}
!21 = !{!"llvm.loop.unroll.runtime.disable"}
!22 = !{!23, !15, i64 0}
!23 = !{!"_ZTSSt10_Head_baseILm2EiLb0EE", !15, i64 0}
!24 = !{!25}
!25 = distinct !{!25, !26, !"_ZSt10make_tupleIJRiRdS0_EESt5tupleIJDpNSt17__decay_and_stripIT_E6__typeEEEDpOS4_: %agg.result"}
!26 = distinct !{!26, !"_ZSt10make_tupleIJRiRdS0_EESt5tupleIJDpNSt17__decay_and_stripIT_E6__typeEEEDpOS4_"}
!27 = !{!28, !29, i64 0}
!28 = !{!"_ZTSSt10_Head_baseILm1EdLb0EE", !29, i64 0}
!29 = !{!"double", !6, i64 0}
!30 = !{!31, !15, i64 0}
!31 = !{!"_ZTSSt10_Head_baseILm0EiLb0EE", !15, i64 0}
!32 = distinct !{!32, !17}
!33 = !{!5, !5, i64 0}
!34 = !{!29, !29, i64 0}
