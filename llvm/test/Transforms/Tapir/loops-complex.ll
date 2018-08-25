; RUN: opt < %s -passes=loop-spawning -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.timer = type <{ double, double, double, i8, [3 x i8], %struct.timezone, [4 x i8] }>
%struct.timezone = type { i32, i32 }
%"class.std::basic_ostream" = type { i32 (...)**, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", %"class.std::basic_ostream"*, i8, i8, %"class.std::basic_streambuf"*, %"class.std::ctype"*, %"class.std::num_put"*, %"class.std::num_get"* }
%"class.std::ios_base" = type { i32 (...)**, i64, i64, i32, i32, i32, %"struct.std::ios_base::_Callback_list"*, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, %"struct.std::ios_base::_Words"*, %"class.std::locale" }
%"struct.std::ios_base::_Callback_list" = type { %"struct.std::ios_base::_Callback_list"*, void (i32, %"class.std::ios_base"*, i32)*, i32, i32 }
%"struct.std::ios_base::_Words" = type { i8*, i64 }
%"class.std::locale" = type { %"class.std::locale::_Impl"* }
%"class.std::locale::_Impl" = type { i32, %"class.std::locale::facet"**, i64, %"class.std::locale::facet"**, i8** }
%"class.std::locale::facet" = type <{ i32 (...)**, i32, [4 x i8] }>
%"class.std::basic_streambuf" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %"class.std::locale" }
%"class.std::ctype" = type <{ %"class.std::locale::facet.base", [4 x i8], %struct.__locale_struct*, i8, [7 x i8], i32*, i32*, i16*, i8, [256 x i8], [256 x i8], i8, [6 x i8] }>
%"class.std::locale::facet.base" = type <{ i32 (...)**, i32 }>
%struct.__locale_struct = type { [13 x %struct.__locale_data*], i16*, i32*, i32*, [13 x i8*] }
%struct.__locale_data = type opaque
%"class.std::num_put" = type { %"class.std::locale::facet.base", [4 x i8] }
%"class.std::num_get" = type { %"class.std::locale::facet.base", [4 x i8] }
%"struct.std::pair" = type { i32, i32 }
%struct.seg = type { i32, i32 }
%struct.timeval = type { i64, i64 }
%"class.std::__cxx11::basic_string" = type { %"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider", i64, %union.anon }
%"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" = type { i8* }
%union.anon = type { i64, [8 x i8] }

@_ZL3_tm = internal global %struct.timer zeroinitializer, align 8
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str.6 = private unnamed_addr constant [5 x i8] c"m = \00", align 1
@.str.7 = private unnamed_addr constant [6 x i8] c" n = \00", align 1
@.str.8 = private unnamed_addr constant [3 x i8] c", \00", align 1
@.str.11 = private unnamed_addr constant [6 x i8] c"split\00", align 1
@.str.12 = private unnamed_addr constant [31 x i8] c"Suffix Array:  Too many rounds\00", align 1
@.str.13 = private unnamed_addr constant [9 x i8] c"nSegs = \00", align 1
@.str.14 = private unnamed_addr constant [10 x i8] c" nKeys = \00", align 1
@.str.15 = private unnamed_addr constant [18 x i8] c" common length = \00", align 1
@.str.16 = private unnamed_addr constant [16 x i8] c"filter and scan\00", align 1
@.str.17 = private unnamed_addr constant [4 x i8] c" : \00", align 1

; Function Attrs: uwtable
define i32* @_Z19suffixArrayInternalPhl(i8* nocapture readonly %ss, i64 %n) local_unnamed_addr #4 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %now.i.i.i.i933 = alloca %struct.timeval, align 8
  %now.i.i.i.i885 = alloca %struct.timeval, align 8
  %now.i.i.i.i837 = alloca %struct.timeval, align 8
  %syncreg.i770 = tail call token @llvm.syncregion.start()
  %__dnew.i.i.i.i744 = alloca i64, align 8
  %now.i.i.i.i718 = alloca %struct.timeval, align 8
  %now.i.i.i.i675 = alloca %struct.timeval, align 8
  %now.i.i.i.i = alloca %struct.timeval, align 8
  %now.i.i = alloca %struct.timeval, align 8
  %flags = alloca [256 x i32], align 16
  %0 = bitcast [256 x i32]* %flags to i8*
  %syncreg = tail call token @llvm.syncregion.start()
  %syncreg24 = tail call token @llvm.syncregion.start()
  %syncreg77 = tail call token @llvm.syncregion.start()
  %agg.tmp124 = alloca %"class.std::__cxx11::basic_string", align 8
  %agg.tmp129 = alloca %"class.std::__cxx11::basic_string", align 8
  %agg.tmp144 = alloca %"class.std::__cxx11::basic_string", align 8
  %agg.tmp161 = alloca %"class.std::__cxx11::basic_string", align 8
  %syncreg175 = tail call token @llvm.syncregion.start()
  %agg.tmp210 = alloca %"class.std::__cxx11::basic_string", align 8
  %syncreg219 = tail call token @llvm.syncregion.start()
  %agg.tmp303 = alloca %"class.std::__cxx11::basic_string", align 8
  %syncreg312 = tail call token @llvm.syncregion.start()
  %agg.tmp365 = alloca %"class.std::__cxx11::basic_string", align 8
  %syncreg376 = tail call token @llvm.syncregion.start()
  store i8 1, i8* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 3), align 8, !tbaa !17
  %1 = bitcast %struct.timeval* %now.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %1) #2
  %call.i.i = call i32 @gettimeofday(%struct.timeval* nonnull %now.i.i, %struct.timezone* nonnull getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 5)) #2
  %tv_sec.i.i = getelementptr inbounds %struct.timeval, %struct.timeval* %now.i.i, i64 0, i32 0
  %2 = load i64, i64* %tv_sec.i.i, align 8, !tbaa !23
  %conv.i.i = sitofp i64 %2 to double
  %tv_usec.i.i = getelementptr inbounds %struct.timeval, %struct.timeval* %now.i.i, i64 0, i32 1
  %3 = load i64, i64* %tv_usec.i.i, align 8, !tbaa !25
  %conv2.i.i = sitofp i64 %3 to double
  %div.i.i = fmul fast double %conv2.i.i, 0x3EB0C6F7A0B5ED8D
  %add.i.i = fadd fast double %div.i.i, %conv.i.i
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %1) #2
  store double %add.i.i, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 1), align 8, !tbaa !26
  %add = add i64 %n, 48
  %call = tail call noalias i8* @malloc(i64 %add) #2
  call void @llvm.lifetime.start.p0i8(i64 1024, i8* nonnull %0) #2
  call void @llvm.memset.p0i8.i64(i8* nonnull %0, i8 0, i64 1024, i32 16, i1 false)
  %conv = trunc i64 %n to i32
  %cmp31094 = icmp eq i32 %conv, 0
  br i1 %cmp31094, label %pfor.cond.cleanup, label %pfor.detach.lr.ph

pfor.detach.lr.ph:                                ; preds = %entry
  %wide.trip.count1148 = and i64 %n, 4294967295
  br label %pfor.detach

pfor.cond.cleanup:                                ; preds = %pfor.inc, %entry
  sync within %syncreg, label %pfor.cond.cleanup.split

pfor.cond.cleanup.split:                          ; preds = %pfor.cond.cleanup
  br label %for.body41.i.i

pfor.detach:                                      ; preds = %pfor.inc, %pfor.detach.lr.ph
  %indvars.iv1146 = phi i64 [ 0, %pfor.detach.lr.ph ], [ %indvars.iv.next1147, %pfor.inc ]
  detach within %syncreg, label %pfor.body, label %pfor.inc

pfor.body:                                        ; preds = %pfor.detach
  %arrayidx8 = getelementptr inbounds i8, i8* %ss, i64 %indvars.iv1146
  %4 = load i8, i8* %arrayidx8, align 1, !tbaa !16
  %idxprom9 = zext i8 %4 to i64
  %arrayidx10 = getelementptr inbounds [256 x i32], [256 x i32]* %flags, i64 0, i64 %idxprom9
  %5 = load i32, i32* %arrayidx10, align 4, !tbaa !6
  %tobool = icmp eq i32 %5, 0
  br i1 %tobool, label %if.then, label %pfor.preattach

if.then:                                          ; preds = %pfor.body
  store i32 1, i32* %arrayidx10, align 4, !tbaa !6
  br label %pfor.preattach

pfor.preattach:                                   ; preds = %pfor.body, %if.then
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.preattach, %pfor.detach
  %indvars.iv.next1147 = add nuw nsw i64 %indvars.iv1146, 1
  %exitcond1149 = icmp eq i64 %indvars.iv.next1147, %wide.trip.count1148
  br i1 %exitcond1149, label %pfor.cond.cleanup, label %pfor.detach, !llvm.loop !34

for.body41.i.i:                                   ; preds = %for.body41.i.i, %pfor.cond.cleanup.split
  %indvars.iv125.i.i = phi i64 [ 0, %pfor.cond.cleanup.split ], [ %indvars.iv.next126.i.i.7, %for.body41.i.i ]
  %r.3119.i.i = phi i32 [ 1, %pfor.cond.cleanup.split ], [ %add.i.i.i.7, %for.body41.i.i ]
  %arrayidx.i79.i.i = getelementptr inbounds [256 x i32], [256 x i32]* %flags, i64 0, i64 %indvars.iv125.i.i
  %6 = load i32, i32* %arrayidx.i79.i.i, align 16, !tbaa !6
  store i32 %r.3119.i.i, i32* %arrayidx.i79.i.i, align 16, !tbaa !6
  %add.i.i.i = add i32 %6, %r.3119.i.i
  %indvars.iv.next126.i.i = or i64 %indvars.iv125.i.i, 1
  %arrayidx.i79.i.i.1 = getelementptr inbounds [256 x i32], [256 x i32]* %flags, i64 0, i64 %indvars.iv.next126.i.i
  %7 = load i32, i32* %arrayidx.i79.i.i.1, align 4, !tbaa !6
  store i32 %add.i.i.i, i32* %arrayidx.i79.i.i.1, align 4, !tbaa !6
  %add.i.i.i.1 = add i32 %7, %add.i.i.i
  %indvars.iv.next126.i.i.1 = or i64 %indvars.iv125.i.i, 2
  %arrayidx.i79.i.i.2 = getelementptr inbounds [256 x i32], [256 x i32]* %flags, i64 0, i64 %indvars.iv.next126.i.i.1
  %8 = load i32, i32* %arrayidx.i79.i.i.2, align 8, !tbaa !6
  store i32 %add.i.i.i.1, i32* %arrayidx.i79.i.i.2, align 8, !tbaa !6
  %add.i.i.i.2 = add i32 %8, %add.i.i.i.1
  %indvars.iv.next126.i.i.2 = or i64 %indvars.iv125.i.i, 3
  %arrayidx.i79.i.i.3 = getelementptr inbounds [256 x i32], [256 x i32]* %flags, i64 0, i64 %indvars.iv.next126.i.i.2
  %9 = load i32, i32* %arrayidx.i79.i.i.3, align 4, !tbaa !6
  store i32 %add.i.i.i.2, i32* %arrayidx.i79.i.i.3, align 4, !tbaa !6
  %add.i.i.i.3 = add i32 %9, %add.i.i.i.2
  %indvars.iv.next126.i.i.3 = or i64 %indvars.iv125.i.i, 4
  %arrayidx.i79.i.i.4 = getelementptr inbounds [256 x i32], [256 x i32]* %flags, i64 0, i64 %indvars.iv.next126.i.i.3
  %10 = load i32, i32* %arrayidx.i79.i.i.4, align 16, !tbaa !6
  store i32 %add.i.i.i.3, i32* %arrayidx.i79.i.i.4, align 16, !tbaa !6
  %add.i.i.i.4 = add i32 %10, %add.i.i.i.3
  %indvars.iv.next126.i.i.4 = or i64 %indvars.iv125.i.i, 5
  %arrayidx.i79.i.i.5 = getelementptr inbounds [256 x i32], [256 x i32]* %flags, i64 0, i64 %indvars.iv.next126.i.i.4
  %11 = load i32, i32* %arrayidx.i79.i.i.5, align 4, !tbaa !6
  store i32 %add.i.i.i.4, i32* %arrayidx.i79.i.i.5, align 4, !tbaa !6
  %add.i.i.i.5 = add i32 %11, %add.i.i.i.4
  %indvars.iv.next126.i.i.5 = or i64 %indvars.iv125.i.i, 6
  %arrayidx.i79.i.i.6 = getelementptr inbounds [256 x i32], [256 x i32]* %flags, i64 0, i64 %indvars.iv.next126.i.i.5
  %12 = load i32, i32* %arrayidx.i79.i.i.6, align 8, !tbaa !6
  store i32 %add.i.i.i.5, i32* %arrayidx.i79.i.i.6, align 8, !tbaa !6
  %add.i.i.i.6 = add i32 %12, %add.i.i.i.5
  %indvars.iv.next126.i.i.6 = or i64 %indvars.iv125.i.i, 7
  %arrayidx.i79.i.i.7 = getelementptr inbounds [256 x i32], [256 x i32]* %flags, i64 0, i64 %indvars.iv.next126.i.i.6
  %13 = load i32, i32* %arrayidx.i79.i.i.7, align 4, !tbaa !6
  store i32 %add.i.i.i.6, i32* %arrayidx.i79.i.i.7, align 4, !tbaa !6
  %add.i.i.i.7 = add i32 %13, %add.i.i.i.6
  %indvars.iv.next126.i.i.7 = add nuw nsw i64 %indvars.iv125.i.i, 8
  %exitcond128.i.i.7 = icmp eq i64 %indvars.iv.next126.i.i.7, 256
  br i1 %exitcond128.i.i.7, label %_ZN8sequence4scanIjiN5utils4addFIjEENS_4getAIjiEEEET_PS6_T0_S8_T1_T2_S6_bb.exit, label %for.body41.i.i

_ZN8sequence4scanIjiN5utils4addFIjEENS_4getAIjiEEEET_PS6_T0_S8_T1_T2_S6_bb.exit: ; preds = %for.body41.i.i
  %call1.i = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull getelementptr inbounds ([5 x i8], [5 x i8]* @.str.6, i64 0, i64 0), i64 4)
  %conv.i = zext i32 %add.i.i.i.7 to i64
  %call.i = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertImEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, i64 %conv.i)
  %call1.i627 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) %call.i, i8* nonnull getelementptr inbounds ([6 x i8], [6 x i8]* @.str.7, i64 0, i64 0), i64 5)
  %call.i628 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIlEERSoT_(%"class.std::basic_ostream"* nonnull %call.i, i64 %n)
  %14 = bitcast %"class.std::basic_ostream"* %call.i628 to i8**
  %vtable.i = load i8*, i8** %14, align 8, !tbaa !35
  %vbase.offset.ptr.i = getelementptr i8, i8* %vtable.i, i64 -24
  %15 = bitcast i8* %vbase.offset.ptr.i to i64*
  %vbase.offset.i = load i64, i64* %15, align 8
  %16 = bitcast %"class.std::basic_ostream"* %call.i628 to i8*
  %add.ptr.i = getelementptr inbounds i8, i8* %16, i64 %vbase.offset.i
  %_M_ctype.i = getelementptr inbounds i8, i8* %add.ptr.i, i64 240
  %17 = bitcast i8* %_M_ctype.i to %"class.std::ctype"**
  %18 = load %"class.std::ctype"*, %"class.std::ctype"** %17, align 8, !tbaa !37
  %tobool.i990 = icmp eq %"class.std::ctype"* %18, null
  br i1 %tobool.i990, label %if.then.i991, label %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit

if.then.i991:                                     ; preds = %_ZN8sequence4scanIjiN5utils4addFIjEENS_4getAIjiEEEET_PS6_T0_S8_T1_T2_S6_bb.exit
  tail call void @_ZSt16__throw_bad_castv() #12
  unreachable

_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit:    ; preds = %_ZN8sequence4scanIjiN5utils4addFIjEENS_4getAIjiEEEET_PS6_T0_S8_T1_T2_S6_bb.exit
  %_M_widen_ok.i = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %18, i64 0, i32 8
  %19 = load i8, i8* %_M_widen_ok.i, align 8, !tbaa !39
  %tobool.i960 = icmp eq i8 %19, 0
  br i1 %tobool.i960, label %if.end.i, label %if.then.i962

if.then.i962:                                     ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit
  %arrayidx.i961 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %18, i64 0, i32 9, i64 10
  %20 = load i8, i8* %arrayidx.i961, align 1, !tbaa !16
  br label %_ZNKSt5ctypeIcE5widenEc.exit

if.end.i:                                         ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %18)
  %21 = bitcast %"class.std::ctype"* %18 to i8 (%"class.std::ctype"*, i8)***
  %vtable.i963 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %21, align 8, !tbaa !35
  %vfn.i = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vtable.i963, i64 6
  %22 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vfn.i, align 8
  %call.i964 = tail call signext i8 %22(%"class.std::ctype"* nonnull %18, i8 signext 10)
  br label %_ZNKSt5ctypeIcE5widenEc.exit

_ZNKSt5ctypeIcE5widenEc.exit:                     ; preds = %if.then.i962, %if.end.i
  %retval.0.i965 = phi i8 [ %20, %if.then.i962 ], [ %call.i964, %if.end.i ]
  %call1.i631 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %call.i628, i8 signext %retval.0.i965)
  %call.i632 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %call1.i631)
  br i1 %cmp31094, label %pfor.cond.cleanup35, label %pfor.detach36.lr.ph

pfor.detach36.lr.ph:                              ; preds = %_ZNKSt5ctypeIcE5widenEc.exit
  %wide.trip.count1144 = and i64 %n, 4294967295
  br label %pfor.detach36

pfor.cond.cleanup35:                              ; preds = %pfor.inc50, %_ZNKSt5ctypeIcE5widenEc.exit
  sync within %syncreg24, label %sync.continue52

pfor.detach36:                                    ; preds = %pfor.inc50, %pfor.detach36.lr.ph
  %indvars.iv1142 = phi i64 [ 0, %pfor.detach36.lr.ph ], [ %indvars.iv.next1143, %pfor.inc50 ]
  detach within %syncreg24, label %pfor.body41, label %pfor.inc50

pfor.body41:                                      ; preds = %pfor.detach36
  %arrayidx43 = getelementptr inbounds i8, i8* %ss, i64 %indvars.iv1142
  %23 = load i8, i8* %arrayidx43, align 1, !tbaa !16
  %idxprom44 = zext i8 %23 to i64
  %arrayidx45 = getelementptr inbounds [256 x i32], [256 x i32]* %flags, i64 0, i64 %idxprom44
  %24 = load i32, i32* %arrayidx45, align 4, !tbaa !6
  %conv46 = trunc i32 %24 to i8
  %arrayidx48 = getelementptr inbounds i8, i8* %call, i64 %indvars.iv1142
  store i8 %conv46, i8* %arrayidx48, align 1, !tbaa !16
  reattach within %syncreg24, label %pfor.inc50

pfor.inc50:                                       ; preds = %pfor.body41, %pfor.detach36
  %indvars.iv.next1143 = add nuw nsw i64 %indvars.iv1142, 1
  %exitcond1145 = icmp eq i64 %indvars.iv.next1143, %wide.trip.count1144
  br i1 %exitcond1145, label %pfor.cond.cleanup35, label %pfor.detach36, !llvm.loop !41

sync.continue52:                                  ; preds = %pfor.cond.cleanup35
  %conv571088 = and i64 %n, 4294967295
  %cmp591089 = icmp ugt i64 %add, %conv571088
  br i1 %cmp591089, label %for.body62.lr.ph, label %for.cond.cleanup60

for.body62.lr.ph:                                 ; preds = %sync.continue52
  %25 = add i64 %n, 1
  %26 = and i64 %25, 4294967295
  %27 = icmp ugt i64 %add, %26
  %umax = select i1 %27, i64 %add, i64 %26
  %28 = add i64 %umax, 1
  %29 = sub i64 %28, %26
  %min.iters.check = icmp ult i64 %29, 128
  br i1 %min.iters.check, label %for.body62.preheader, label %vector.scevcheck

for.body62.preheader:                             ; preds = %middle.block, %vector.scevcheck, %for.body62.lr.ph
  %conv571091.ph = phi i64 [ %conv571088, %vector.scevcheck ], [ %conv571088, %for.body62.lr.ph ], [ %ind.end, %middle.block ]
  %i54.01090.ph = phi i32 [ %conv, %vector.scevcheck ], [ %conv, %for.body62.lr.ph ], [ %ind.end1180, %middle.block ]
  br label %for.body62

vector.scevcheck:                                 ; preds = %for.body62.lr.ph
  %30 = add i32 %conv, 1
  %31 = zext i32 %30 to i64
  %32 = icmp ugt i64 %add, %31
  %umax1175 = select i1 %32, i64 %add, i64 %31
  %33 = sub i64 %umax1175, %31
  %34 = trunc i64 %33 to i32
  %35 = add i32 %conv, %34
  %36 = icmp ult i32 %35, %conv
  %37 = icmp ugt i64 %33, 4294967295
  %38 = or i1 %36, %37
  %39 = trunc i64 %33 to i32
  %40 = add i32 %30, %39
  %41 = icmp ult i32 %40, %30
  %42 = icmp ugt i64 %33, 4294967295
  %43 = or i1 %41, %42
  %44 = or i1 %38, %43
  br i1 %44, label %for.body62.preheader, label %vector.ph

vector.ph:                                        ; preds = %vector.scevcheck
  %n.vec = and i64 %29, -128
  %ind.end = add i64 %conv571088, %n.vec
  %cast.crd = trunc i64 %n.vec to i32
  %ind.end1180 = add i32 %conv, %cast.crd
  %45 = add i64 %n.vec, -128
  %46 = lshr exact i64 %45, 7
  %47 = add nuw nsw i64 %46, 1
  %xtraiter1209 = and i64 %47, 7
  %48 = icmp ult i64 %45, 896
  br i1 %48, label %middle.block.unr-lcssa, label %vector.ph.new

vector.ph.new:                                    ; preds = %vector.ph
  %unroll_iter1212 = sub nsw i64 %47, %xtraiter1209
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph.new
  %index = phi i64 [ 0, %vector.ph.new ], [ %index.next.7, %vector.body ]
  %niter1213 = phi i64 [ %unroll_iter1212, %vector.ph.new ], [ %niter1213.nsub.7, %vector.body ]
  %49 = add i64 %conv571088, %index
  %50 = getelementptr inbounds i8, i8* %call, i64 %49
  %51 = bitcast i8* %50 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %51, align 1, !tbaa !16
  %52 = getelementptr i8, i8* %50, i64 32
  %53 = bitcast i8* %52 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %53, align 1, !tbaa !16
  %54 = getelementptr i8, i8* %50, i64 64
  %55 = bitcast i8* %54 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %55, align 1, !tbaa !16
  %56 = getelementptr i8, i8* %50, i64 96
  %57 = bitcast i8* %56 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %57, align 1, !tbaa !16
  %index.next = or i64 %index, 128
  %58 = add i64 %conv571088, %index.next
  %59 = getelementptr inbounds i8, i8* %call, i64 %58
  %60 = bitcast i8* %59 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %60, align 1, !tbaa !16
  %61 = getelementptr i8, i8* %59, i64 32
  %62 = bitcast i8* %61 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %62, align 1, !tbaa !16
  %63 = getelementptr i8, i8* %59, i64 64
  %64 = bitcast i8* %63 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %64, align 1, !tbaa !16
  %65 = getelementptr i8, i8* %59, i64 96
  %66 = bitcast i8* %65 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %66, align 1, !tbaa !16
  %index.next.1 = or i64 %index, 256
  %67 = add i64 %conv571088, %index.next.1
  %68 = getelementptr inbounds i8, i8* %call, i64 %67
  %69 = bitcast i8* %68 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %69, align 1, !tbaa !16
  %70 = getelementptr i8, i8* %68, i64 32
  %71 = bitcast i8* %70 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %71, align 1, !tbaa !16
  %72 = getelementptr i8, i8* %68, i64 64
  %73 = bitcast i8* %72 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %73, align 1, !tbaa !16
  %74 = getelementptr i8, i8* %68, i64 96
  %75 = bitcast i8* %74 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %75, align 1, !tbaa !16
  %index.next.2 = or i64 %index, 384
  %76 = add i64 %conv571088, %index.next.2
  %77 = getelementptr inbounds i8, i8* %call, i64 %76
  %78 = bitcast i8* %77 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %78, align 1, !tbaa !16
  %79 = getelementptr i8, i8* %77, i64 32
  %80 = bitcast i8* %79 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %80, align 1, !tbaa !16
  %81 = getelementptr i8, i8* %77, i64 64
  %82 = bitcast i8* %81 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %82, align 1, !tbaa !16
  %83 = getelementptr i8, i8* %77, i64 96
  %84 = bitcast i8* %83 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %84, align 1, !tbaa !16
  %index.next.3 = or i64 %index, 512
  %85 = add i64 %conv571088, %index.next.3
  %86 = getelementptr inbounds i8, i8* %call, i64 %85
  %87 = bitcast i8* %86 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %87, align 1, !tbaa !16
  %88 = getelementptr i8, i8* %86, i64 32
  %89 = bitcast i8* %88 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %89, align 1, !tbaa !16
  %90 = getelementptr i8, i8* %86, i64 64
  %91 = bitcast i8* %90 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %91, align 1, !tbaa !16
  %92 = getelementptr i8, i8* %86, i64 96
  %93 = bitcast i8* %92 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %93, align 1, !tbaa !16
  %index.next.4 = or i64 %index, 640
  %94 = add i64 %conv571088, %index.next.4
  %95 = getelementptr inbounds i8, i8* %call, i64 %94
  %96 = bitcast i8* %95 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %96, align 1, !tbaa !16
  %97 = getelementptr i8, i8* %95, i64 32
  %98 = bitcast i8* %97 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %98, align 1, !tbaa !16
  %99 = getelementptr i8, i8* %95, i64 64
  %100 = bitcast i8* %99 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %100, align 1, !tbaa !16
  %101 = getelementptr i8, i8* %95, i64 96
  %102 = bitcast i8* %101 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %102, align 1, !tbaa !16
  %index.next.5 = or i64 %index, 768
  %103 = add i64 %conv571088, %index.next.5
  %104 = getelementptr inbounds i8, i8* %call, i64 %103
  %105 = bitcast i8* %104 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %105, align 1, !tbaa !16
  %106 = getelementptr i8, i8* %104, i64 32
  %107 = bitcast i8* %106 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %107, align 1, !tbaa !16
  %108 = getelementptr i8, i8* %104, i64 64
  %109 = bitcast i8* %108 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %109, align 1, !tbaa !16
  %110 = getelementptr i8, i8* %104, i64 96
  %111 = bitcast i8* %110 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %111, align 1, !tbaa !16
  %index.next.6 = or i64 %index, 896
  %112 = add i64 %conv571088, %index.next.6
  %113 = getelementptr inbounds i8, i8* %call, i64 %112
  %114 = bitcast i8* %113 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %114, align 1, !tbaa !16
  %115 = getelementptr i8, i8* %113, i64 32
  %116 = bitcast i8* %115 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %116, align 1, !tbaa !16
  %117 = getelementptr i8, i8* %113, i64 64
  %118 = bitcast i8* %117 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %118, align 1, !tbaa !16
  %119 = getelementptr i8, i8* %113, i64 96
  %120 = bitcast i8* %119 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %120, align 1, !tbaa !16
  %index.next.7 = add i64 %index, 1024
  %niter1213.nsub.7 = add i64 %niter1213, -8
  %niter1213.ncmp.7 = icmp eq i64 %niter1213.nsub.7, 0
  br i1 %niter1213.ncmp.7, label %middle.block.unr-lcssa, label %vector.body, !llvm.loop !42

middle.block.unr-lcssa:                           ; preds = %vector.body, %vector.ph
  %index.unr = phi i64 [ 0, %vector.ph ], [ %index.next.7, %vector.body ]
  %lcmp.mod1211 = icmp eq i64 %xtraiter1209, 0
  br i1 %lcmp.mod1211, label %middle.block, label %vector.body.epil.preheader

vector.body.epil.preheader:                       ; preds = %middle.block.unr-lcssa
  br label %vector.body.epil

vector.body.epil:                                 ; preds = %vector.body.epil, %vector.body.epil.preheader
  %index.epil = phi i64 [ %index.unr, %vector.body.epil.preheader ], [ %index.next.epil, %vector.body.epil ]
  %epil.iter1210 = phi i64 [ %xtraiter1209, %vector.body.epil.preheader ], [ %epil.iter1210.sub, %vector.body.epil ]
  %121 = add i64 %conv571088, %index.epil
  %122 = getelementptr inbounds i8, i8* %call, i64 %121
  %123 = bitcast i8* %122 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %123, align 1, !tbaa !16
  %124 = getelementptr i8, i8* %122, i64 32
  %125 = bitcast i8* %124 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %125, align 1, !tbaa !16
  %126 = getelementptr i8, i8* %122, i64 64
  %127 = bitcast i8* %126 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %127, align 1, !tbaa !16
  %128 = getelementptr i8, i8* %122, i64 96
  %129 = bitcast i8* %128 to <32 x i8>*
  store <32 x i8> zeroinitializer, <32 x i8>* %129, align 1, !tbaa !16
  %index.next.epil = add i64 %index.epil, 128
  %epil.iter1210.sub = add i64 %epil.iter1210, -1
  %epil.iter1210.cmp = icmp eq i64 %epil.iter1210.sub, 0
  br i1 %epil.iter1210.cmp, label %middle.block, label %vector.body.epil, !llvm.loop !44

middle.block:                                     ; preds = %vector.body.epil, %middle.block.unr-lcssa
  %cmp.n = icmp eq i64 %29, %n.vec
  br i1 %cmp.n, label %for.cond.cleanup60, label %for.body62.preheader

for.cond.cleanup60:                               ; preds = %for.body62, %middle.block, %sync.continue52
  %conv.i633 = uitofp i32 %add.i.i.i.7 to double
  %130 = tail call fast double @llvm.log2.f64(double %conv.i633) #2
  %div69 = fdiv fast double 9.600000e+01, %130
  %131 = tail call fast double @llvm.floor.f64(double %div69)
  %conv70 = fptoui double %131 to i32
  %call.i634 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, double %130)
  %call1.i636 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) %call.i634, i8* nonnull getelementptr inbounds ([3 x i8], [3 x i8]* @.str.8, i64 0, i64 0), i64 2)
  %conv.i637 = zext i32 %conv70 to i64
  %call.i638 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertImEERSoT_(%"class.std::basic_ostream"* nonnull %call.i634, i64 %conv.i637)
  %132 = bitcast %"class.std::basic_ostream"* %call.i638 to i8**
  %vtable.i640 = load i8*, i8** %132, align 8, !tbaa !35
  %vbase.offset.ptr.i641 = getelementptr i8, i8* %vtable.i640, i64 -24
  %133 = bitcast i8* %vbase.offset.ptr.i641 to i64*
  %vbase.offset.i642 = load i64, i64* %133, align 8
  %134 = bitcast %"class.std::basic_ostream"* %call.i638 to i8*
  %add.ptr.i643 = getelementptr inbounds i8, i8* %134, i64 %vbase.offset.i642
  %_M_ctype.i966 = getelementptr inbounds i8, i8* %add.ptr.i643, i64 240
  %135 = bitcast i8* %_M_ctype.i966 to %"class.std::ctype"**
  %136 = load %"class.std::ctype"*, %"class.std::ctype"** %135, align 8, !tbaa !37
  %tobool.i993 = icmp eq %"class.std::ctype"* %136, null
  br i1 %tobool.i993, label %if.then.i994, label %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit996

if.then.i994:                                     ; preds = %for.cond.cleanup60
  tail call void @_ZSt16__throw_bad_castv() #12
  unreachable

_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit996: ; preds = %for.cond.cleanup60
  %_M_widen_ok.i968 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %136, i64 0, i32 8
  %137 = load i8, i8* %_M_widen_ok.i968, align 8, !tbaa !39
  %tobool.i969 = icmp eq i8 %137, 0
  br i1 %tobool.i969, label %if.end.i975, label %if.then.i971

if.then.i971:                                     ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit996
  %arrayidx.i970 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %136, i64 0, i32 9, i64 10
  %138 = load i8, i8* %arrayidx.i970, align 1, !tbaa !16
  br label %_ZNKSt5ctypeIcE5widenEc.exit977

if.end.i975:                                      ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit996
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %136)
  %139 = bitcast %"class.std::ctype"* %136 to i8 (%"class.std::ctype"*, i8)***
  %vtable.i972 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %139, align 8, !tbaa !35
  %vfn.i973 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vtable.i972, i64 6
  %140 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vfn.i973, align 8
  %call.i974 = tail call signext i8 %140(%"class.std::ctype"* nonnull %136, i8 signext 10)
  br label %_ZNKSt5ctypeIcE5widenEc.exit977

_ZNKSt5ctypeIcE5widenEc.exit977:                  ; preds = %if.then.i971, %if.end.i975
  %retval.0.i976 = phi i8 [ %138, %if.then.i971 ], [ %call.i974, %if.end.i975 ]
  %call1.i645 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %call.i638, i8 signext %retval.0.i976)
  %call.i646 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %call1.i645)
  %mul75 = shl i64 %n, 4
  %call76 = tail call noalias i8* @malloc(i64 %mul75) #2
  %141 = bitcast i8* %call76 to i128*
  br i1 %cmp31094, label %pfor.cond.cleanup88, label %pfor.detach90.lr.ph

pfor.detach90.lr.ph:                              ; preds = %_ZNKSt5ctypeIcE5widenEc.exit977
  %cmp1001081 = icmp ugt i32 %conv70, 1
  %conv104 = zext i32 %add.i.i.i.7 to i128
  %wide.trip.count1136 = and i64 %n, 4294967295
  br i1 %cmp1001081, label %pfor.detach90.us.preheader, label %pfor.detach90.preheader

pfor.detach90.preheader:                          ; preds = %pfor.detach90.lr.ph
  br label %pfor.detach90

pfor.detach90.us.preheader:                       ; preds = %pfor.detach90.lr.ph
  %142 = add nsw i64 %conv.i637, -1
  %143 = add nsw i64 %conv.i637, -2
  %144 = add i32 %conv70, 7
  %145 = and i32 %144, 7
  %xtraiter1203 = zext i32 %145 to i64
  %146 = icmp ult i64 %143, 7
  %unroll_iter1207 = sub nsw i64 %142, %xtraiter1203
  %lcmp.mod1205 = icmp eq i32 %145, 0
  br label %pfor.detach90.us

pfor.detach90.us:                                 ; preds = %pfor.detach90.us.preheader, %pfor.inc119.us
  %indvars.iv1134 = phi i64 [ %indvars.iv.next1135, %pfor.inc119.us ], [ 0, %pfor.detach90.us.preheader ]
  detach within %syncreg77, label %for.body103.lr.ph.us, label %pfor.inc119.us

pfor.inc119.us:                                   ; preds = %for.cond99.for.cond.cleanup101_crit_edge.us, %pfor.detach90.us
  %indvars.iv.next1135 = add nuw nsw i64 %indvars.iv1134, 1
  %exitcond1137 = icmp eq i64 %indvars.iv.next1135, %wide.trip.count1136
  br i1 %exitcond1137, label %pfor.cond.cleanup88, label %pfor.detach90.us, !llvm.loop !46

for.body103.us:                                   ; preds = %for.body103.us, %for.body103.lr.ph.us.new
  %indvars.iv1130 = phi i64 [ 1, %for.body103.lr.ph.us.new ], [ %indvars.iv.next1131.7, %for.body103.us ]
  %r.01082.us = phi i128 [ %conv98.us, %for.body103.lr.ph.us.new ], [ %add110.us.7, %for.body103.us ]
  %niter1208 = phi i64 [ %unroll_iter1207, %for.body103.lr.ph.us.new ], [ %niter1208.nsub.7, %for.body103.us ]
  %mul105.us = mul i128 %r.01082.us, %conv104
  %add106.us = add nuw i64 %indvars.iv1130, %indvars.iv1134
  %idxprom107.us = and i64 %add106.us, 4294967295
  %arrayidx108.us = getelementptr inbounds i8, i8* %call, i64 %idxprom107.us
  %147 = load i8, i8* %arrayidx108.us, align 1, !tbaa !16
  %conv109.us = zext i8 %147 to i128
  %add110.us = add i128 %mul105.us, %conv109.us
  %indvars.iv.next1131 = add nuw nsw i64 %indvars.iv1130, 1
  %mul105.us.1 = mul i128 %add110.us, %conv104
  %add106.us.1 = add nuw i64 %indvars.iv.next1131, %indvars.iv1134
  %idxprom107.us.1 = and i64 %add106.us.1, 4294967295
  %arrayidx108.us.1 = getelementptr inbounds i8, i8* %call, i64 %idxprom107.us.1
  %148 = load i8, i8* %arrayidx108.us.1, align 1, !tbaa !16
  %conv109.us.1 = zext i8 %148 to i128
  %add110.us.1 = add i128 %mul105.us.1, %conv109.us.1
  %indvars.iv.next1131.1 = add nuw nsw i64 %indvars.iv1130, 2
  %mul105.us.2 = mul i128 %add110.us.1, %conv104
  %add106.us.2 = add nuw i64 %indvars.iv.next1131.1, %indvars.iv1134
  %idxprom107.us.2 = and i64 %add106.us.2, 4294967295
  %arrayidx108.us.2 = getelementptr inbounds i8, i8* %call, i64 %idxprom107.us.2
  %149 = load i8, i8* %arrayidx108.us.2, align 1, !tbaa !16
  %conv109.us.2 = zext i8 %149 to i128
  %add110.us.2 = add i128 %mul105.us.2, %conv109.us.2
  %indvars.iv.next1131.2 = add nuw nsw i64 %indvars.iv1130, 3
  %mul105.us.3 = mul i128 %add110.us.2, %conv104
  %add106.us.3 = add nuw i64 %indvars.iv.next1131.2, %indvars.iv1134
  %idxprom107.us.3 = and i64 %add106.us.3, 4294967295
  %arrayidx108.us.3 = getelementptr inbounds i8, i8* %call, i64 %idxprom107.us.3
  %150 = load i8, i8* %arrayidx108.us.3, align 1, !tbaa !16
  %conv109.us.3 = zext i8 %150 to i128
  %add110.us.3 = add i128 %mul105.us.3, %conv109.us.3
  %indvars.iv.next1131.3 = add nuw nsw i64 %indvars.iv1130, 4
  %mul105.us.4 = mul i128 %add110.us.3, %conv104
  %add106.us.4 = add nuw i64 %indvars.iv.next1131.3, %indvars.iv1134
  %idxprom107.us.4 = and i64 %add106.us.4, 4294967295
  %arrayidx108.us.4 = getelementptr inbounds i8, i8* %call, i64 %idxprom107.us.4
  %151 = load i8, i8* %arrayidx108.us.4, align 1, !tbaa !16
  %conv109.us.4 = zext i8 %151 to i128
  %add110.us.4 = add i128 %mul105.us.4, %conv109.us.4
  %indvars.iv.next1131.4 = add nuw nsw i64 %indvars.iv1130, 5
  %mul105.us.5 = mul i128 %add110.us.4, %conv104
  %add106.us.5 = add nuw i64 %indvars.iv.next1131.4, %indvars.iv1134
  %idxprom107.us.5 = and i64 %add106.us.5, 4294967295
  %arrayidx108.us.5 = getelementptr inbounds i8, i8* %call, i64 %idxprom107.us.5
  %152 = load i8, i8* %arrayidx108.us.5, align 1, !tbaa !16
  %conv109.us.5 = zext i8 %152 to i128
  %add110.us.5 = add i128 %mul105.us.5, %conv109.us.5
  %indvars.iv.next1131.5 = add nuw nsw i64 %indvars.iv1130, 6
  %mul105.us.6 = mul i128 %add110.us.5, %conv104
  %add106.us.6 = add nuw i64 %indvars.iv.next1131.5, %indvars.iv1134
  %idxprom107.us.6 = and i64 %add106.us.6, 4294967295
  %arrayidx108.us.6 = getelementptr inbounds i8, i8* %call, i64 %idxprom107.us.6
  %153 = load i8, i8* %arrayidx108.us.6, align 1, !tbaa !16
  %conv109.us.6 = zext i8 %153 to i128
  %add110.us.6 = add i128 %mul105.us.6, %conv109.us.6
  %indvars.iv.next1131.6 = add nuw nsw i64 %indvars.iv1130, 7
  %mul105.us.7 = mul i128 %add110.us.6, %conv104
  %add106.us.7 = add nuw i64 %indvars.iv.next1131.6, %indvars.iv1134
  %idxprom107.us.7 = and i64 %add106.us.7, 4294967295
  %arrayidx108.us.7 = getelementptr inbounds i8, i8* %call, i64 %idxprom107.us.7
  %154 = load i8, i8* %arrayidx108.us.7, align 1, !tbaa !16
  %conv109.us.7 = zext i8 %154 to i128
  %add110.us.7 = add i128 %mul105.us.7, %conv109.us.7
  %indvars.iv.next1131.7 = add nuw nsw i64 %indvars.iv1130, 8
  %niter1208.nsub.7 = add i64 %niter1208, -8
  %niter1208.ncmp.7 = icmp eq i64 %niter1208.nsub.7, 0
  br i1 %niter1208.ncmp.7, label %for.cond99.for.cond.cleanup101_crit_edge.us.unr-lcssa, label %for.body103.us

for.body103.lr.ph.us:                             ; preds = %pfor.detach90.us
  %arrayidx97.us = getelementptr inbounds i8, i8* %call, i64 %indvars.iv1134
  %155 = load i8, i8* %arrayidx97.us, align 1, !tbaa !16
  %conv98.us = zext i8 %155 to i128
  br i1 %146, label %for.cond99.for.cond.cleanup101_crit_edge.us.unr-lcssa, label %for.body103.lr.ph.us.new

for.body103.lr.ph.us.new:                         ; preds = %for.body103.lr.ph.us
  br label %for.body103.us

for.cond99.for.cond.cleanup101_crit_edge.us.unr-lcssa: ; preds = %for.body103.us, %for.body103.lr.ph.us
  %add110.us.lcssa.ph = phi i128 [ undef, %for.body103.lr.ph.us ], [ %add110.us.7, %for.body103.us ]
  %indvars.iv1130.unr = phi i64 [ 1, %for.body103.lr.ph.us ], [ %indvars.iv.next1131.7, %for.body103.us ]
  %r.01082.us.unr = phi i128 [ %conv98.us, %for.body103.lr.ph.us ], [ %add110.us.7, %for.body103.us ]
  br i1 %lcmp.mod1205, label %for.cond99.for.cond.cleanup101_crit_edge.us, label %for.body103.us.epil.preheader

for.body103.us.epil.preheader:                    ; preds = %for.cond99.for.cond.cleanup101_crit_edge.us.unr-lcssa
  br label %for.body103.us.epil

for.body103.us.epil:                              ; preds = %for.body103.us.epil, %for.body103.us.epil.preheader
  %indvars.iv1130.epil = phi i64 [ %indvars.iv1130.unr, %for.body103.us.epil.preheader ], [ %indvars.iv.next1131.epil, %for.body103.us.epil ]
  %r.01082.us.epil = phi i128 [ %r.01082.us.unr, %for.body103.us.epil.preheader ], [ %add110.us.epil, %for.body103.us.epil ]
  %epil.iter1204 = phi i64 [ %xtraiter1203, %for.body103.us.epil.preheader ], [ %epil.iter1204.sub, %for.body103.us.epil ]
  %mul105.us.epil = mul i128 %r.01082.us.epil, %conv104
  %add106.us.epil = add nuw i64 %indvars.iv1130.epil, %indvars.iv1134
  %idxprom107.us.epil = and i64 %add106.us.epil, 4294967295
  %arrayidx108.us.epil = getelementptr inbounds i8, i8* %call, i64 %idxprom107.us.epil
  %156 = load i8, i8* %arrayidx108.us.epil, align 1, !tbaa !16
  %conv109.us.epil = zext i8 %156 to i128
  %add110.us.epil = add i128 %mul105.us.epil, %conv109.us.epil
  %indvars.iv.next1131.epil = add nuw nsw i64 %indvars.iv1130.epil, 1
  %epil.iter1204.sub = add i64 %epil.iter1204, -1
  %epil.iter1204.cmp = icmp eq i64 %epil.iter1204.sub, 0
  br i1 %epil.iter1204.cmp, label %for.cond99.for.cond.cleanup101_crit_edge.us, label %for.body103.us.epil, !llvm.loop !47

for.cond99.for.cond.cleanup101_crit_edge.us:      ; preds = %for.body103.us.epil, %for.cond99.for.cond.cleanup101_crit_edge.us.unr-lcssa
  %add110.us.lcssa = phi i128 [ %add110.us.lcssa.ph, %for.cond99.for.cond.cleanup101_crit_edge.us.unr-lcssa ], [ %add110.us.epil, %for.body103.us.epil ]
  %shl.us = shl i128 %add110.us.lcssa, 32
  %conv114.us = zext i64 %indvars.iv1134 to i128
  %add115.us = or i128 %shl.us, %conv114.us
  %arrayidx117.us = getelementptr inbounds i128, i128* %141, i64 %indvars.iv1134
  store i128 %add115.us, i128* %arrayidx117.us, align 16, !tbaa !2
  reattach within %syncreg77, label %pfor.inc119.us

for.body62:                                       ; preds = %for.body62.preheader, %for.body62
  %conv571091 = phi i64 [ %conv57, %for.body62 ], [ %conv571091.ph, %for.body62.preheader ]
  %i54.01090 = phi i32 [ %inc66, %for.body62 ], [ %i54.01090.ph, %for.body62.preheader ]
  %arrayidx64 = getelementptr inbounds i8, i8* %call, i64 %conv571091
  store i8 0, i8* %arrayidx64, align 1, !tbaa !16
  %inc66 = add i32 %i54.01090, 1
  %conv57 = zext i32 %inc66 to i64
  %cmp59 = icmp ugt i64 %add, %conv57
  br i1 %cmp59, label %for.body62, label %for.cond.cleanup60, !llvm.loop !48

pfor.cond.cleanup88:                              ; preds = %pfor.inc119, %pfor.inc119.us, %_ZNKSt5ctypeIcE5widenEc.exit977
  sync within %syncreg77, label %sync.continue121

pfor.detach90:                                    ; preds = %pfor.detach90.preheader, %pfor.inc119
  %indvars.iv1138 = phi i64 [ %indvars.iv.next1139, %pfor.inc119 ], [ 0, %pfor.detach90.preheader ]
  detach within %syncreg77, label %for.cond.cleanup101, label %pfor.inc119

for.cond.cleanup101:                              ; preds = %pfor.detach90
  %arrayidx97 = getelementptr inbounds i8, i8* %call, i64 %indvars.iv1138
  %157 = load i8, i8* %arrayidx97, align 1, !tbaa !16
  %conv98 = zext i8 %157 to i128
  %shl = shl nuw nsw i128 %conv98, 32
  %conv114 = zext i64 %indvars.iv1138 to i128
  %add115 = or i128 %shl, %conv114
  %arrayidx117 = getelementptr inbounds i128, i128* %141, i64 %indvars.iv1138
  store i128 %add115, i128* %arrayidx117, align 16, !tbaa !2
  reattach within %syncreg77, label %pfor.inc119

pfor.inc119:                                      ; preds = %for.cond.cleanup101, %pfor.detach90
  %indvars.iv.next1139 = add nuw nsw i64 %indvars.iv1138, 1
  %exitcond1141 = icmp eq i64 %indvars.iv.next1139, %wide.trip.count1136
  br i1 %exitcond1141, label %pfor.cond.cleanup88, label %pfor.detach90, !llvm.loop !46

sync.continue121:                                 ; preds = %pfor.cond.cleanup88
  tail call void @free(i8* %call) #2
  %158 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp124, i64 0, i32 2
  %159 = bitcast %"class.std::__cxx11::basic_string"* %agg.tmp124 to %union.anon**
  store %union.anon* %158, %union.anon** %159, align 8, !tbaa !10
  %160 = bitcast %union.anon* %158 to i8*
  %_M_p.i.phi.trans.insert.i.i.i.i = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp124, i64 0, i32 0, i32 0
  %161 = bitcast %union.anon* %158 to i32*
  store i32 2037411683, i32* %161, align 8
  %_M_string_length.i.i.i.i.i.i = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp124, i64 0, i32 1
  store i64 4, i64* %_M_string_length.i.i.i.i.i.i, align 8, !tbaa !13
  %arrayidx.i.i.i.i.i = getelementptr inbounds i8, i8* %160, i64 4
  store i8 0, i8* %arrayidx.i.i.i.i.i, align 4, !tbaa !16
  %call2.i.i651 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull %160, i64 4)
          to label %call2.i.i.noexc unwind label %lpad125

call2.i.i.noexc:                                  ; preds = %sync.continue121
  %call1.i.i652 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) %call2.i.i651, i8* nonnull getelementptr inbounds ([4 x i8], [4 x i8]* @.str.17, i64 0, i64 0), i64 3)
          to label %call1.i.i.noexc unwind label %lpad125

call1.i.i.noexc:                                  ; preds = %call2.i.i.noexc
  %162 = load i8, i8* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 3), align 8, !tbaa !17, !range !22
  %tobool.i.i.i = icmp eq i8 %162, 0
  br i1 %tobool.i.i.i, label %_ZN5timer10reportNextEv.exit.i, label %if.end.i.i.i

if.end.i.i.i:                                     ; preds = %call1.i.i.noexc
  %163 = bitcast %struct.timeval* %now.i.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %163) #2
  %call.i.i.i.i = call i32 @gettimeofday(%struct.timeval* nonnull %now.i.i.i.i, %struct.timezone* nonnull getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 5)) #2
  %tv_sec.i.i.i.i = getelementptr inbounds %struct.timeval, %struct.timeval* %now.i.i.i.i, i64 0, i32 0
  %164 = load i64, i64* %tv_sec.i.i.i.i, align 8, !tbaa !23
  %conv.i.i.i.i = sitofp i64 %164 to double
  %tv_usec.i.i.i.i = getelementptr inbounds %struct.timeval, %struct.timeval* %now.i.i.i.i, i64 0, i32 1
  %165 = load i64, i64* %tv_usec.i.i.i.i, align 8, !tbaa !25
  %conv2.i.i.i.i = sitofp i64 %165 to double
  %div.i.i.i.i = fmul fast double %conv2.i.i.i.i, 0x3EB0C6F7A0B5ED8D
  %add.i.i.i.i = fadd fast double %div.i.i.i.i, %conv.i.i.i.i
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %163) #2
  %166 = load double, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 1), align 8, !tbaa !26
  %sub.i.i.i = fsub fast double %add.i.i.i.i, %166
  %167 = load double, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 0), align 8, !tbaa !27
  %add.i.i.i650 = fadd fast double %sub.i.i.i, %167
  store double %add.i.i.i650, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 0), align 8, !tbaa !27
  store double %add.i.i.i.i, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 1), align 8, !tbaa !26
  br label %_ZN5timer10reportNextEv.exit.i

_ZN5timer10reportNextEv.exit.i:                   ; preds = %if.end.i.i.i, %call1.i.i.noexc
  %retval.0.i.i.i = phi double [ %sub.i.i.i, %if.end.i.i.i ], [ 0.000000e+00, %call1.i.i.noexc ]
  invoke void @_ZN5timer7reportTEd(%struct.timer* nonnull @_ZL3_tm, double %retval.0.i.i.i)
          to label %invoke.cont126 unwind label %lpad125

invoke.cont126:                                   ; preds = %_ZN5timer10reportNextEv.exit.i
  %168 = load i8*, i8** %_M_p.i.phi.trans.insert.i.i.i.i, align 8, !tbaa !28
  %cmp.i.i.i655 = icmp eq i8* %168, %160
  br i1 %cmp.i.i.i655, label %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit657, label %if.then.i.i656

if.then.i.i656:                                   ; preds = %invoke.cont126
  call void @_ZdlPv(i8* %168) #2
  br label %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit657

_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit657: ; preds = %invoke.cont126, %if.then.i.i656
  call void @_Z10sampleSortIoSt4lessIoElEvPT_T1_T0_(i128* %141, i64 %n)
  %169 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp129, i64 0, i32 2
  %170 = bitcast %"class.std::__cxx11::basic_string"* %agg.tmp129 to %union.anon**
  store %union.anon* %169, %union.anon** %170, align 8, !tbaa !10
  %171 = bitcast %union.anon* %169 to i8*
  %_M_p.i.phi.trans.insert.i.i.i.i661 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp129, i64 0, i32 0, i32 0
  %172 = bitcast %union.anon* %169 to i32*
  store i32 1953656691, i32* %172, align 8
  %_M_string_length.i.i.i.i.i.i670 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp129, i64 0, i32 1
  store i64 4, i64* %_M_string_length.i.i.i.i.i.i670, align 8, !tbaa !13
  %arrayidx.i.i.i.i.i671 = getelementptr inbounds i8, i8* %171, i64 4
  store i8 0, i8* %arrayidx.i.i.i.i.i671, align 4, !tbaa !16
  %call2.i.i692 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull %171, i64 4)
          to label %call2.i.i.noexc691 unwind label %lpad133

call2.i.i.noexc691:                               ; preds = %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit657
  %call1.i.i694 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) %call2.i.i692, i8* nonnull getelementptr inbounds ([4 x i8], [4 x i8]* @.str.17, i64 0, i64 0), i64 3)
          to label %call1.i.i.noexc693 unwind label %lpad133

call1.i.i.noexc693:                               ; preds = %call2.i.i.noexc691
  %173 = load i8, i8* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 3), align 8, !tbaa !17, !range !22
  %tobool.i.i.i678 = icmp eq i8 %173, 0
  br i1 %tobool.i.i.i678, label %_ZN5timer10reportNextEv.exit.i690, label %if.end.i.i.i688

if.end.i.i.i688:                                  ; preds = %call1.i.i.noexc693
  %174 = bitcast %struct.timeval* %now.i.i.i.i675 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %174) #2
  %call.i.i.i.i679 = call i32 @gettimeofday(%struct.timeval* nonnull %now.i.i.i.i675, %struct.timezone* nonnull getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 5)) #2
  %tv_sec.i.i.i.i680 = getelementptr inbounds %struct.timeval, %struct.timeval* %now.i.i.i.i675, i64 0, i32 0
  %175 = load i64, i64* %tv_sec.i.i.i.i680, align 8, !tbaa !23
  %conv.i.i.i.i681 = sitofp i64 %175 to double
  %tv_usec.i.i.i.i682 = getelementptr inbounds %struct.timeval, %struct.timeval* %now.i.i.i.i675, i64 0, i32 1
  %176 = load i64, i64* %tv_usec.i.i.i.i682, align 8, !tbaa !25
  %conv2.i.i.i.i683 = sitofp i64 %176 to double
  %div.i.i.i.i684 = fmul fast double %conv2.i.i.i.i683, 0x3EB0C6F7A0B5ED8D
  %add.i.i.i.i685 = fadd fast double %div.i.i.i.i684, %conv.i.i.i.i681
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %174) #2
  %177 = load double, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 1), align 8, !tbaa !26
  %sub.i.i.i686 = fsub fast double %add.i.i.i.i685, %177
  %178 = load double, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 0), align 8, !tbaa !27
  %add.i.i.i687 = fadd fast double %sub.i.i.i686, %178
  store double %add.i.i.i687, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 0), align 8, !tbaa !27
  store double %add.i.i.i.i685, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 1), align 8, !tbaa !26
  br label %_ZN5timer10reportNextEv.exit.i690

_ZN5timer10reportNextEv.exit.i690:                ; preds = %if.end.i.i.i688, %call1.i.i.noexc693
  %retval.0.i.i.i689 = phi double [ %sub.i.i.i686, %if.end.i.i.i688 ], [ 0.000000e+00, %call1.i.i.noexc693 ]
  invoke void @_ZN5timer7reportTEd(%struct.timer* nonnull @_ZL3_tm, double %retval.0.i.i.i689)
          to label %invoke.cont134 unwind label %lpad133

invoke.cont134:                                   ; preds = %_ZN5timer10reportNextEv.exit.i690
  %179 = load i8*, i8** %_M_p.i.phi.trans.insert.i.i.i.i661, align 8, !tbaa !28
  %cmp.i.i.i698 = icmp eq i8* %179, %171
  br i1 %cmp.i.i.i698, label %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit700, label %if.then.i.i699

if.then.i.i699:                                   ; preds = %invoke.cont134
  call void @_ZdlPv(i8* %179) #2
  br label %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit700

_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit700: ; preds = %invoke.cont134, %if.then.i.i699
  %mul138 = shl i64 %n, 2
  %call139 = call noalias i8* @malloc(i64 %mul138) #2
  %180 = bitcast i8* %call139 to i32*
  %mul140 = shl i64 %n, 3
  %call141 = call noalias i8* @malloc(i64 %mul140) #2
  %181 = bitcast i8* %call141 to %struct.seg*
  %call143 = call %"struct.std::pair"* @_Z15splitSegmentTopP3segjPjPo(%struct.seg* %181, i32 %conv, i32* %180, i128* %141)
  %182 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp144, i64 0, i32 2
  %183 = bitcast %"class.std::__cxx11::basic_string"* %agg.tmp144 to %union.anon**
  store %union.anon* %182, %union.anon** %183, align 8, !tbaa !10
  %184 = bitcast %union.anon* %182 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull %184, i8* nonnull getelementptr inbounds ([6 x i8], [6 x i8]* @.str.11, i64 0, i64 0), i64 5, i32 1, i1 false) #2
  %_M_string_length.i.i.i.i.i.i713 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp144, i64 0, i32 1
  store i64 5, i64* %_M_string_length.i.i.i.i.i.i713, align 8, !tbaa !13
  %arrayidx.i.i.i.i.i714 = getelementptr inbounds i8, i8* %184, i64 5
  store i8 0, i8* %arrayidx.i.i.i.i.i714, align 1, !tbaa !16
  %_M_p.i.i.i.i719 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp144, i64 0, i32 0, i32 0
  %call2.i.i735 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull %184, i64 5)
          to label %call2.i.i.noexc734 unwind label %lpad148

call2.i.i.noexc734:                               ; preds = %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit700
  %call1.i.i737 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) %call2.i.i735, i8* nonnull getelementptr inbounds ([4 x i8], [4 x i8]* @.str.17, i64 0, i64 0), i64 3)
          to label %call1.i.i.noexc736 unwind label %lpad148

call1.i.i.noexc736:                               ; preds = %call2.i.i.noexc734
  %185 = load i8, i8* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 3), align 8, !tbaa !17, !range !22
  %tobool.i.i.i721 = icmp eq i8 %185, 0
  br i1 %tobool.i.i.i721, label %_ZN5timer10reportNextEv.exit.i733, label %if.end.i.i.i731

if.end.i.i.i731:                                  ; preds = %call1.i.i.noexc736
  %186 = bitcast %struct.timeval* %now.i.i.i.i718 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %186) #2
  %call.i.i.i.i722 = call i32 @gettimeofday(%struct.timeval* nonnull %now.i.i.i.i718, %struct.timezone* nonnull getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 5)) #2
  %tv_sec.i.i.i.i723 = getelementptr inbounds %struct.timeval, %struct.timeval* %now.i.i.i.i718, i64 0, i32 0
  %187 = load i64, i64* %tv_sec.i.i.i.i723, align 8, !tbaa !23
  %conv.i.i.i.i724 = sitofp i64 %187 to double
  %tv_usec.i.i.i.i725 = getelementptr inbounds %struct.timeval, %struct.timeval* %now.i.i.i.i718, i64 0, i32 1
  %188 = load i64, i64* %tv_usec.i.i.i.i725, align 8, !tbaa !25
  %conv2.i.i.i.i726 = sitofp i64 %188 to double
  %div.i.i.i.i727 = fmul fast double %conv2.i.i.i.i726, 0x3EB0C6F7A0B5ED8D
  %add.i.i.i.i728 = fadd fast double %div.i.i.i.i727, %conv.i.i.i.i724
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %186) #2
  %189 = load double, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 1), align 8, !tbaa !26
  %sub.i.i.i729 = fsub fast double %add.i.i.i.i728, %189
  %190 = load double, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 0), align 8, !tbaa !27
  %add.i.i.i730 = fadd fast double %sub.i.i.i729, %190
  store double %add.i.i.i730, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 0), align 8, !tbaa !27
  store double %add.i.i.i.i728, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 1), align 8, !tbaa !26
  br label %_ZN5timer10reportNextEv.exit.i733

_ZN5timer10reportNextEv.exit.i733:                ; preds = %if.end.i.i.i731, %call1.i.i.noexc736
  %retval.0.i.i.i732 = phi double [ %sub.i.i.i729, %if.end.i.i.i731 ], [ 0.000000e+00, %call1.i.i.noexc736 ]
  invoke void @_ZN5timer7reportTEd(%struct.timer* nonnull @_ZL3_tm, double %retval.0.i.i.i732)
          to label %invoke.cont149 unwind label %lpad148

invoke.cont149:                                   ; preds = %_ZN5timer10reportNextEv.exit.i733
  %191 = load i8*, i8** %_M_p.i.i.i.i719, align 8, !tbaa !28
  %cmp.i.i.i741 = icmp eq i8* %191, %184
  br i1 %cmp.i.i.i741, label %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit743, label %if.then.i.i742

if.then.i.i742:                                   ; preds = %invoke.cont149
  call void @_ZdlPv(i8* %191) #2
  br label %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit743

_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit743: ; preds = %invoke.cont149, %if.then.i.i742
  %call154 = call noalias i8* @malloc(i64 %mul138) #2
  %192 = bitcast i8* %call154 to i32*
  %call156 = call noalias i8* @malloc(i64 %mul140) #2
  %193 = bitcast i8* %call156 to %struct.seg*
  %194 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp161, i64 0, i32 2
  %195 = bitcast %"class.std::__cxx11::basic_string"* %agg.tmp161 to %union.anon**
  %196 = bitcast %union.anon* %194 to i8*
  %197 = bitcast i64* %__dnew.i.i.i.i744 to i8*
  %_M_p.i18.i.i.i.i749 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp161, i64 0, i32 0, i32 0
  %_M_allocated_capacity.i.i.i.i.i750 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp161, i64 0, i32 2, i32 0
  %_M_string_length.i.i.i.i.i.i756 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp161, i64 0, i32 1
  %198 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp210, i64 0, i32 2
  %199 = bitcast %"class.std::__cxx11::basic_string"* %agg.tmp210 to %union.anon**
  %200 = bitcast %union.anon* %198 to i8*
  %_M_string_length.i.i.i.i.i.i832 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp210, i64 0, i32 1
  %_M_p.i.i.i.i838 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp210, i64 0, i32 0, i32 0
  %201 = bitcast %struct.timeval* %now.i.i.i.i837 to i8*
  %tv_sec.i.i.i.i842 = getelementptr inbounds %struct.timeval, %struct.timeval* %now.i.i.i.i837, i64 0, i32 0
  %tv_usec.i.i.i.i844 = getelementptr inbounds %struct.timeval, %struct.timeval* %now.i.i.i.i837, i64 0, i32 1
  %div276 = sdiv i64 %n, 10
  %202 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp303, i64 0, i32 2
  %203 = bitcast %"class.std::__cxx11::basic_string"* %agg.tmp303 to %union.anon**
  %204 = bitcast %union.anon* %202 to i8*
  %205 = bitcast %union.anon* %202 to i32*
  %_M_string_length.i.i.i.i.i.i880 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp303, i64 0, i32 1
  %_M_p.i.i.i.i886 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp303, i64 0, i32 0, i32 0
  %206 = bitcast %struct.timeval* %now.i.i.i.i885 to i8*
  %tv_sec.i.i.i.i890 = getelementptr inbounds %struct.timeval, %struct.timeval* %now.i.i.i.i885, i64 0, i32 0
  %tv_usec.i.i.i.i892 = getelementptr inbounds %struct.timeval, %struct.timeval* %now.i.i.i.i885, i64 0, i32 1
  %207 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp365, i64 0, i32 2
  %208 = bitcast %"class.std::__cxx11::basic_string"* %agg.tmp365 to %union.anon**
  %209 = bitcast %union.anon* %207 to i8*
  %_M_string_length.i.i.i.i.i.i928 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp365, i64 0, i32 1
  %_M_p.i.i.i.i934 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp365, i64 0, i32 0, i32 0
  %210 = bitcast %struct.timeval* %now.i.i.i.i933 to i8*
  %tv_sec.i.i.i.i938 = getelementptr inbounds %struct.timeval, %struct.timeval* %now.i.i.i.i933, i64 0, i32 0
  %tv_usec.i.i.i.i940 = getelementptr inbounds %struct.timeval, %struct.timeval* %now.i.i.i.i933, i64 0, i32 1
  %211 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %agg.tmp210, i64 0, i32 2, i32 1, i64 7
  %arrayidx.i.i.i.i.i881 = getelementptr inbounds i8, i8* %204, i64 4
  %arrayidx.i.i.i.i.i929 = getelementptr inbounds i8, i8* %209, i64 5
  br label %while.cond

while.cond:                                       ; preds = %cleanup.cont, %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit743
  %nKeys.0 = phi i32 [ %conv, %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit743 ], [ %call.i799, %cleanup.cont ]
  %round.0 = phi i32 [ 0, %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit743 ], [ %inc158, %cleanup.cont ]
  %offset.0 = phi i32 [ %conv70, %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit743 ], [ %mul374, %cleanup.cont ]
  %inc158 = add nuw nsw i32 %round.0, 1
  store %union.anon* %194, %union.anon** %195, align 8, !tbaa !10
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %197) #2
  store i64 30, i64* %__dnew.i.i.i.i744, align 8, !tbaa !32
  %call5.i.i.i9.i759 = invoke i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(%"class.std::__cxx11::basic_string"* nonnull %agg.tmp161, i64* nonnull dereferenceable(8) %__dnew.i.i.i.i744, i64 0)
          to label %call5.i.i.i9.i.noexc758 unwind label %lpad163

call5.i.i.i9.i.noexc758:                          ; preds = %while.cond
  store i8* %call5.i.i.i9.i759, i8** %_M_p.i18.i.i.i.i749, align 8, !tbaa !28
  %212 = load i64, i64* %__dnew.i.i.i.i744, align 8, !tbaa !32
  store i64 %212, i64* %_M_allocated_capacity.i.i.i.i.i750, align 8, !tbaa !16
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %call5.i.i.i9.i759, i8* nonnull getelementptr inbounds ([31 x i8], [31 x i8]* @.str.12, i64 0, i64 0), i64 30, i32 1, i1 false) #2
  store i64 %212, i64* %_M_string_length.i.i.i.i.i.i756, align 8, !tbaa !13
  %213 = load i8*, i8** %_M_p.i18.i.i.i.i749, align 8, !tbaa !28
  %arrayidx.i.i.i.i.i757 = getelementptr inbounds i8, i8* %213, i64 %212
  store i8 0, i8* %arrayidx.i.i.i.i.i757, align 1, !tbaa !16
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %197) #2
  %cmp159 = icmp ugt i32 %round.0, 39
  %214 = load i8*, i8** %_M_p.i18.i.i.i.i749, align 8, !tbaa !28
  br i1 %cmp159, label %if.then.i, label %invoke.cont166

if.then.i:                                        ; preds = %call5.i.i.i9.i.noexc758
  %215 = load i64, i64* %_M_string_length.i.i.i.i.i.i756, align 8, !tbaa !13
  %call2.i793 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* %214, i64 %215)
          to label %call.i761.noexc unwind label %lpad165

call.i761.noexc:                                  ; preds = %if.then.i
  %call.i.i762764 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(%"class.std::basic_ostream"* nonnull dereferenceable(272) %call2.i793)
          to label %call.i.i762.noexc unwind label %lpad165

call.i.i762.noexc:                                ; preds = %call.i761.noexc
  call void @abort() #13
  unreachable

invoke.cont166:                                   ; preds = %call5.i.i.i9.i.noexc758
  %cmp.i.i.i767 = icmp eq i8* %214, %196
  br i1 %cmp.i.i.i767, label %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit769, label %if.then.i.i768

if.then.i.i768:                                   ; preds = %invoke.cont166
  call void @_ZdlPv(i8* %214) #2
  br label %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit769

_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit769: ; preds = %invoke.cont166, %if.then.i.i768
  %cmp.i = icmp ult i32 %nKeys.0, 2048
  br i1 %cmp.i, label %if.then.i771, label %pfor.detach.lr.ph.i

if.then.i771:                                     ; preds = %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit769
  %cmp13.i.i = icmp eq i32 %nKeys.0, 0
  br i1 %cmp13.i.i, label %while.end, label %for.body.lr.ph.i.i

for.body.lr.ph.i.i:                               ; preds = %if.then.i771
  %wide.trip.count.i.i = zext i32 %nKeys.0 to i64
  %216 = add nsw i64 %wide.trip.count.i.i, -1
  %xtraiter = and i64 %wide.trip.count.i.i, 3
  %217 = icmp ult i64 %216, 3
  br i1 %217, label %_ZN8sequence6filterI3segj5isSegEET0_PT_S5_S3_T1_.exit.loopexit.unr-lcssa, label %for.body.lr.ph.i.i.new

for.body.lr.ph.i.i.new:                           ; preds = %for.body.lr.ph.i.i
  %unroll_iter = sub nsw i64 %wide.trip.count.i.i, %xtraiter
  br label %for.body.i.i

for.body.i.i:                                     ; preds = %for.inc.i.i.3, %for.body.lr.ph.i.i.new
  %indvars.iv.i.i = phi i64 [ 0, %for.body.lr.ph.i.i.new ], [ %indvars.iv.next.i.i.3, %for.inc.i.i.3 ]
  %k.015.i.i = phi i32 [ 0, %for.body.lr.ph.i.i.new ], [ %k.1.i.i.3, %for.inc.i.i.3 ]
  %niter = phi i64 [ %unroll_iter, %for.body.lr.ph.i.i.new ], [ %niter.nsub.3, %for.inc.i.i.3 ]
  %arrayidx.i.i = getelementptr inbounds %struct.seg, %struct.seg* %181, i64 %indvars.iv.i.i
  %agg.tmp.sroa.0.0..sroa_cast.i.i = bitcast %struct.seg* %arrayidx.i.i to i64*
  %agg.tmp.sroa.0.0.copyload.i.i = load i64, i64* %agg.tmp.sroa.0.0..sroa_cast.i.i, align 4
  %s.sroa.1.0.extract.shift.i.i.i = lshr i64 %agg.tmp.sroa.0.0.copyload.i.i, 32
  %s.sroa.1.0.extract.trunc.i.i.i = trunc i64 %s.sroa.1.0.extract.shift.i.i.i to i32
  %cmp.i.i.i772 = icmp ugt i32 %s.sroa.1.0.extract.trunc.i.i.i, 1
  br i1 %cmp.i.i.i772, label %if.then.i.i773, label %for.inc.i.i

if.then.i.i773:                                   ; preds = %for.body.i.i
  %inc.i.i = add i32 %k.015.i.i, 1
  %idxprom3.i.i = zext i32 %k.015.i.i to i64
  %arrayidx4.i.i = getelementptr inbounds %struct.seg, %struct.seg* %193, i64 %idxprom3.i.i
  %218 = bitcast %struct.seg* %arrayidx4.i.i to i64*
  store i64 %agg.tmp.sroa.0.0.copyload.i.i, i64* %218, align 4
  br label %for.inc.i.i

for.inc.i.i:                                      ; preds = %if.then.i.i773, %for.body.i.i
  %k.1.i.i = phi i32 [ %inc.i.i, %if.then.i.i773 ], [ %k.015.i.i, %for.body.i.i ]
  %indvars.iv.next.i.i = or i64 %indvars.iv.i.i, 1
  %arrayidx.i.i.1 = getelementptr inbounds %struct.seg, %struct.seg* %181, i64 %indvars.iv.next.i.i
  %agg.tmp.sroa.0.0..sroa_cast.i.i.1 = bitcast %struct.seg* %arrayidx.i.i.1 to i64*
  %agg.tmp.sroa.0.0.copyload.i.i.1 = load i64, i64* %agg.tmp.sroa.0.0..sroa_cast.i.i.1, align 4
  %s.sroa.1.0.extract.shift.i.i.i.1 = lshr i64 %agg.tmp.sroa.0.0.copyload.i.i.1, 32
  %s.sroa.1.0.extract.trunc.i.i.i.1 = trunc i64 %s.sroa.1.0.extract.shift.i.i.i.1 to i32
  %cmp.i.i.i772.1 = icmp ugt i32 %s.sroa.1.0.extract.trunc.i.i.i.1, 1
  br i1 %cmp.i.i.i772.1, label %if.then.i.i773.1, label %for.inc.i.i.1

pfor.detach.lr.ph.i:                              ; preds = %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit769
  %conv.i774 = zext i32 %nKeys.0 to i64
  %call1.i775 = call noalias i8* @malloc(i64 %conv.i774) #2
  br label %pfor.detach.i

pfor.cond.cleanup.i:                              ; preds = %pfor.inc.i
  sync within %syncreg.i770, label %sync.continue.i

pfor.detach.i:                                    ; preds = %pfor.inc.i, %pfor.detach.lr.ph.i
  %indvars.iv.i = phi i64 [ 0, %pfor.detach.lr.ph.i ], [ %indvars.iv.next.i, %pfor.inc.i ]
  detach within %syncreg.i770, label %pfor.body.i, label %pfor.inc.i unwind label %lpad10.i

pfor.body.i:                                      ; preds = %pfor.detach.i
  %arrayidx.i = getelementptr inbounds %struct.seg, %struct.seg* %181, i64 %indvars.iv.i
  %agg.tmp6.sroa.0.0..sroa_cast.i = bitcast %struct.seg* %arrayidx.i to i64*
  %agg.tmp6.sroa.0.0.copyload.i = load i64, i64* %agg.tmp6.sroa.0.0..sroa_cast.i, align 4
  %s.sroa.1.0.extract.shift.i.i = lshr i64 %agg.tmp6.sroa.0.0.copyload.i, 32
  %s.sroa.1.0.extract.trunc.i.i = trunc i64 %s.sroa.1.0.extract.shift.i.i to i32
  %cmp.i.i = icmp ugt i32 %s.sroa.1.0.extract.trunc.i.i, 1
  %arrayidx9.i = getelementptr inbounds i8, i8* %call1.i775, i64 %indvars.iv.i
  %frombool.i = zext i1 %cmp.i.i to i8
  store i8 %frombool.i, i8* %arrayidx9.i, align 1, !tbaa !49
  reattach within %syncreg.i770, label %pfor.inc.i

pfor.inc.i:                                       ; preds = %pfor.body.i, %pfor.detach.i
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %exitcond.i = icmp eq i64 %indvars.iv.next.i, %conv.i774
  br i1 %exitcond.i, label %pfor.cond.cleanup.i, label %pfor.detach.i, !llvm.loop !50

lpad10.i:                                         ; preds = %pfor.detach.i
  %219 = landingpad { i8*, i32 }
          cleanup
  sync within %syncreg.i770, label %sync.continue14.i

sync.continue.i:                                  ; preds = %pfor.cond.cleanup.i
  %call.i.i776 = call { %struct.seg*, i64 } @_ZN8sequence4packI3segjNS_4getAIS1_jEEEE4_seqIT_EPS5_PbT0_S9_T1_(%struct.seg* %193, i8* %call1.i775, i32 0, i32 %nKeys.0, %struct.seg* %181)
  %220 = extractvalue { %struct.seg*, i64 } %call.i.i776, 1
  %conv.i.i777 = trunc i64 %220 to i32
  call void @free(i8* %call1.i775) #2
  br label %_ZN8sequence6filterI3segj5isSegEET0_PT_S5_S3_T1_.exit

sync.continue14.i:                                ; preds = %lpad10.i
  resume { i8*, i32 } %219

_ZN8sequence6filterI3segj5isSegEET0_PT_S5_S3_T1_.exit.loopexit.unr-lcssa: ; preds = %for.inc.i.i.3, %for.body.lr.ph.i.i
  %k.1.i.i.lcssa.ph = phi i32 [ undef, %for.body.lr.ph.i.i ], [ %k.1.i.i.3, %for.inc.i.i.3 ]
  %indvars.iv.i.i.unr = phi i64 [ 0, %for.body.lr.ph.i.i ], [ %indvars.iv.next.i.i.3, %for.inc.i.i.3 ]
  %k.015.i.i.unr = phi i32 [ 0, %for.body.lr.ph.i.i ], [ %k.1.i.i.3, %for.inc.i.i.3 ]
  %lcmp.mod = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod, label %_ZN8sequence6filterI3segj5isSegEET0_PT_S5_S3_T1_.exit, label %for.body.i.i.epil.preheader

for.body.i.i.epil.preheader:                      ; preds = %_ZN8sequence6filterI3segj5isSegEET0_PT_S5_S3_T1_.exit.loopexit.unr-lcssa
  br label %for.body.i.i.epil

for.body.i.i.epil:                                ; preds = %for.inc.i.i.epil, %for.body.i.i.epil.preheader
  %indvars.iv.i.i.epil = phi i64 [ %indvars.iv.i.i.unr, %for.body.i.i.epil.preheader ], [ %indvars.iv.next.i.i.epil, %for.inc.i.i.epil ]
  %k.015.i.i.epil = phi i32 [ %k.015.i.i.unr, %for.body.i.i.epil.preheader ], [ %k.1.i.i.epil, %for.inc.i.i.epil ]
  %epil.iter = phi i64 [ %xtraiter, %for.body.i.i.epil.preheader ], [ %epil.iter.sub, %for.inc.i.i.epil ]
  %arrayidx.i.i.epil = getelementptr inbounds %struct.seg, %struct.seg* %181, i64 %indvars.iv.i.i.epil
  %agg.tmp.sroa.0.0..sroa_cast.i.i.epil = bitcast %struct.seg* %arrayidx.i.i.epil to i64*
  %agg.tmp.sroa.0.0.copyload.i.i.epil = load i64, i64* %agg.tmp.sroa.0.0..sroa_cast.i.i.epil, align 4
  %s.sroa.1.0.extract.shift.i.i.i.epil = lshr i64 %agg.tmp.sroa.0.0.copyload.i.i.epil, 32
  %s.sroa.1.0.extract.trunc.i.i.i.epil = trunc i64 %s.sroa.1.0.extract.shift.i.i.i.epil to i32
  %cmp.i.i.i772.epil = icmp ugt i32 %s.sroa.1.0.extract.trunc.i.i.i.epil, 1
  br i1 %cmp.i.i.i772.epil, label %if.then.i.i773.epil, label %for.inc.i.i.epil

if.then.i.i773.epil:                              ; preds = %for.body.i.i.epil
  %inc.i.i.epil = add i32 %k.015.i.i.epil, 1
  %idxprom3.i.i.epil = zext i32 %k.015.i.i.epil to i64
  %arrayidx4.i.i.epil = getelementptr inbounds %struct.seg, %struct.seg* %193, i64 %idxprom3.i.i.epil
  %221 = bitcast %struct.seg* %arrayidx4.i.i.epil to i64*
  store i64 %agg.tmp.sroa.0.0.copyload.i.i.epil, i64* %221, align 4
  br label %for.inc.i.i.epil

for.inc.i.i.epil:                                 ; preds = %if.then.i.i773.epil, %for.body.i.i.epil
  %k.1.i.i.epil = phi i32 [ %inc.i.i.epil, %if.then.i.i773.epil ], [ %k.015.i.i.epil, %for.body.i.i.epil ]
  %indvars.iv.next.i.i.epil = add nuw nsw i64 %indvars.iv.i.i.epil, 1
  %epil.iter.sub = add i64 %epil.iter, -1
  %epil.iter.cmp = icmp eq i64 %epil.iter.sub, 0
  br i1 %epil.iter.cmp, label %_ZN8sequence6filterI3segj5isSegEET0_PT_S5_S3_T1_.exit, label %for.body.i.i.epil, !llvm.loop !51

_ZN8sequence6filterI3segj5isSegEET0_PT_S5_S3_T1_.exit: ; preds = %_ZN8sequence6filterI3segj5isSegEET0_PT_S5_S3_T1_.exit.loopexit.unr-lcssa, %for.inc.i.i.epil, %sync.continue.i
  %retval.0.i = phi i32 [ %conv.i.i777, %sync.continue.i ], [ %k.1.i.i.lcssa.ph, %_ZN8sequence6filterI3segj5isSegEET0_PT_S5_S3_T1_.exit.loopexit.unr-lcssa ], [ %k.1.i.i.epil, %for.inc.i.i.epil ]
  %cmp172 = icmp eq i32 %retval.0.i, 0
  br i1 %cmp172, label %while.end, label %pfor.detach186.lr.ph

lpad125:                                          ; preds = %_ZN5timer10reportNextEv.exit.i, %call2.i.i.noexc, %sync.continue121
  %222 = landingpad { i8*, i32 }
          cleanup
  %223 = extractvalue { i8*, i32 } %222, 0
  %224 = extractvalue { i8*, i32 } %222, 1
  %225 = load i8*, i8** %_M_p.i.phi.trans.insert.i.i.i.i, align 8, !tbaa !28
  %cmp.i.i.i780 = icmp eq i8* %225, %160
  br i1 %cmp.i.i.i780, label %ehcleanup425, label %if.then.i.i781

if.then.i.i781:                                   ; preds = %lpad125
  call void @_ZdlPv(i8* %225) #2
  br label %ehcleanup425

lpad133:                                          ; preds = %_ZN5timer10reportNextEv.exit.i690, %call2.i.i.noexc691, %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit657
  %226 = landingpad { i8*, i32 }
          cleanup
  %227 = extractvalue { i8*, i32 } %226, 0
  %228 = extractvalue { i8*, i32 } %226, 1
  %229 = load i8*, i8** %_M_p.i.phi.trans.insert.i.i.i.i661, align 8, !tbaa !28
  %cmp.i.i.i785 = icmp eq i8* %229, %171
  br i1 %cmp.i.i.i785, label %ehcleanup425, label %if.then.i.i786

if.then.i.i786:                                   ; preds = %lpad133
  call void @_ZdlPv(i8* %229) #2
  br label %ehcleanup425

lpad148:                                          ; preds = %_ZN5timer10reportNextEv.exit.i733, %call2.i.i.noexc734, %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit700
  %230 = landingpad { i8*, i32 }
          cleanup
  %231 = extractvalue { i8*, i32 } %230, 0
  %232 = extractvalue { i8*, i32 } %230, 1
  %233 = load i8*, i8** %_M_p.i.i.i.i719, align 8, !tbaa !28
  %cmp.i.i.i790 = icmp eq i8* %233, %184
  br i1 %cmp.i.i.i790, label %ehcleanup425, label %if.then.i.i791

if.then.i.i791:                                   ; preds = %lpad148
  call void @_ZdlPv(i8* %233) #2
  br label %ehcleanup425

lpad163:                                          ; preds = %while.cond
  %234 = landingpad { i8*, i32 }
          cleanup
  %235 = extractvalue { i8*, i32 } %234, 0
  %236 = extractvalue { i8*, i32 } %234, 1
  br label %ehcleanup425

lpad165:                                          ; preds = %if.then.i, %call.i761.noexc
  %237 = landingpad { i8*, i32 }
          cleanup
  %238 = extractvalue { i8*, i32 } %237, 0
  %239 = extractvalue { i8*, i32 } %237, 1
  %240 = load i8*, i8** %_M_p.i18.i.i.i.i749, align 8, !tbaa !28
  %cmp.i.i.i796 = icmp eq i8* %240, %196
  br i1 %cmp.i.i.i796, label %ehcleanup425, label %if.then.i.i797

if.then.i.i797:                                   ; preds = %lpad165
  call void @_ZdlPv(i8* %240) #2
  br label %ehcleanup425

pfor.detach186.lr.ph:                             ; preds = %_ZN8sequence6filterI3segj5isSegEET0_PT_S5_S3_T1_.exit
  %wide.trip.count1116 = zext i32 %retval.0.i to i64
  br label %pfor.detach186

pfor.cond.cleanup185:                             ; preds = %pfor.inc197
  sync within %syncreg175, label %sync.continue199

pfor.detach186:                                   ; preds = %pfor.inc197, %pfor.detach186.lr.ph
  %indvars.iv1114 = phi i64 [ 0, %pfor.detach186.lr.ph ], [ %indvars.iv.next1115, %pfor.inc197 ]
  detach within %syncreg175, label %pfor.body191, label %pfor.inc197

pfor.body191:                                     ; preds = %pfor.detach186
  %length = getelementptr inbounds %struct.seg, %struct.seg* %193, i64 %indvars.iv1114, i32 1
  %241 = load i32, i32* %length, align 4, !tbaa !52
  %arrayidx195 = getelementptr inbounds i32, i32* %192, i64 %indvars.iv1114
  store i32 %241, i32* %arrayidx195, align 4, !tbaa !6
  reattach within %syncreg175, label %pfor.inc197

pfor.inc197:                                      ; preds = %pfor.body191, %pfor.detach186
  %indvars.iv.next1115 = add nuw nsw i64 %indvars.iv1114, 1
  %exitcond1117 = icmp eq i64 %indvars.iv.next1115, %wide.trip.count1116
  br i1 %exitcond1117, label %pfor.cond.cleanup185, label %pfor.detach186, !llvm.loop !54

sync.continue199:                                 ; preds = %pfor.cond.cleanup185
  %call.i799 = call i32 @_ZN8sequence4scanIjjN5utils4addFIjEENS_4getAIjjEEEET_PS6_T0_S8_T1_T2_S6_bb(i32* %192, i32 0, i32 %retval.0.i, i32* %192, i32 0, i1 zeroext false, i1 zeroext false)
  %call1.i801 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull getelementptr inbounds ([9 x i8], [9 x i8]* @.str.13, i64 0, i64 0), i64 8)
  %conv.i802 = zext i32 %retval.0.i to i64
  %call.i803 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertImEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, i64 %conv.i802)
  %call1.i805 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) %call.i803, i8* nonnull getelementptr inbounds ([10 x i8], [10 x i8]* @.str.14, i64 0, i64 0), i64 9)
  %conv.i806 = zext i32 %call.i799 to i64
  %call.i807 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertImEERSoT_(%"class.std::basic_ostream"* nonnull %call.i803, i64 %conv.i806)
  %call1.i809 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) %call.i807, i8* nonnull getelementptr inbounds ([18 x i8], [18 x i8]* @.str.15, i64 0, i64 0), i64 17)
  %conv.i810 = zext i32 %offset.0 to i64
  %call.i811 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertImEERSoT_(%"class.std::basic_ostream"* nonnull %call.i807, i64 %conv.i810)
  %242 = bitcast %"class.std::basic_ostream"* %call.i811 to i8**
  %vtable.i813 = load i8*, i8** %242, align 8, !tbaa !35
  %vbase.offset.ptr.i814 = getelementptr i8, i8* %vtable.i813, i64 -24
  %243 = bitcast i8* %vbase.offset.ptr.i814 to i64*
  %vbase.offset.i815 = load i64, i64* %243, align 8
  %244 = bitcast %"class.std::basic_ostream"* %call.i811 to i8*
  %add.ptr.i816 = getelementptr inbounds i8, i8* %244, i64 %vbase.offset.i815
  %_M_ctype.i978 = getelementptr inbounds i8, i8* %add.ptr.i816, i64 240
  %245 = bitcast i8* %_M_ctype.i978 to %"class.std::ctype"**
  %246 = load %"class.std::ctype"*, %"class.std::ctype"** %245, align 8, !tbaa !37
  %tobool.i997 = icmp eq %"class.std::ctype"* %246, null
  br i1 %tobool.i997, label %if.then.i998, label %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit1000

if.then.i998:                                     ; preds = %sync.continue199
  call void @_ZSt16__throw_bad_castv() #12
  unreachable

_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit1000: ; preds = %sync.continue199
  %_M_widen_ok.i980 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %246, i64 0, i32 8
  %247 = load i8, i8* %_M_widen_ok.i980, align 8, !tbaa !39
  %tobool.i981 = icmp eq i8 %247, 0
  br i1 %tobool.i981, label %if.end.i987, label %if.then.i983

if.then.i983:                                     ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit1000
  %arrayidx.i982 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %246, i64 0, i32 9, i64 10
  %248 = load i8, i8* %arrayidx.i982, align 1, !tbaa !16
  br label %_ZNKSt5ctypeIcE5widenEc.exit989

if.end.i987:                                      ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit1000
  call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %246)
  %249 = bitcast %"class.std::ctype"* %246 to i8 (%"class.std::ctype"*, i8)***
  %vtable.i984 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %249, align 8, !tbaa !35
  %vfn.i985 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vtable.i984, i64 6
  %250 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vfn.i985, align 8
  %call.i986 = call signext i8 %250(%"class.std::ctype"* nonnull %246, i8 signext 10)
  br label %_ZNKSt5ctypeIcE5widenEc.exit989

_ZNKSt5ctypeIcE5widenEc.exit989:                  ; preds = %if.then.i983, %if.end.i987
  %retval.0.i988 = phi i8 [ %248, %if.then.i983 ], [ %call.i986, %if.end.i987 ]
  %call1.i818 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %call.i811, i8 signext %retval.0.i988)
  %call.i819 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %call1.i818)
  store %union.anon* %198, %union.anon** %199, align 8, !tbaa !10
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull %200, i8* nonnull getelementptr inbounds ([16 x i8], [16 x i8]* @.str.16, i64 0, i64 0), i64 15, i32 1, i1 false) #2
  store i64 15, i64* %_M_string_length.i.i.i.i.i.i832, align 8, !tbaa !13
  store i8 0, i8* %211, align 1, !tbaa !16
  %call2.i.i854 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull %200, i64 15)
          to label %call2.i.i.noexc853 unwind label %lpad214

call2.i.i.noexc853:                               ; preds = %_ZNKSt5ctypeIcE5widenEc.exit989
  %call1.i.i856 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) %call2.i.i854, i8* nonnull getelementptr inbounds ([4 x i8], [4 x i8]* @.str.17, i64 0, i64 0), i64 3)
          to label %call1.i.i.noexc855 unwind label %lpad214

call1.i.i.noexc855:                               ; preds = %call2.i.i.noexc853
  %251 = load i8, i8* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 3), align 8, !tbaa !17, !range !22
  %tobool.i.i.i840 = icmp eq i8 %251, 0
  br i1 %tobool.i.i.i840, label %_ZN5timer10reportNextEv.exit.i852, label %if.end.i.i.i850

if.end.i.i.i850:                                  ; preds = %call1.i.i.noexc855
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %201) #2
  %call.i.i.i.i841 = call i32 @gettimeofday(%struct.timeval* nonnull %now.i.i.i.i837, %struct.timezone* nonnull getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 5)) #2
  %252 = load i64, i64* %tv_sec.i.i.i.i842, align 8, !tbaa !23
  %conv.i.i.i.i843 = sitofp i64 %252 to double
  %253 = load i64, i64* %tv_usec.i.i.i.i844, align 8, !tbaa !25
  %conv2.i.i.i.i845 = sitofp i64 %253 to double
  %div.i.i.i.i846 = fmul fast double %conv2.i.i.i.i845, 0x3EB0C6F7A0B5ED8D
  %add.i.i.i.i847 = fadd fast double %div.i.i.i.i846, %conv.i.i.i.i843
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %201) #2
  %254 = load double, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 1), align 8, !tbaa !26
  %sub.i.i.i848 = fsub fast double %add.i.i.i.i847, %254
  %255 = load double, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 0), align 8, !tbaa !27
  %add.i.i.i849 = fadd fast double %sub.i.i.i848, %255
  store double %add.i.i.i849, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 0), align 8, !tbaa !27
  store double %add.i.i.i.i847, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 1), align 8, !tbaa !26
  br label %_ZN5timer10reportNextEv.exit.i852

_ZN5timer10reportNextEv.exit.i852:                ; preds = %if.end.i.i.i850, %call1.i.i.noexc855
  %retval.0.i.i.i851 = phi double [ %sub.i.i.i848, %if.end.i.i.i850 ], [ 0.000000e+00, %call1.i.i.noexc855 ]
  invoke void @_ZN5timer7reportTEd(%struct.timer* nonnull @_ZL3_tm, double %retval.0.i.i.i851)
          to label %invoke.cont215 unwind label %lpad214

invoke.cont215:                                   ; preds = %_ZN5timer10reportNextEv.exit.i852
  %256 = load i8*, i8** %_M_p.i.i.i.i838, align 8, !tbaa !28
  %cmp.i.i.i860 = icmp eq i8* %256, %200
  br i1 %cmp.i.i.i860, label %pfor.detach230.preheader, label %if.then.i.i861

pfor.detach230.preheader:                         ; preds = %if.then.i.i861, %invoke.cont215
  br label %pfor.detach230

if.then.i.i861:                                   ; preds = %invoke.cont215
  call void @_ZdlPv(i8* %256) #2
  br label %pfor.detach230.preheader

pfor.cond.cleanup229:                             ; preds = %pfor.inc292
  sync within %syncreg219, label %sync.continue296

lpad214:                                          ; preds = %_ZN5timer10reportNextEv.exit.i852, %call2.i.i.noexc853, %_ZNKSt5ctypeIcE5widenEc.exit989
  %257 = landingpad { i8*, i32 }
          cleanup
  %258 = extractvalue { i8*, i32 } %257, 0
  %259 = extractvalue { i8*, i32 } %257, 1
  %260 = load i8*, i8** %_M_p.i.i.i.i838, align 8, !tbaa !28
  %cmp.i.i.i865 = icmp eq i8* %260, %200
  br i1 %cmp.i.i.i865, label %ehcleanup425, label %if.then.i.i866

if.then.i.i866:                                   ; preds = %lpad214
  call void @_ZdlPv(i8* %260) #2
  br label %ehcleanup425

pfor.detach230:                                   ; preds = %pfor.detach230.preheader, %pfor.inc292
  %indvars.iv1122 = phi i64 [ %indvars.iv.next1123, %pfor.inc292 ], [ 0, %pfor.detach230.preheader ]
  detach within %syncreg219, label %pfor.body235, label %pfor.inc292 unwind label %lpad294.loopexit

pfor.body235:                                     ; preds = %pfor.detach230
  %syncreg242 = call token @llvm.syncregion.start()
  %start238 = getelementptr inbounds %struct.seg, %struct.seg* %193, i64 %indvars.iv1122, i32 0
  %261 = load i32, i32* %start238, align 4, !tbaa !55
  %idx.ext = zext i32 %261 to i64
  %add.ptr = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %call143, i64 %idx.ext
  %length241 = getelementptr inbounds %struct.seg, %struct.seg* %193, i64 %indvars.iv1122, i32 1
  %262 = load i32, i32* %length241, align 4, !tbaa !52
  %cmp2511072 = icmp eq i32 %262, 0
  br i1 %cmp2511072, label %pfor.cond.cleanup252, label %pfor.detach254.lr.ph

pfor.detach254.lr.ph:                             ; preds = %pfor.body235
  %wide.trip.count1120 = zext i32 %262 to i64
  br label %pfor.detach254

pfor.cond.cleanup252:                             ; preds = %pfor.inc270, %pfor.body235
  sync within %syncreg242, label %sync.continue272

pfor.detach254:                                   ; preds = %pfor.inc270, %pfor.detach254.lr.ph
  %indvars.iv1118 = phi i64 [ 0, %pfor.detach254.lr.ph ], [ %indvars.iv.next1119, %pfor.inc270 ]
  detach within %syncreg242, label %pfor.body259, label %pfor.inc270

pfor.body259:                                     ; preds = %pfor.detach254
  %arrayidx261 = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %add.ptr, i64 %indvars.iv1118
  %second = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %arrayidx261, i64 0, i32 1
  %263 = load i32, i32* %second, align 4, !tbaa !29
  %add262 = add i32 %263, %offset.0
  %conv263 = zext i32 %add262 to i64
  %cmp264 = icmp slt i64 %conv263, %n
  br i1 %cmp264, label %cond.false, label %cond.end

cond.false:                                       ; preds = %pfor.body259
  %arrayidx266 = getelementptr inbounds i32, i32* %180, i64 %conv263
  %264 = load i32, i32* %arrayidx266, align 4, !tbaa !6
  br label %cond.end

cond.end:                                         ; preds = %pfor.body259, %cond.false
  %cond = phi i32 [ %264, %cond.false ], [ 0, %pfor.body259 ]
  %first = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %arrayidx261, i64 0, i32 0
  store i32 %cond, i32* %first, align 4, !tbaa !56
  reattach within %syncreg242, label %pfor.inc270

pfor.inc270:                                      ; preds = %cond.end, %pfor.detach254
  %indvars.iv.next1119 = add nuw nsw i64 %indvars.iv1118, 1
  %exitcond1121 = icmp eq i64 %indvars.iv.next1119, %wide.trip.count1120
  br i1 %exitcond1121, label %pfor.cond.cleanup252, label %pfor.detach254, !llvm.loop !57

sync.continue272:                                 ; preds = %pfor.cond.cleanup252
  %conv275 = zext i32 %262 to i64
  %cmp277 = icmp sgt i64 %div276, %conv275
  br i1 %cmp277, label %if.else, label %if.then278

if.then278:                                       ; preds = %sync.continue272
  invoke void @_Z10sampleSortISt4pairIjjE9pairCompFjEvPT_T1_T0_(%"struct.std::pair"* %add.ptr, i32 %262)
          to label %if.end286 unwind label %lpad280

lpad280:                                          ; preds = %if.else, %if.then278
  %265 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg219, { i8*, i32 } %265)
          to label %det.rethrow.unreachable unwind label %lpad294.loopexit.split-lp

det.rethrow.unreachable:                          ; preds = %lpad280
  unreachable

if.else:                                          ; preds = %sync.continue272
  invoke void @_Z9quickSortISt4pairIjjE9pairCompFjEvPT_T1_T0_(%"struct.std::pair"* %add.ptr, i32 %262)
          to label %if.end286 unwind label %lpad280

if.end286:                                        ; preds = %if.else, %if.then278
  reattach within %syncreg219, label %pfor.inc292

pfor.inc292:                                      ; preds = %pfor.detach230, %if.end286
  %indvars.iv.next1123 = add nuw nsw i64 %indvars.iv1122, 1
  %exitcond1125 = icmp eq i64 %indvars.iv.next1123, %conv.i802
  br i1 %exitcond1125, label %pfor.cond.cleanup229, label %pfor.detach230, !llvm.loop !59

lpad294.loopexit:                                 ; preds = %pfor.detach230
  %lpad.loopexit1002 = landingpad { i8*, i32 }
          cleanup
  br label %lpad294

lpad294.loopexit.split-lp:                        ; preds = %lpad280
  %lpad.loopexit.split-lp1003 = landingpad { i8*, i32 }
          cleanup
  br label %lpad294

lpad294:                                          ; preds = %lpad294.loopexit.split-lp, %lpad294.loopexit
  %lpad.phi1004 = phi { i8*, i32 } [ %lpad.loopexit1002, %lpad294.loopexit ], [ %lpad.loopexit.split-lp1003, %lpad294.loopexit.split-lp ]
  %266 = extractvalue { i8*, i32 } %lpad.phi1004, 0
  %267 = extractvalue { i8*, i32 } %lpad.phi1004, 1
  sync within %syncreg219, label %ehcleanup425

sync.continue296:                                 ; preds = %pfor.cond.cleanup229
  store %union.anon* %202, %union.anon** %203, align 8, !tbaa !10
  store i32 1953656691, i32* %205, align 8
  store i64 4, i64* %_M_string_length.i.i.i.i.i.i880, align 8, !tbaa !13
  store i8 0, i8* %arrayidx.i.i.i.i.i881, align 4, !tbaa !16
  %call2.i.i902 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull %204, i64 4)
          to label %call2.i.i.noexc901 unwind label %lpad307

call2.i.i.noexc901:                               ; preds = %sync.continue296
  %call1.i.i904 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) %call2.i.i902, i8* nonnull getelementptr inbounds ([4 x i8], [4 x i8]* @.str.17, i64 0, i64 0), i64 3)
          to label %call1.i.i.noexc903 unwind label %lpad307

call1.i.i.noexc903:                               ; preds = %call2.i.i.noexc901
  %268 = load i8, i8* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 3), align 8, !tbaa !17, !range !22
  %tobool.i.i.i888 = icmp eq i8 %268, 0
  br i1 %tobool.i.i.i888, label %_ZN5timer10reportNextEv.exit.i900, label %if.end.i.i.i898

if.end.i.i.i898:                                  ; preds = %call1.i.i.noexc903
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %206) #2
  %call.i.i.i.i889 = call i32 @gettimeofday(%struct.timeval* nonnull %now.i.i.i.i885, %struct.timezone* nonnull getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 5)) #2
  %269 = load i64, i64* %tv_sec.i.i.i.i890, align 8, !tbaa !23
  %conv.i.i.i.i891 = sitofp i64 %269 to double
  %270 = load i64, i64* %tv_usec.i.i.i.i892, align 8, !tbaa !25
  %conv2.i.i.i.i893 = sitofp i64 %270 to double
  %div.i.i.i.i894 = fmul fast double %conv2.i.i.i.i893, 0x3EB0C6F7A0B5ED8D
  %add.i.i.i.i895 = fadd fast double %div.i.i.i.i894, %conv.i.i.i.i891
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %206) #2
  %271 = load double, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 1), align 8, !tbaa !26
  %sub.i.i.i896 = fsub fast double %add.i.i.i.i895, %271
  %272 = load double, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 0), align 8, !tbaa !27
  %add.i.i.i897 = fadd fast double %sub.i.i.i896, %272
  store double %add.i.i.i897, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 0), align 8, !tbaa !27
  store double %add.i.i.i.i895, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 1), align 8, !tbaa !26
  br label %_ZN5timer10reportNextEv.exit.i900

_ZN5timer10reportNextEv.exit.i900:                ; preds = %if.end.i.i.i898, %call1.i.i.noexc903
  %retval.0.i.i.i899 = phi double [ %sub.i.i.i896, %if.end.i.i.i898 ], [ 0.000000e+00, %call1.i.i.noexc903 ]
  invoke void @_ZN5timer7reportTEd(%struct.timer* nonnull @_ZL3_tm, double %retval.0.i.i.i899)
          to label %invoke.cont308 unwind label %lpad307

invoke.cont308:                                   ; preds = %_ZN5timer10reportNextEv.exit.i900
  %273 = load i8*, i8** %_M_p.i.i.i.i886, align 8, !tbaa !28
  %cmp.i.i.i908 = icmp eq i8* %273, %204
  br i1 %cmp.i.i.i908, label %pfor.detach323.preheader, label %if.then.i.i909

pfor.detach323.preheader:                         ; preds = %if.then.i.i909, %invoke.cont308
  br label %pfor.detach323

if.then.i.i909:                                   ; preds = %invoke.cont308
  call void @_ZdlPv(i8* %273) #2
  br label %pfor.detach323.preheader

pfor.cond.cleanup322:                             ; preds = %pfor.inc349
  sync within %syncreg312, label %sync.continue358

lpad307:                                          ; preds = %_ZN5timer10reportNextEv.exit.i900, %call2.i.i.noexc901, %sync.continue296
  %274 = landingpad { i8*, i32 }
          cleanup
  %275 = extractvalue { i8*, i32 } %274, 0
  %276 = extractvalue { i8*, i32 } %274, 1
  %277 = load i8*, i8** %_M_p.i.i.i.i886, align 8, !tbaa !28
  %cmp.i.i.i913 = icmp eq i8* %277, %204
  br i1 %cmp.i.i.i913, label %ehcleanup425, label %if.then.i.i914

if.then.i.i914:                                   ; preds = %lpad307
  call void @_ZdlPv(i8* %277) #2
  br label %ehcleanup425

pfor.detach323:                                   ; preds = %pfor.detach323.preheader, %pfor.inc349
  %indvars.iv1126 = phi i64 [ %indvars.iv.next1127, %pfor.inc349 ], [ 0, %pfor.detach323.preheader ]
  detach within %syncreg312, label %pfor.body328, label %pfor.inc349 unwind label %lpad351.loopexit

pfor.body328:                                     ; preds = %pfor.detach323
  %start332 = getelementptr inbounds %struct.seg, %struct.seg* %193, i64 %indvars.iv1126, i32 0
  %278 = load i32, i32* %start332, align 4, !tbaa !55
  %arrayidx334 = getelementptr inbounds i32, i32* %192, i64 %indvars.iv1126
  %279 = load i32, i32* %arrayidx334, align 4, !tbaa !6
  %idx.ext335 = zext i32 %279 to i64
  %add.ptr336 = getelementptr inbounds %struct.seg, %struct.seg* %181, i64 %idx.ext335
  %length339 = getelementptr inbounds %struct.seg, %struct.seg* %193, i64 %indvars.iv1126, i32 1
  %280 = load i32, i32* %length339, align 4, !tbaa !52
  %idx.ext340 = zext i32 %278 to i64
  %add.ptr341 = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %call143, i64 %idx.ext340
  invoke void @_Z12splitSegmentISt4pairIjjEEvP3segjjPjPT_(%struct.seg* %add.ptr336, i32 %278, i32 %280, i32* %180, %"struct.std::pair"* %add.ptr341)
          to label %invoke.cont345 unwind label %lpad342

invoke.cont345:                                   ; preds = %pfor.body328
  reattach within %syncreg312, label %pfor.inc349

pfor.inc349:                                      ; preds = %pfor.detach323, %invoke.cont345
  %indvars.iv.next1127 = add nuw nsw i64 %indvars.iv1126, 1
  %exitcond1129 = icmp eq i64 %indvars.iv.next1127, %conv.i802
  br i1 %exitcond1129, label %pfor.cond.cleanup322, label %pfor.detach323, !llvm.loop !60

lpad342:                                          ; preds = %pfor.body328
  %281 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg312, { i8*, i32 } %281)
          to label %det.rethrow.unreachable357 unwind label %lpad351.loopexit.split-lp

det.rethrow.unreachable357:                       ; preds = %lpad342
  unreachable

lpad351.loopexit:                                 ; preds = %pfor.detach323
  %lpad.loopexit = landingpad { i8*, i32 }
          cleanup
  br label %lpad351

lpad351.loopexit.split-lp:                        ; preds = %lpad342
  %lpad.loopexit.split-lp = landingpad { i8*, i32 }
          cleanup
  br label %lpad351

lpad351:                                          ; preds = %lpad351.loopexit.split-lp, %lpad351.loopexit
  %lpad.phi = phi { i8*, i32 } [ %lpad.loopexit, %lpad351.loopexit ], [ %lpad.loopexit.split-lp, %lpad351.loopexit.split-lp ]
  %282 = extractvalue { i8*, i32 } %lpad.phi, 0
  %283 = extractvalue { i8*, i32 } %lpad.phi, 1
  sync within %syncreg312, label %ehcleanup425

sync.continue358:                                 ; preds = %pfor.cond.cleanup322
  store %union.anon* %207, %union.anon** %208, align 8, !tbaa !10
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull %209, i8* nonnull getelementptr inbounds ([6 x i8], [6 x i8]* @.str.11, i64 0, i64 0), i64 5, i32 1, i1 false) #2
  store i64 5, i64* %_M_string_length.i.i.i.i.i.i928, align 8, !tbaa !13
  store i8 0, i8* %arrayidx.i.i.i.i.i929, align 1, !tbaa !16
  %call2.i.i950 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull %209, i64 5)
          to label %call2.i.i.noexc949 unwind label %lpad369

call2.i.i.noexc949:                               ; preds = %sync.continue358
  %call1.i.i952 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) %call2.i.i950, i8* nonnull getelementptr inbounds ([4 x i8], [4 x i8]* @.str.17, i64 0, i64 0), i64 3)
          to label %call1.i.i.noexc951 unwind label %lpad369

call1.i.i.noexc951:                               ; preds = %call2.i.i.noexc949
  %284 = load i8, i8* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 3), align 8, !tbaa !17, !range !22
  %tobool.i.i.i936 = icmp eq i8 %284, 0
  br i1 %tobool.i.i.i936, label %_ZN5timer10reportNextEv.exit.i948, label %if.end.i.i.i946

if.end.i.i.i946:                                  ; preds = %call1.i.i.noexc951
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %210) #2
  %call.i.i.i.i937 = call i32 @gettimeofday(%struct.timeval* nonnull %now.i.i.i.i933, %struct.timezone* nonnull getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 5)) #2
  %285 = load i64, i64* %tv_sec.i.i.i.i938, align 8, !tbaa !23
  %conv.i.i.i.i939 = sitofp i64 %285 to double
  %286 = load i64, i64* %tv_usec.i.i.i.i940, align 8, !tbaa !25
  %conv2.i.i.i.i941 = sitofp i64 %286 to double
  %div.i.i.i.i942 = fmul fast double %conv2.i.i.i.i941, 0x3EB0C6F7A0B5ED8D
  %add.i.i.i.i943 = fadd fast double %div.i.i.i.i942, %conv.i.i.i.i939
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %210) #2
  %287 = load double, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 1), align 8, !tbaa !26
  %sub.i.i.i944 = fsub fast double %add.i.i.i.i943, %287
  %288 = load double, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 0), align 8, !tbaa !27
  %add.i.i.i945 = fadd fast double %sub.i.i.i944, %288
  store double %add.i.i.i945, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 0), align 8, !tbaa !27
  store double %add.i.i.i.i943, double* getelementptr inbounds (%struct.timer, %struct.timer* @_ZL3_tm, i64 0, i32 1), align 8, !tbaa !26
  br label %_ZN5timer10reportNextEv.exit.i948

_ZN5timer10reportNextEv.exit.i948:                ; preds = %if.end.i.i.i946, %call1.i.i.noexc951
  %retval.0.i.i.i947 = phi double [ %sub.i.i.i944, %if.end.i.i.i946 ], [ 0.000000e+00, %call1.i.i.noexc951 ]
  invoke void @_ZN5timer7reportTEd(%struct.timer* nonnull @_ZL3_tm, double %retval.0.i.i.i947)
          to label %invoke.cont370 unwind label %lpad369

invoke.cont370:                                   ; preds = %_ZN5timer10reportNextEv.exit.i948
  %289 = load i8*, i8** %_M_p.i.i.i.i934, align 8, !tbaa !28
  %cmp.i.i.i956 = icmp eq i8* %289, %209
  br i1 %cmp.i.i.i956, label %cleanup.cont, label %if.then.i.i957

if.then.i.i957:                                   ; preds = %invoke.cont370
  call void @_ZdlPv(i8* %289) #2
  br label %cleanup.cont

cleanup.cont:                                     ; preds = %invoke.cont370, %if.then.i.i957
  %mul374 = shl i32 %offset.0, 1
  br label %while.cond

lpad369:                                          ; preds = %_ZN5timer10reportNextEv.exit.i948, %call2.i.i.noexc949, %sync.continue358
  %290 = landingpad { i8*, i32 }
          cleanup
  %291 = extractvalue { i8*, i32 } %290, 0
  %292 = extractvalue { i8*, i32 } %290, 1
  %293 = load i8*, i8** %_M_p.i.i.i.i934, align 8, !tbaa !28
  %cmp.i.i.i = icmp eq i8* %293, %209
  br i1 %cmp.i.i.i, label %ehcleanup425, label %if.then.i.i

if.then.i.i:                                      ; preds = %lpad369
  call void @_ZdlPv(i8* %293) #2
  br label %ehcleanup425

while.end:                                        ; preds = %if.then.i771, %_ZN8sequence6filterI3segj5isSegEET0_PT_S5_S3_T1_.exit
  br i1 %cmp31094, label %pfor.cond.cleanup387, label %pfor.detach388.lr.ph

pfor.detach388.lr.ph:                             ; preds = %while.end
  %wide.trip.count = and i64 %n, 4294967295
  br label %pfor.detach388

pfor.cond.cleanup387:                             ; preds = %pfor.inc400, %while.end
  sync within %syncreg376, label %sync.continue403

pfor.detach388:                                   ; preds = %pfor.inc400, %pfor.detach388.lr.ph
  %indvars.iv = phi i64 [ 0, %pfor.detach388.lr.ph ], [ %indvars.iv.next, %pfor.inc400 ]
  detach within %syncreg376, label %pfor.body393, label %pfor.inc400

pfor.body393:                                     ; preds = %pfor.detach388
  %second396 = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %call143, i64 %indvars.iv, i32 1
  %294 = load i32, i32* %second396, align 4, !tbaa !29
  %arrayidx398 = getelementptr inbounds i32, i32* %180, i64 %indvars.iv
  store i32 %294, i32* %arrayidx398, align 4, !tbaa !6
  reattach within %syncreg376, label %pfor.inc400

pfor.inc400:                                      ; preds = %pfor.body393, %pfor.detach388
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %pfor.cond.cleanup387, label %pfor.detach388, !llvm.loop !61

sync.continue403:                                 ; preds = %pfor.cond.cleanup387
  %295 = bitcast %"struct.std::pair"* %call143 to i8*
  call void @free(i8* %295) #2
  call void @free(i8* %call141) #2
  call void @free(i8* %call156) #2
  call void @free(i8* %call154) #2
  call void @llvm.lifetime.end.p0i8(i64 1024, i8* nonnull %0) #2
  ret i32* %180

ehcleanup425:                                     ; preds = %lpad369, %if.then.i.i, %lpad307, %if.then.i.i914, %lpad214, %if.then.i.i866, %lpad163, %lpad165, %if.then.i.i797, %lpad148, %if.then.i.i791, %lpad133, %if.then.i.i786, %lpad125, %if.then.i.i781, %lpad294, %lpad351
  %ehselector.slot.10 = phi i32 [ %267, %lpad294 ], [ %283, %lpad351 ], [ %224, %lpad125 ], [ %224, %if.then.i.i781 ], [ %228, %lpad133 ], [ %228, %if.then.i.i786 ], [ %232, %lpad148 ], [ %232, %if.then.i.i791 ], [ %236, %lpad163 ], [ %239, %lpad165 ], [ %239, %if.then.i.i797 ], [ %259, %lpad214 ], [ %259, %if.then.i.i866 ], [ %276, %lpad307 ], [ %276, %if.then.i.i914 ], [ %292, %lpad369 ], [ %292, %if.then.i.i ]
  %exn.slot.10 = phi i8* [ %266, %lpad294 ], [ %282, %lpad351 ], [ %223, %lpad125 ], [ %223, %if.then.i.i781 ], [ %227, %lpad133 ], [ %227, %if.then.i.i786 ], [ %231, %lpad148 ], [ %231, %if.then.i.i791 ], [ %235, %lpad163 ], [ %238, %lpad165 ], [ %238, %if.then.i.i797 ], [ %258, %lpad214 ], [ %258, %if.then.i.i866 ], [ %275, %lpad307 ], [ %275, %if.then.i.i914 ], [ %291, %lpad369 ], [ %291, %if.then.i.i ]
  call void @llvm.lifetime.end.p0i8(i64 1024, i8* nonnull %0) #2
  %lpad.val441 = insertvalue { i8*, i32 } undef, i8* %exn.slot.10, 0
  %lpad.val442 = insertvalue { i8*, i32 } %lpad.val441, i32 %ehselector.slot.10, 1
  resume { i8*, i32 } %lpad.val442

if.then.i.i773.1:                                 ; preds = %for.inc.i.i
  %inc.i.i.1 = add i32 %k.1.i.i, 1
  %idxprom3.i.i.1 = zext i32 %k.1.i.i to i64
  %arrayidx4.i.i.1 = getelementptr inbounds %struct.seg, %struct.seg* %193, i64 %idxprom3.i.i.1
  %296 = bitcast %struct.seg* %arrayidx4.i.i.1 to i64*
  store i64 %agg.tmp.sroa.0.0.copyload.i.i.1, i64* %296, align 4
  br label %for.inc.i.i.1

for.inc.i.i.1:                                    ; preds = %if.then.i.i773.1, %for.inc.i.i
  %k.1.i.i.1 = phi i32 [ %inc.i.i.1, %if.then.i.i773.1 ], [ %k.1.i.i, %for.inc.i.i ]
  %indvars.iv.next.i.i.1 = or i64 %indvars.iv.i.i, 2
  %arrayidx.i.i.2 = getelementptr inbounds %struct.seg, %struct.seg* %181, i64 %indvars.iv.next.i.i.1
  %agg.tmp.sroa.0.0..sroa_cast.i.i.2 = bitcast %struct.seg* %arrayidx.i.i.2 to i64*
  %agg.tmp.sroa.0.0.copyload.i.i.2 = load i64, i64* %agg.tmp.sroa.0.0..sroa_cast.i.i.2, align 4
  %s.sroa.1.0.extract.shift.i.i.i.2 = lshr i64 %agg.tmp.sroa.0.0.copyload.i.i.2, 32
  %s.sroa.1.0.extract.trunc.i.i.i.2 = trunc i64 %s.sroa.1.0.extract.shift.i.i.i.2 to i32
  %cmp.i.i.i772.2 = icmp ugt i32 %s.sroa.1.0.extract.trunc.i.i.i.2, 1
  br i1 %cmp.i.i.i772.2, label %if.then.i.i773.2, label %for.inc.i.i.2

if.then.i.i773.2:                                 ; preds = %for.inc.i.i.1
  %inc.i.i.2 = add i32 %k.1.i.i.1, 1
  %idxprom3.i.i.2 = zext i32 %k.1.i.i.1 to i64
  %arrayidx4.i.i.2 = getelementptr inbounds %struct.seg, %struct.seg* %193, i64 %idxprom3.i.i.2
  %297 = bitcast %struct.seg* %arrayidx4.i.i.2 to i64*
  store i64 %agg.tmp.sroa.0.0.copyload.i.i.2, i64* %297, align 4
  br label %for.inc.i.i.2

for.inc.i.i.2:                                    ; preds = %if.then.i.i773.2, %for.inc.i.i.1
  %k.1.i.i.2 = phi i32 [ %inc.i.i.2, %if.then.i.i773.2 ], [ %k.1.i.i.1, %for.inc.i.i.1 ]
  %indvars.iv.next.i.i.2 = or i64 %indvars.iv.i.i, 3
  %arrayidx.i.i.3 = getelementptr inbounds %struct.seg, %struct.seg* %181, i64 %indvars.iv.next.i.i.2
  %agg.tmp.sroa.0.0..sroa_cast.i.i.3 = bitcast %struct.seg* %arrayidx.i.i.3 to i64*
  %agg.tmp.sroa.0.0.copyload.i.i.3 = load i64, i64* %agg.tmp.sroa.0.0..sroa_cast.i.i.3, align 4
  %s.sroa.1.0.extract.shift.i.i.i.3 = lshr i64 %agg.tmp.sroa.0.0.copyload.i.i.3, 32
  %s.sroa.1.0.extract.trunc.i.i.i.3 = trunc i64 %s.sroa.1.0.extract.shift.i.i.i.3 to i32
  %cmp.i.i.i772.3 = icmp ugt i32 %s.sroa.1.0.extract.trunc.i.i.i.3, 1
  br i1 %cmp.i.i.i772.3, label %if.then.i.i773.3, label %for.inc.i.i.3

if.then.i.i773.3:                                 ; preds = %for.inc.i.i.2
  %inc.i.i.3 = add i32 %k.1.i.i.2, 1
  %idxprom3.i.i.3 = zext i32 %k.1.i.i.2 to i64
  %arrayidx4.i.i.3 = getelementptr inbounds %struct.seg, %struct.seg* %193, i64 %idxprom3.i.i.3
  %298 = bitcast %struct.seg* %arrayidx4.i.i.3 to i64*
  store i64 %agg.tmp.sroa.0.0.copyload.i.i.3, i64* %298, align 4
  br label %for.inc.i.i.3

for.inc.i.i.3:                                    ; preds = %if.then.i.i773.3, %for.inc.i.i.2
  %k.1.i.i.3 = phi i32 [ %inc.i.i.3, %if.then.i.i773.3 ], [ %k.1.i.i.2, %for.inc.i.i.2 ]
  %indvars.iv.next.i.i.3 = add nuw nsw i64 %indvars.iv.i.i, 4
  %niter.nsub.3 = add i64 %niter, -4
  %niter.ncmp.3 = icmp eq i64 %niter.nsub.3, 0
  br i1 %niter.ncmp.3, label %_ZN8sequence6filterI3segj5isSegEET0_PT_S5_S3_T1_.exit.loopexit.unr-lcssa, label %for.body.i.i
}

; Function Attrs: uwtable
declare void @_Z10sampleSortISt4pairIjjE9pairCompFjEvPT_T1_T0_(%"struct.std::pair"* %A, i32 %n) local_unnamed_addr #4

; Function Attrs: uwtable
declare void @_Z10sampleSortIoSt4lessIoElEvPT_T1_T0_(i128* %A, i64 %n) local_unnamed_addr #4

; Function Attrs: uwtable
declare void @_Z12splitSegmentISt4pairIjjEEvP3segjjPjPT_(%struct.seg* %segOut, i32 %start, i32 %l, i32* %ranks, %"struct.std::pair"* %Cs) local_unnamed_addr #4

; Function Attrs: uwtable
declare noalias %"struct.std::pair"* @_Z15splitSegmentTopP3segjPjPo(%struct.seg* nocapture %segOut, i32 %n, i32* nocapture %ranks, i128* nocapture %Cs) local_unnamed_addr #4

; Function Attrs: uwtable
declare void @_Z9quickSortISt4pairIjjE9pairCompFjEvPT_T1_T0_(%"struct.std::pair"* %A, i32 %n) local_unnamed_addr #4

; Function Attrs: uwtable
declare void @_ZN5timer7reportTEd(%struct.timer* %this, double %time) local_unnamed_addr #4

; Function Attrs: uwtable
declare { %struct.seg*, i64 } @_ZN8sequence4packI3segjNS_4getAIS1_jEEEE4_seqIT_EPS5_PbT0_S9_T1_(%struct.seg* %Out, i8* %Fl, i32 %s, i32 %e, %struct.seg* %f.coerce) local_unnamed_addr #4

; Function Attrs: uwtable
declare i32 @_ZN8sequence4scanIjjN5utils4addFIjEENS_4getAIjjEEEET_PS6_T0_S8_T1_T2_S6_bb(i32* %Out, i32 %s, i32 %e, i32* %g.coerce, i32 %zero, i1 zeroext %inclusive, i1 zeroext %back) local_unnamed_addr #4

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"*) local_unnamed_addr #0

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"*, i8 signext) local_unnamed_addr #0

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"*) local_unnamed_addr #0

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"*, double) local_unnamed_addr #0

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIlEERSoT_(%"class.std::basic_ostream"*, i64) local_unnamed_addr #0

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertImEERSoT_(%"class.std::basic_ostream"*, i64) local_unnamed_addr #0

declare i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(%"class.std::__cxx11::basic_string"*, i64* dereferenceable(8), i64) local_unnamed_addr #0

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* dereferenceable(272), i8*, i64) local_unnamed_addr #0

; Function Attrs: noreturn
declare void @_ZSt16__throw_bad_castv() local_unnamed_addr #11

; Function Attrs: inlinehint uwtable
declare dereferenceable(272) %"class.std::basic_ostream"* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(%"class.std::basic_ostream"* dereferenceable(272)) local_unnamed_addr #7

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(i8*) local_unnamed_addr #10

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #5

; Function Attrs: nounwind
declare noalias i8* @malloc(i64) local_unnamed_addr #1

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #5

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #5

declare i32 @__gxx_personality_v0(...)

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #5

; Function Attrs: argmemonly
declare void @llvm.detached.rethrow.sl_p0i8i32s(token, { i8*, i32 }) #6

; Function Attrs: nounwind
declare void @free(i8* nocapture) local_unnamed_addr #1

; Function Attrs: noreturn nounwind
declare void @abort() local_unnamed_addr #9

; Function Attrs: nounwind
declare i32 @gettimeofday(%struct.timeval* nocapture, %struct.timezone* nocapture) local_unnamed_addr #1

; Function Attrs: nounwind readnone speculatable
declare double @llvm.floor.f64(double) #8

; Function Attrs: nounwind readnone speculatable
declare double @llvm.log2.f64(double) #8

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #5

; CHECK: define internal fastcc void @_Z19suffixArrayInternalPhl_pfor.detach.ls1(i64 %indvars.iv1146.start.ls1
; CHECK: define internal fastcc void @_Z19suffixArrayInternalPhl_pfor.detach36.ls1(i64 %indvars.iv1142.start.ls1
; CHECK: define internal fastcc void @_Z19suffixArrayInternalPhl_pfor.detach90.ls1(i64 %indvars.iv1138.start.ls1
; CHECK: define internal fastcc void @_Z19suffixArrayInternalPhl_pfor.detach90.us.ls1(i64 %indvars.iv1134.start.ls1
; CHECK: define internal fastcc void @_Z19suffixArrayInternalPhl_pfor.detach.i.ls2(i64 %indvars.iv.i.start.ls2
; CHECK: define internal fastcc void @_Z19suffixArrayInternalPhl_pfor.detach323.ls2(i64 %indvars.iv1126.start.ls2
; CHECK: define internal fastcc void @_Z19suffixArrayInternalPhl_pfor.detach254.ls3(i64 %indvars.iv1118.start.ls3
; CHECK: define internal fastcc void @_Z19suffixArrayInternalPhl_pfor.detach230.ls2(i64 %indvars.iv1122.start.ls2
; CHECK: pfor.detach254.lr.ph.ls2:
; CHECK: call fastcc void @_Z19suffixArrayInternalPhl_pfor.detach254.ls3(i64 0, i64 %wide.trip.count1120.ls2
; CHECK: define internal fastcc void @_Z19suffixArrayInternalPhl_pfor.detach186.ls2(i64 %indvars.iv1114.start.ls2
; CHECK: define internal fastcc void @_Z19suffixArrayInternalPhl_pfor.detach388.ls1(i64 %indvars.iv.start.ls1

!2 = !{!3, !3, i64 0}
!3 = !{!"__int128", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !4, i64 0}
!9 = !{!"tapir.loop.spawn.strategy", i32 1}
!10 = !{!11, !12, i64 0}
!11 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderE", !12, i64 0}
!12 = !{!"any pointer", !4, i64 0}
!13 = !{!14, !15, i64 8}
!14 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", !11, i64 0, !15, i64 8, !4, i64 16}
!15 = !{!"long", !4, i64 0}
!16 = !{!4, !4, i64 0}
!17 = !{!18, !20, i64 24}
!18 = !{!"_ZTS5timer", !19, i64 0, !19, i64 8, !19, i64 16, !20, i64 24, !21, i64 28}
!19 = !{!"double", !4, i64 0}
!20 = !{!"bool", !4, i64 0}
!21 = !{!"_ZTS8timezone", !7, i64 0, !7, i64 4}
!22 = !{i8 0, i8 2}
!23 = !{!24, !15, i64 0}
!24 = !{!"_ZTS7timeval", !15, i64 0, !15, i64 8}
!25 = !{!24, !15, i64 8}
!26 = !{!18, !19, i64 8}
!27 = !{!18, !19, i64 0}
!28 = !{!14, !12, i64 0}
!29 = !{!30, !7, i64 4}
!30 = !{!"_ZTSSt4pairIjjE", !7, i64 0, !7, i64 4}
!32 = !{!15, !15, i64 0}
!34 = distinct !{!34, !9}
!35 = !{!36, !36, i64 0}
!36 = !{!"vtable pointer", !5, i64 0}
!37 = !{!38, !12, i64 240}
!38 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !12, i64 216, !4, i64 224, !20, i64 225, !12, i64 232, !12, i64 240, !12, i64 248, !12, i64 256}
!39 = !{!40, !4, i64 56}
!40 = !{!"_ZTSSt5ctypeIcE", !12, i64 16, !20, i64 24, !12, i64 32, !12, i64 40, !12, i64 48, !4, i64 56, !4, i64 57, !4, i64 313, !4, i64 569}
!41 = distinct !{!41, !9}
!42 = distinct !{!42, !43}
!43 = !{!"llvm.loop.isvectorized", i32 1}
!44 = distinct !{!44, !45}
!45 = !{!"llvm.loop.unroll.disable"}
!46 = distinct !{!46, !9}
!47 = distinct !{!47, !45}
!48 = distinct !{!48, !43}
!49 = !{!20, !20, i64 0}
!50 = distinct !{!50, !9}
!51 = distinct !{!51, !45}
!52 = !{!53, !7, i64 4}
!53 = !{!"_ZTS3seg", !7, i64 0, !7, i64 4}
!54 = distinct !{!54, !9}
!55 = !{!53, !7, i64 0}
!56 = !{!30, !7, i64 0}
!57 = distinct !{!57, !9, !58}
!58 = !{!"tapir.loop.grainsize", i32 256}
!59 = distinct !{!59, !9}
!60 = distinct !{!60, !9}
!61 = distinct !{!61, !9}
