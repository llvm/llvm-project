; RUN: opt < %s -analyze -tasks -verify-task-info -S 2>&1 | FileCheck %s

%"class.std::ios_base::Init" = type { i8 }
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
%class.matrix_serial = type <{ float*, i32, [4 x i8] }>
%struct.rgb = type { i8, i8, i8 }
%class.CUtilTimer = type { double, double, i64, i64 }
%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external hidden global i8
@_ZZ20process_image_serialP3rgbS0_iE7quant90 = private unnamed_addr constant [64 x float] [float 3.000000e+00, float 2.000000e+00, float 2.000000e+00, float 3.000000e+00, float 5.000000e+00, float 8.000000e+00, float 1.000000e+01, float 1.200000e+01, float 2.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00, float 5.000000e+00, float 1.200000e+01, float 1.200000e+01, float 1.100000e+01, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 5.000000e+00, float 8.000000e+00, float 1.100000e+01, float 1.400000e+01, float 1.100000e+01, float 3.000000e+00, float 3.000000e+00, float 4.000000e+00, float 6.000000e+00, float 1.000000e+01, float 1.700000e+01, float 1.600000e+01, float 1.200000e+01, float 4.000000e+00, float 4.000000e+00, float 7.000000e+00, float 1.100000e+01, float 1.400000e+01, float 2.200000e+01, float 2.100000e+01, float 1.500000e+01, float 5.000000e+00, float 7.000000e+00, float 1.100000e+01, float 1.300000e+01, float 1.600000e+01, float 1.200000e+01, float 2.300000e+01, float 1.800000e+01, float 1.000000e+01, float 1.300000e+01, float 1.600000e+01, float 1.700000e+01, float 2.100000e+01, float 2.400000e+01, float 2.400000e+01, float 2.100000e+01, float 1.400000e+01, float 1.800000e+01, float 1.900000e+01, float 2.000000e+01, float 2.200000e+01, float 2.000000e+01, float 2.000000e+01, float 2.000000e+01], align 16
@.str = private unnamed_addr constant [3 x i8] c"rb\00", align 1
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str.1 = private unnamed_addr constant [61 x i8] c"The input file could not be opened. Program will be exiting\0A\00", align 1
@.str.2 = private unnamed_addr constant [49 x i8] c"Unable to allocate the memory for bitmap header\0A\00", align 1
@.str.3 = private unnamed_addr constant [78 x i8] c"Read error from the file. No bytes were read from the file. Program exiting \0A\00", align 1
@.str.4 = private unnamed_addr constant [25 x i8] c"This is not a RGB image\0A\00", align 1
@.str.5 = private unnamed_addr constant [47 x i8] c"Unable to allocate the memory for bitmap date\0A\00", align 1
@.str.6 = private unnamed_addr constant [14 x i8] c"Wrong choice\0A\00", align 1
@.str.7 = private unnamed_addr constant [3 x i8] c"wb\00", align 1
@.str.8 = private unnamed_addr constant [55 x i8] c"The file could not be opened. Program will be exiting\0A\00", align 1
@.str.9 = private unnamed_addr constant [77 x i8] c"Write error to the file. No bytes were wrtten to the file. Program exiting \0A\00", align 1
@.str.10 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@.str.11 = private unnamed_addr constant [70 x i8] c"Program usage is <modified_program> <inputfile.bmp> <outputfile.bmp>\0A\00", align 1

; Function Attrs: uwtable
declare void @_Z20process_image_serialP3rgbS0_i(%struct.rgb* nocapture readonly %indataset, %struct.rgb* nocapture %outdataset, i32 %startindex) local_unnamed_addr #5 align 32

declare void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) unnamed_addr #0

; Function Attrs: nounwind
declare void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) unnamed_addr #1

; Function Attrs: nounwind
declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) local_unnamed_addr #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #4

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #4

; Function Attrs: nounwind
declare float @cosf(float) local_unnamed_addr #1

declare void @_ZN13matrix_serialC1Ei(%class.matrix_serial*, i32) unnamed_addr #0

declare i32 @__gxx_personality_v0(...)

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #4

declare void @_ZN13matrix_serial9transposeERS_(%class.matrix_serial*, %class.matrix_serial* dereferenceable(16)) local_unnamed_addr #0

declare void @_ZN13matrix_serialmlERS_(%class.matrix_serial* sret, %class.matrix_serial*, %class.matrix_serial* dereferenceable(16)) local_unnamed_addr #0

declare dereferenceable(16) %class.matrix_serial* @_ZN13matrix_serialaSERKS_(%class.matrix_serial*, %class.matrix_serial* dereferenceable(16)) local_unnamed_addr #0

; Function Attrs: nounwind
declare void @_ZN13matrix_serialD1Ev(%class.matrix_serial*) unnamed_addr #1

; Function Attrs: uwtable
define i32 @_Z18read_process_writePcS_i(i8* nocapture readonly %input, i8* nocapture readonly %output, i32 %choice) local_unnamed_addr #5 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %__mallocedMemory.i262 = alloca i8*, align 8
  %__mallocedMemory.i = alloca i8*, align 8
  %t = alloca %class.CUtilTimer, align 8
  %syncreg = tail call token @llvm.syncregion.start()
  %syncreg72 = tail call token @llvm.syncregion.start()
  %0 = bitcast %class.CUtilTimer* %t to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %0) #2
  call void @llvm.memset.p0i8.i64(i8* nonnull %0, i8 0, i64 32, i32 8, i1 false) #2
  %call = tail call %struct._IO_FILE* @fopen(i8* %input, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0))
  %cmp = icmp eq %struct._IO_FILE* %call, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call1.i = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull getelementptr inbounds ([61 x i8], [61 x i8]* @.str.1, i64 0, i64 0), i64 60)
  br label %cleanup140

if.end:                                           ; preds = %entry
  %call2 = tail call noalias i8* @malloc(i64 54) #2
  %cmp3 = icmp eq i8* %call2, null
  br i1 %cmp3, label %if.then4, label %if.end6

if.then4:                                         ; preds = %if.end
  %call1.i253 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull getelementptr inbounds ([49 x i8], [49 x i8]* @.str.2, i64 0, i64 0), i64 48)
  br label %cleanup140

if.end6:                                          ; preds = %if.end
  %call7 = tail call i64 @fread(i8* nonnull %call2, i64 54, i64 1, %struct._IO_FILE* nonnull %call)
  %cmp8 = icmp eq i64 %call7, 0
  br i1 %cmp8, label %if.then9, label %if.end11

if.then9:                                         ; preds = %if.end6
  %call1.i255 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull getelementptr inbounds ([78 x i8], [78 x i8]* @.str.3, i64 0, i64 0), i64 77)
  br label %cleanup140

if.end11:                                         ; preds = %if.end6
  %bitsperpixel = getelementptr inbounds i8, i8* %call2, i64 28
  %1 = bitcast i8* %bitsperpixel to i16*
  %2 = load i16, i16* %1, align 1, !tbaa !61
  %cmp12 = icmp eq i16 %2, 24
  br i1 %cmp12, label %if.end15, label %if.then13

if.then13:                                        ; preds = %if.end11
  %call1.i257 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull getelementptr inbounds ([25 x i8], [25 x i8]* @.str.4, i64 0, i64 0), i64 24)
  br label %cleanup140

if.end15:                                         ; preds = %if.end11
  %width = getelementptr inbounds i8, i8* %call2, i64 18
  %3 = bitcast i8* %width to i32*
  %4 = load i32, i32* %3, align 1, !tbaa !65
  %height = getelementptr inbounds i8, i8* %call2, i64 22
  %5 = bitcast i8* %height to i32*
  %6 = load i32, i32* %5, align 1, !tbaa !66
  %mul = mul nsw i32 %6, %4
  %conv16 = sext i32 %mul to i64
  %mul17 = mul nsw i64 %conv16, 3
  %7 = bitcast i8** %__mallocedMemory.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %7) #2
  %call4.i = call i32 @posix_memalign(i8** nonnull %__mallocedMemory.i, i64 32, i64 %mul17) #2
  %tobool5.i = icmp eq i32 %call4.i, 0
  %8 = load i8*, i8** %__mallocedMemory.i, align 8
  %retval.0.i = select i1 %tobool5.i, i8* %8, i8* null
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %7) #2
  %9 = bitcast i8* %retval.0.i to %struct.rgb*
  %cmp19 = icmp eq i8* %retval.0.i, null
  br i1 %cmp19, label %if.then20, label %if.end22

if.then20:                                        ; preds = %if.end15
  %call1.i259 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull getelementptr inbounds ([47 x i8], [47 x i8]* @.str.5, i64 0, i64 0), i64 46)
  br label %cleanup140

if.end22:                                         ; preds = %if.end15
  %dataoffset = getelementptr inbounds i8, i8* %call2, i64 10
  %10 = bitcast i8* %dataoffset to i32*
  %11 = load i32, i32* %10, align 1, !tbaa !67
  %conv23 = zext i32 %11 to i64
  %call25 = call i32 @fseek(%struct._IO_FILE* nonnull %call, i64 %conv23, i32 0)
  %call27 = call i64 @fread(i8* nonnull %retval.0.i, i64 3, i64 %conv16, %struct._IO_FILE* nonnull %call)
  %cmp28 = icmp eq i64 %call27, 0
  br i1 %cmp28, label %if.then29, label %if.end31

if.then29:                                        ; preds = %if.end22
  %call1.i261 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull getelementptr inbounds ([78 x i8], [78 x i8]* @.str.3, i64 0, i64 0), i64 77)
  br label %cleanup140

if.end31:                                         ; preds = %if.end22
  %12 = bitcast i8** %__mallocedMemory.i262 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %12) #2
  %call4.i263 = call i32 @posix_memalign(i8** nonnull %__mallocedMemory.i262, i64 32, i64 %mul17) #2
  %tobool5.i264 = icmp eq i32 %call4.i263, 0
  %13 = load i8*, i8** %__mallocedMemory.i262, align 8
  %retval.0.i265 = select i1 %tobool5.i264, i8* %13, i8* null
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %12) #2
  %14 = bitcast i8* %retval.0.i265 to %struct.rgb*
  %cmp35 = icmp eq i8* %retval.0.i265, null
  br i1 %cmp35, label %if.then36, label %if.end38

if.then36:                                        ; preds = %if.end31
  %call1.i267 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull getelementptr inbounds ([47 x i8], [47 x i8]* @.str.5, i64 0, i64 0), i64 46)
  br label %cleanup140

if.end38:                                         ; preds = %if.end31
  %div = sdiv i32 %mul, 64
  %cmp41291 = icmp sgt i32 %mul, 63
  br label %for.body

for.cond.cleanup:                                 ; preds = %sw.epilog
  %call114 = call %struct._IO_FILE* @fopen(i8* %output, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.7, i64 0, i64 0))
  %cmp115 = icmp eq %struct._IO_FILE* %call114, null
  br i1 %cmp115, label %if.then116, label %if.end118

for.body:                                         ; preds = %if.end38, %sw.epilog
  %avg_time.0294 = phi double [ 0.000000e+00, %if.end38 ], [ %add110, %sw.epilog ]
  %j.0293 = phi i32 [ 0, %if.end38 ], [ %inc112, %sw.epilog ]
  switch i32 %choice, label %sw.default [
    i32 1, label %sw.bb
    i32 2, label %sw.bb45
    i32 3, label %sw.bb56
    i32 4, label %sw.bb71
  ]

sw.bb:                                            ; preds = %for.body
  call void @_ZN10CUtilTimer5startEv(%class.CUtilTimer* nonnull %t)
  br i1 %cmp41291, label %for.body43.preheader, label %for.cond.cleanup42

for.body43.preheader:                             ; preds = %sw.bb
  br label %for.body43

for.cond.cleanup42:                               ; preds = %for.body43, %sw.bb
  call void @_ZN10CUtilTimer4stopEv(%class.CUtilTimer* nonnull %t)
  br label %sw.epilog

for.body43:                                       ; preds = %for.body43.preheader, %for.body43
  %i.0292 = phi i32 [ %inc, %for.body43 ], [ 0, %for.body43.preheader ]
  %mul44 = shl nsw i32 %i.0292, 6
  call void @_Z20process_image_serialP3rgbS0_i(%struct.rgb* %9, %struct.rgb* %14, i32 %mul44)
  %inc = add nuw nsw i32 %i.0292, 1
  %exitcond295 = icmp eq i32 %inc, %div
  br i1 %exitcond295, label %for.cond.cleanup42, label %for.body43

sw.bb45:                                          ; preds = %for.body
  call void @_ZN10CUtilTimer5startEv(%class.CUtilTimer* nonnull %t)
  br i1 %cmp41291, label %for.body51.preheader, label %for.cond.cleanup50

for.body51.preheader:                             ; preds = %sw.bb45
  br label %for.body51

for.cond.cleanup50:                               ; preds = %for.body51, %sw.bb45
  call void @_ZN10CUtilTimer4stopEv(%class.CUtilTimer* nonnull %t)
  br label %sw.epilog

for.body51:                                       ; preds = %for.body51.preheader, %for.body51
  %i46.0290 = phi i32 [ %inc54, %for.body51 ], [ 0, %for.body51.preheader ]
  %mul52 = shl nsw i32 %i46.0290, 6
  call void @_Z20process_image_serialP3rgbS0_i(%struct.rgb* %9, %struct.rgb* %14, i32 %mul52)
  %inc54 = add nuw nsw i32 %i46.0290, 1
  %exitcond = icmp eq i32 %inc54, %div
  br i1 %exitcond, label %for.cond.cleanup50, label %for.body51

sw.bb56:                                          ; preds = %for.body
  call void @_ZN10CUtilTimer5startEv(%class.CUtilTimer* nonnull %t)
  br i1 %cmp41291, label %pfor.detach.preheader, label %pfor.cond.cleanup

pfor.detach.preheader:                            ; preds = %sw.bb56
  br label %pfor.detach

pfor.cond.cleanup:                                ; preds = %pfor.inc, %sw.bb56
  sync within %syncreg, label %sync.continue

pfor.detach:                                      ; preds = %pfor.detach.preheader, %pfor.inc
  %__begin.0288 = phi i32 [ %inc65, %pfor.inc ], [ 0, %pfor.detach.preheader ]
  detach within %syncreg, label %pfor.body, label %pfor.inc unwind label %lpad66.loopexit

pfor.body:                                        ; preds = %pfor.detach
  %mul64 = shl nsw i32 %__begin.0288, 6
  invoke void @_Z20process_image_serialP3rgbS0_i(%struct.rgb* %9, %struct.rgb* %14, i32 %mul64)
          to label %pfor.preattach unwind label %lpad

pfor.preattach:                                   ; preds = %pfor.body
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.detach, %pfor.preattach
  %inc65 = add nuw nsw i32 %__begin.0288, 1
  %cmp60 = icmp slt i32 %inc65, %div
  br i1 %cmp60, label %pfor.detach, label %pfor.cond.cleanup, !llvm.loop !68

lpad:                                             ; preds = %pfor.body
  %15 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg, { i8*, i32 } %15)
          to label %det.rethrow.unreachable unwind label %lpad66.loopexit.split-lp

det.rethrow.unreachable:                          ; preds = %lpad
  unreachable

lpad66.loopexit:                                  ; preds = %pfor.detach
  %lpad.loopexit = landingpad { i8*, i32 }
          cleanup
  br label %lpad66

lpad66.loopexit.split-lp:                         ; preds = %lpad
  %lpad.loopexit.split-lp = landingpad { i8*, i32 }
          cleanup
  br label %lpad66

lpad66:                                           ; preds = %lpad66.loopexit.split-lp, %lpad66.loopexit
  %lpad.phi = phi { i8*, i32 } [ %lpad.loopexit, %lpad66.loopexit ], [ %lpad.loopexit.split-lp, %lpad66.loopexit.split-lp ]
  %16 = extractvalue { i8*, i32 } %lpad.phi, 0
  %17 = extractvalue { i8*, i32 } %lpad.phi, 1
  sync within %syncreg, label %ehcleanup

sync.continue:                                    ; preds = %pfor.cond.cleanup
  call void @_ZN10CUtilTimer4stopEv(%class.CUtilTimer* nonnull %t)
  br label %sw.epilog

sw.bb71:                                          ; preds = %for.body
  call void @_ZN10CUtilTimer5startEv(%class.CUtilTimer* nonnull %t)
  br i1 %cmp41291, label %pfor.detach84.preheader, label %pfor.cond.cleanup83

pfor.detach84.preheader:                          ; preds = %sw.bb71
  br label %pfor.detach84

pfor.cond.cleanup83:                              ; preds = %pfor.inc96, %sw.bb71
  sync within %syncreg72, label %sync.continue105

pfor.detach84:                                    ; preds = %pfor.detach84.preheader, %pfor.inc96
  %__begin74.0286 = phi i32 [ %inc97, %pfor.inc96 ], [ 0, %pfor.detach84.preheader ]
  detach within %syncreg72, label %pfor.body89, label %pfor.inc96 unwind label %lpad98.loopexit

pfor.body89:                                      ; preds = %pfor.detach84
  %mul90 = shl nsw i32 %__begin74.0286, 6
  invoke void @_Z20process_image_serialP3rgbS0_i(%struct.rgb* %9, %struct.rgb* %14, i32 %mul90)
          to label %pfor.preattach95 unwind label %lpad91

pfor.preattach95:                                 ; preds = %pfor.body89
  reattach within %syncreg72, label %pfor.inc96

pfor.inc96:                                       ; preds = %pfor.detach84, %pfor.preattach95
  %inc97 = add nuw nsw i32 %__begin74.0286, 1
  %cmp82 = icmp slt i32 %inc97, %div
  br i1 %cmp82, label %pfor.detach84, label %pfor.cond.cleanup83, !llvm.loop !70

lpad91:                                           ; preds = %pfor.body89
  %18 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg72, { i8*, i32 } %18)
          to label %det.rethrow.unreachable104 unwind label %lpad98.loopexit.split-lp

det.rethrow.unreachable104:                       ; preds = %lpad91
  unreachable

lpad98.loopexit:                                  ; preds = %pfor.detach84
  %lpad.loopexit278 = landingpad { i8*, i32 }
          cleanup
  br label %lpad98

lpad98.loopexit.split-lp:                         ; preds = %lpad91
  %lpad.loopexit.split-lp279 = landingpad { i8*, i32 }
          cleanup
  br label %lpad98

lpad98:                                           ; preds = %lpad98.loopexit.split-lp, %lpad98.loopexit
  %lpad.phi280 = phi { i8*, i32 } [ %lpad.loopexit278, %lpad98.loopexit ], [ %lpad.loopexit.split-lp279, %lpad98.loopexit.split-lp ]
  %19 = extractvalue { i8*, i32 } %lpad.phi280, 0
  %20 = extractvalue { i8*, i32 } %lpad.phi280, 1
  sync within %syncreg72, label %ehcleanup

sync.continue105:                                 ; preds = %pfor.cond.cleanup83
  call void @_ZN10CUtilTimer4stopEv(%class.CUtilTimer* nonnull %t)
  br label %sw.epilog

sw.default:                                       ; preds = %for.body
  %call1.i269 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull getelementptr inbounds ([14 x i8], [14 x i8]* @.str.6, i64 0, i64 0), i64 13)
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.default, %sync.continue105, %sync.continue, %for.cond.cleanup50, %for.cond.cleanup42
  %call109 = call double @_ZN10CUtilTimer8get_timeEv(%class.CUtilTimer* nonnull %t)
  %add110 = fadd double %avg_time.0294, %call109
  %inc112 = add nuw nsw i32 %j.0293, 1
  %cmp39 = icmp ult i32 %inc112, 5
  br i1 %cmp39, label %for.body, label %for.cond.cleanup

ehcleanup:                                        ; preds = %lpad98, %lpad66
  %exn.slot67.0 = phi i8* [ %16, %lpad66 ], [ %19, %lpad98 ]
  %ehselector.slot68.0 = phi i32 [ %17, %lpad66 ], [ %20, %lpad98 ]
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %0) #2
  %lpad.val161 = insertvalue { i8*, i32 } undef, i8* %exn.slot67.0, 0
  %lpad.val162 = insertvalue { i8*, i32 } %lpad.val161, i32 %ehselector.slot68.0, 1
  resume { i8*, i32 } %lpad.val162

if.then116:                                       ; preds = %for.cond.cleanup
  %call1.i271 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull getelementptr inbounds ([55 x i8], [55 x i8]* @.str.8, i64 0, i64 0), i64 54)
  br label %cleanup140

if.end118:                                        ; preds = %for.cond.cleanup
  %call119 = call i64 @fwrite(i8* nonnull %call2, i64 1, i64 54, %struct._IO_FILE* nonnull %call114)
  %cmp120 = icmp eq i64 %call119, 0
  br i1 %cmp120, label %if.then121, label %if.end123

if.then121:                                       ; preds = %if.end118
  %call1.i273 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull getelementptr inbounds ([77 x i8], [77 x i8]* @.str.9, i64 0, i64 0), i64 76)
  br label %cleanup140

if.end123:                                        ; preds = %if.end118
  %21 = load i32, i32* %10, align 1, !tbaa !67
  %conv126 = zext i32 %21 to i64
  %call128 = call i32 @fseek(%struct._IO_FILE* nonnull %call114, i64 %conv126, i32 0)
  %call130 = call i64 @fwrite(i8* nonnull %retval.0.i265, i64 3, i64 %conv16, %struct._IO_FILE* nonnull %call114)
  %cmp131 = icmp eq i64 %call130, 0
  br i1 %cmp131, label %if.then132, label %if.end134

if.then132:                                       ; preds = %if.end123
  %call1.i275 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* nonnull getelementptr inbounds ([77 x i8], [77 x i8]* @.str.9, i64 0, i64 0), i64 76)
  br label %cleanup140

if.end134:                                        ; preds = %if.end123
  %call.i = call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* nonnull @_ZSt4cout, double %add110)
  %call1.i277 = call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) %call.i, i8* nonnull getelementptr inbounds ([2 x i8], [2 x i8]* @.str.10, i64 0, i64 0), i64 1)
  %call137 = call i32 @fclose(%struct._IO_FILE* nonnull %call)
  %call138 = call i32 @fclose(%struct._IO_FILE* nonnull %call114)
  call void @free(i8* nonnull %call2) #2
  call void @free(i8* %retval.0.i) #2
  call void @free(i8* %retval.0.i265) #2
  br label %cleanup140

cleanup140:                                       ; preds = %if.then20, %if.then29, %if.then36, %if.then116, %if.then121, %if.then132, %if.end134, %if.then13, %if.then9, %if.then4, %if.then
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %0) #2
  ret i32 0
}

; CHECK: <phi sp entry>%for.body
; CHECK-DAG: %sw.default<sp exit>
; CHECK-DAG: %for.cond.cleanup50<sp exit>
; CHECK-DAG: %for.cond.cleanup42<sp exit>
; CHECK: <phi sp entry>%sw.epilog

; Verify that all of the above checks pass before the subtasks are
; printed.

; CHECK: task at depth 0
; CHECK: task at depth 1
; CHECK: task at depth 1

; Function Attrs: nounwind
declare noalias %struct._IO_FILE* @fopen(i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #1

; Function Attrs: nounwind
declare noalias i8* @malloc(i64) local_unnamed_addr #1

; Function Attrs: nounwind
declare i64 @fread(i8* nocapture, i64, i64, %struct._IO_FILE* nocapture) local_unnamed_addr #1

; Function Attrs: nounwind
declare i32 @fseek(%struct._IO_FILE* nocapture, i64, i32) local_unnamed_addr #1

declare void @_ZN10CUtilTimer5startEv(%class.CUtilTimer*) local_unnamed_addr #0

declare void @_ZN10CUtilTimer4stopEv(%class.CUtilTimer*) local_unnamed_addr #0

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #4

; Function Attrs: argmemonly
declare void @llvm.detached.rethrow.sl_p0i8i32s(token, { i8*, i32 }) #6

declare double @_ZN10CUtilTimer8get_timeEv(%class.CUtilTimer*) local_unnamed_addr #0

; Function Attrs: nounwind
declare i64 @fwrite(i8* nocapture, i64, i64, %struct._IO_FILE* nocapture) local_unnamed_addr #1

; Function Attrs: nounwind
declare i32 @fclose(%struct._IO_FILE* nocapture) local_unnamed_addr #1

; Function Attrs: nounwind
declare void @free(i8* nocapture) local_unnamed_addr #1

; Function Attrs: nounwind
declare float @sqrtf(float) local_unnamed_addr #1

; Function Attrs: nounwind readnone speculatable
declare float @llvm.floor.f32(float) #8

; Function Attrs: nounwind
declare i32 @posix_memalign(i8**, i64, i64) local_unnamed_addr #1

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* dereferenceable(272), i8*, i64) local_unnamed_addr #0

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"*, double) local_unnamed_addr #0

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #4

; Function Attrs: nounwind readnone speculatable
declare <4 x float> @llvm.floor.v4f32(<4 x float>) #8

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { argmemonly nounwind }
attributes #5 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { argmemonly }
attributes #7 = { norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { nounwind readnone speculatable }

!2 = !{!3, !7, i64 8}
!3 = !{!"_ZTS13matrix_serial", !4, i64 0, !7, i64 8}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!"int", !5, i64 0}
!8 = !{!7, !7, i64 0}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.isvectorized", i32 1}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.unroll.disable"}
!13 = !{!3, !4, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"float", !5, i64 0}
!16 = distinct !{!16, !17, !10}
!17 = !{!"llvm.loop.unroll.runtime.disable"}
!18 = distinct !{!18, !10}
!19 = distinct !{!19, !12}
!20 = distinct !{!20, !17, !10}
!21 = !{!22, !5, i64 2}
!22 = !{!"_ZTS3rgb", !5, i64 0, !5, i64 1, !5, i64 2}
!23 = !{!24}
!24 = distinct !{!24, !25}
!25 = distinct !{!25, !"LVerDomain"}
!26 = !{!27}
!27 = distinct !{!27, !25}
!28 = !{!29}
!29 = distinct !{!29, !30}
!30 = distinct !{!30, !"LVerDomain"}
!31 = !{!32}
!32 = distinct !{!32, !30}
!33 = distinct !{!33, !10}
!34 = distinct !{!34, !10}
!35 = !{!22, !5, i64 0}
!36 = !{!37}
!37 = distinct !{!37, !38}
!38 = distinct !{!38, !"LVerDomain"}
!39 = !{!40}
!40 = distinct !{!40, !38}
!41 = !{!42}
!42 = distinct !{!42, !43}
!43 = distinct !{!43, !"LVerDomain"}
!44 = !{!45}
!45 = distinct !{!45, !43}
!46 = distinct !{!46, !10}
!47 = distinct !{!47, !10}
!48 = !{!22, !5, i64 1}
!49 = !{!50}
!50 = distinct !{!50, !51}
!51 = distinct !{!51, !"LVerDomain"}
!52 = !{!53}
!53 = distinct !{!53, !51}
!54 = !{!55}
!55 = distinct !{!55, !56}
!56 = distinct !{!56, !"LVerDomain"}
!57 = !{!58}
!58 = distinct !{!58, !56}
!59 = distinct !{!59, !10}
!60 = distinct !{!60, !10}
!61 = !{!62, !64, i64 28}
!62 = !{!"_ZTS13bitmap_header", !63, i64 0, !7, i64 14, !7, i64 18, !7, i64 22, !64, i64 26, !64, i64 28, !7, i64 30, !7, i64 34, !7, i64 38, !7, i64 42, !7, i64 46, !7, i64 50}
!63 = !{!"_ZTS11file_header", !5, i64 0, !7, i64 2, !64, i64 6, !64, i64 8, !7, i64 10}
!64 = !{!"short", !5, i64 0}
!65 = !{!62, !7, i64 18}
!66 = !{!62, !7, i64 22}
!67 = !{!62, !7, i64 10}
!68 = distinct !{!68, !69}
!69 = !{!"tapir.loop.spawn.strategy", i32 1}
!70 = distinct !{!70, !69}
!71 = !{!4, !4, i64 0}
