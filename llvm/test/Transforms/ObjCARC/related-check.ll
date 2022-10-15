; Make sure this succeeds without hitting an assertion and the output is deterministic
; RUN: mkdir -p %t
; RUN: opt -objc-arc %s -S -o %t/out1.ll
; RUN: opt -objc-arc %s -S -o %t/out2.ll
; RUN: diff -u %t/out1.ll %t/out2.ll

%0 = type opaque
%struct._class_t = type { %struct._class_t*, %struct._class_t*, %struct._objc_cache*, i8* (i8*, i8*)**, %struct._class_ro_t* }
%struct._objc_cache = type opaque
%struct._class_ro_t = type { i32, i32, i32, i8*, i8*, %struct.__method_list_t*, %struct._objc_protocol_list*, %struct._ivar_list_t*, i8*, %struct._prop_list_t* }
%struct.__method_list_t = type { i32, i32, [0 x %struct._objc_method] }
%struct._objc_method = type { i8*, i8*, i8* }
%struct._objc_protocol_list = type { i64, [0 x %struct._protocol_t*] }
%struct._protocol_t = type { i8*, i8*, %struct._objc_protocol_list*, %struct.__method_list_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct.__method_list_t*, %struct._prop_list_t*, i32, i32, i8**, i8*, %struct._prop_list_t* }
%struct._ivar_list_t = type { i32, i32, [0 x %struct._ivar_t] }
%struct._ivar_t = type { i32*, i8*, i8*, i32, i32 }
%struct._prop_list_t = type { i32, i32, [0 x %struct._prop_t] }
%struct._prop_t = type { i8*, i8* }
%struct.__NSConstantString_tag = type { i32*, i32, i8*, i64 }

@.str = private unnamed_addr constant [8 x i8] c"%s: %s\0A\00", align 1
@OBJC_METH_VAR_NAME_ = private unnamed_addr constant [25 x i8] c"fileSystemRepresentation\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_SELECTOR_REFERENCES_ = internal externally_initialized global i8* getelementptr inbounds ([25 x i8], [25 x i8]* @OBJC_METH_VAR_NAME_, i32 0, i32 0), section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8
@"OBJC_CLASS_$_NSString" = external global %struct._class_t
@"OBJC_CLASSLIST_REFERENCES_$_" = internal global %struct._class_t* @"OBJC_CLASS_$_NSString", section "__DATA,__objc_classrefs,regular,no_dead_strip", align 8
@__CFConstantStringClassReference = external global [0 x i32]
@.str.1 = private unnamed_addr constant [3 x i8] c"%@\00", section "__TEXT,__cstring,cstring_literals", align 1
@_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { i32* getelementptr inbounds ([0 x i32], [0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 1992, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.1, i32 0, i32 0), i64 2 }, section "__DATA,__cfstring", align 8 #0
@OBJC_METH_VAR_NAME_.2 = private unnamed_addr constant [18 x i8] c"stringWithFormat:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_SELECTOR_REFERENCES_.3 = internal externally_initialized global i8* getelementptr inbounds ([18 x i8], [18 x i8]* @OBJC_METH_VAR_NAME_.2, i32 0, i32 0), section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8
@llvm.compiler.used = appending global [5 x i8*] [i8* getelementptr inbounds ([25 x i8], [25 x i8]* @OBJC_METH_VAR_NAME_, i32 0, i32 0), i8* bitcast (i8** @OBJC_SELECTOR_REFERENCES_ to i8*), i8* bitcast (%struct._class_t** @"OBJC_CLASSLIST_REFERENCES_$_" to i8*), i8* getelementptr inbounds ([18 x i8], [18 x i8]* @OBJC_METH_VAR_NAME_.2, i32 0, i32 0), i8* bitcast (i8** @OBJC_SELECTOR_REFERENCES_.3 to i8*)], section "llvm.metadata"

; Function Attrs: optsize ssp uwtable(sync)
define i32 @main(i32 noundef %argc, i8** nocapture noundef readnone %argv) local_unnamed_addr #1 {
entry:
  %persistent = alloca i32, align 4
  %personalized = alloca i32, align 4
  %cmp31 = icmp sgt i32 %argc, 1
  br i1 %cmp31, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %0 = bitcast i32* %persistent to i8*
  %1 = bitcast i32* %personalized to i8*
  %2 = load i8*, i8** @OBJC_SELECTOR_REFERENCES_, align 8
  %3 = load i8*, i8** @OBJC_SELECTOR_REFERENCES_.3, align 8
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %if.end19
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret i32 0

for.body:                                         ; preds = %for.body.lr.ph, %if.end19
  %i.032 = phi i32 [ 1, %for.body.lr.ph ], [ %inc, %if.end19 ]
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0) #4
  store i32 0, i32* %persistent, align 4
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %1) #4
  store i32 0, i32* %personalized, align 4
  %call = call zeroext i1 @lookupType(i32* noundef nonnull %persistent, i32* noundef nonnull %personalized) #8, !clang.arc.no_objc_arc_exceptions !15
  br i1 %call, label %if.then, label %if.end19

if.then:                                          ; preds = %for.body
  %4 = load i32, i32* %persistent, align 4
  %cmp1.not = icmp eq i32 %4, 0
  br i1 %cmp1.not, label %if.end, label %if.then2

if.then2:                                         ; preds = %if.then
  %call34 = call %0* bitcast (%0* (...)* @getnsstr to %0* ()*)() #8 [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ], !clang.arc.no_objc_arc_exceptions !15
  call void (...) @llvm.objc.clang.arc.noop.use(%0* %call34) #4
  call void @llvm.objc.release(i8* null) #4, !clang.imprecise_release !15
  %call56 = call %0* bitcast (%0* (...)* @getnsstr to %0* ()*)() #8 [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ], !clang.arc.no_objc_arc_exceptions !15
  call void (...) @llvm.objc.clang.arc.noop.use(%0* %call56) #4
  call void @llvm.objc.release(i8* null) #4, !clang.imprecise_release !15
  br label %if.end

if.end:                                           ; preds = %if.then2, %if.then
  %path.0 = phi %0* [ %call34, %if.then2 ], [ null, %if.then ]
  %name.0 = phi %0* [ %call56, %if.then2 ], [ null, %if.then ]
  %5 = load i32, i32* %personalized, align 4
  %cmp7.not = icmp eq i32 %5, 0
  br i1 %cmp7.not, label %if.end11, label %if.then8

if.then8:                                         ; preds = %if.end
  %call910 = call %0* bitcast (%0* (...)* @getnsstr to %0* ()*)() #8 [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ], !clang.arc.no_objc_arc_exceptions !15
  call void (...) @llvm.objc.clang.arc.noop.use(%0* %call910) #4
  %6 = bitcast %0* %path.0 to i8*
  call void @llvm.objc.release(i8* %6) #4, !clang.imprecise_release !15
  br label %if.end11

if.end11:                                         ; preds = %if.then8, %if.end
  %path.1 = phi %0* [ %call910, %if.then8 ], [ %path.0, %if.end ]
  %cmp12.not = icmp eq %0* %path.1, null
  br i1 %cmp12.not, label %if.else, label %if.then13

if.then13:                                        ; preds = %if.end11
  %7 = bitcast %0* %path.1 to i8*
  %call14 = call i8* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*)*)(i8* noundef nonnull %7, i8* noundef %2) #8, !clang.arc.no_objc_arc_exceptions !15
  %call15 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([8 x i8], [8 x i8]* @.str, i64 0, i64 0), i8* noundef %call14) #8, !clang.arc.no_objc_arc_exceptions !15
  br label %if.end18

if.else:                                          ; preds = %if.end11
  %8 = load i8*, i8** bitcast (%struct._class_t** @"OBJC_CLASSLIST_REFERENCES_$_" to i8**), align 8
  %call1617 = call i8* (i8*, i8*, %0*, ...) bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i8* (i8*, i8*, %0*, ...)*)(i8* noundef %8, i8* noundef %3, %0* noundef nonnull bitcast (%struct.__NSConstantString_tag* @_unnamed_cfstring_ to %0*), %0* noundef null) #8 [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ], !clang.arc.no_objc_arc_exceptions !15
  call void (...) @llvm.objc.clang.arc.noop.use(i8* %call1617) #4
  call void @llvm.objc.release(i8* %call1617) #4, !clang.imprecise_release !15
  br label %if.end18

if.end18:                                         ; preds = %if.else, %if.then13
  %.pre-phi = phi i8* [ null, %if.else ], [ %7, %if.then13 ]
  %9 = bitcast %0* %name.0 to i8*
  call void @llvm.objc.release(i8* %9) #4, !clang.imprecise_release !15
  call void @llvm.objc.release(i8* %.pre-phi) #4, !clang.imprecise_release !15
  br label %if.end19

if.end19:                                         ; preds = %if.end18, %for.body
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %1) #4
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0) #4
  %inc = add nuw nsw i32 %i.032, 1
  %exitcond.not = icmp eq i32 %inc, %argc
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: inaccessiblememonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.objc.clang.arc.noop.use(...) #5

declare zeroext i1 @lookupType(i32* noundef, i32* noundef) #2

declare %0* @getnsstr(...) #2

; Function Attrs: nounwind
declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*) #3

; Function Attrs: nounwind
declare void @llvm.objc.release(i8*) #3

declare i32 @printf(i8* noundef, ...) #2

; Function Attrs: nonlazybind
declare i8* @objc_msgSend(i8*, i8*, ...) #4

attributes #0 = { "objc_arc_inert" }

!15 = !{}
