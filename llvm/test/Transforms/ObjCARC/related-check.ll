; Make sure this succeeds without hitting an assertion and the output is deterministic
; RUN: mkdir -p %t
; RUN: opt -passes=objc-arc %s -S -o %t/out1.ll
; RUN: opt -passes=objc-arc %s -S -o %t/out2.ll
; RUN: diff -u %t/out1.ll %t/out2.ll

%0 = type opaque
%struct._class_t = type { ptr, ptr, ptr, ptr, ptr }
%struct._objc_cache = type opaque
%struct._class_ro_t = type { i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct.__method_list_t = type { i32, i32, [0 x %struct._objc_method] }
%struct._objc_method = type { ptr, ptr, ptr }
%struct._objc_protocol_list = type { i64, [0 x ptr] }
%struct._protocol_t = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr }
%struct._ivar_list_t = type { i32, i32, [0 x %struct._ivar_t] }
%struct._ivar_t = type { ptr, ptr, ptr, i32, i32 }
%struct._prop_list_t = type { i32, i32, [0 x %struct._prop_t] }
%struct._prop_t = type { ptr, ptr }
%struct.__NSConstantString_tag = type { ptr, i32, ptr, i64 }

@.str = private unnamed_addr constant [8 x i8] c"%s: %s\0A\00", align 1
@OBJC_METH_VAR_NAME_ = private unnamed_addr constant [25 x i8] c"fileSystemRepresentation\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_SELECTOR_REFERENCES_ = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8
@"OBJC_CLASS_$_NSString" = external global %struct._class_t
@"OBJC_CLASSLIST_REFERENCES_$_" = internal global ptr @"OBJC_CLASS_$_NSString", section "__DATA,__objc_classrefs,regular,no_dead_strip", align 8
@__CFConstantStringClassReference = external global [0 x i32]
@.str.1 = private unnamed_addr constant [3 x i8] c"%@\00", section "__TEXT,__cstring,cstring_literals", align 1
@_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @.str.1, i64 2 }, section "__DATA,__cfstring", align 8 #0
@OBJC_METH_VAR_NAME_.2 = private unnamed_addr constant [18 x i8] c"stringWithFormat:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_SELECTOR_REFERENCES_.3 = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_.2, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8
@global1 = external local_unnamed_addr constant ptr, align 8
@llvm.compiler.used = appending global [5 x ptr] [ptr @OBJC_METH_VAR_NAME_, ptr @OBJC_SELECTOR_REFERENCES_, ptr @"OBJC_CLASSLIST_REFERENCES_$_", ptr @OBJC_METH_VAR_NAME_.2, ptr @OBJC_SELECTOR_REFERENCES_.3], section "llvm.metadata"

; Function Attrs: optsize ssp uwtable(sync)
define i32 @main(i32 noundef %argc, ptr nocapture noundef readnone %argv) local_unnamed_addr #1 {
entry:
  %persistent = alloca i32, align 4
  %personalized = alloca i32, align 4
  %cmp31 = icmp sgt i32 %argc, 1
  br i1 %cmp31, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %0 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_, align 8
  %1 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.3, align 8
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %if.end19
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret i32 0

for.body:                                         ; preds = %for.body.lr.ph, %if.end19
  %i.032 = phi i32 [ 1, %for.body.lr.ph ], [ %inc, %if.end19 ]
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %persistent) #4
  store i32 0, ptr %persistent, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %personalized) #4
  store i32 0, ptr %personalized, align 4
  %call = call zeroext i1 @lookupType(ptr noundef nonnull %persistent, ptr noundef nonnull %personalized) #8, !clang.arc.no_objc_arc_exceptions !15
  br i1 %call, label %if.then, label %if.end19

if.then:                                          ; preds = %for.body
  %2 = load i32, ptr %persistent, align 4
  %cmp1.not = icmp eq i32 %2, 0
  br i1 %cmp1.not, label %if.end, label %if.then2

if.then2:                                         ; preds = %if.then
  %call34 = call ptr @getnsstr() #8 [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ], !clang.arc.no_objc_arc_exceptions !15
  call void (...) @llvm.objc.clang.arc.noop.use(ptr %call34) #4
  call void @llvm.objc.release(ptr null) #4, !clang.imprecise_release !15
  %call56 = call ptr @getnsstr() #8 [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ], !clang.arc.no_objc_arc_exceptions !15
  call void (...) @llvm.objc.clang.arc.noop.use(ptr %call56) #4
  call void @llvm.objc.release(ptr null) #4, !clang.imprecise_release !15
  br label %if.end

if.end:                                           ; preds = %if.then2, %if.then
  %path.0 = phi ptr [ %call34, %if.then2 ], [ null, %if.then ]
  %name.0 = phi ptr [ %call56, %if.then2 ], [ null, %if.then ]
  %3 = load i32, ptr %personalized, align 4
  %cmp7.not = icmp eq i32 %3, 0
  br i1 %cmp7.not, label %if.end11, label %if.then8

if.then8:                                         ; preds = %if.end
  %call910 = call ptr @getnsstr() #8 [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ], !clang.arc.no_objc_arc_exceptions !15
  call void (...) @llvm.objc.clang.arc.noop.use(ptr %call910) #4
  call void @llvm.objc.release(ptr %path.0) #4, !clang.imprecise_release !15
  br label %if.end11

if.end11:                                         ; preds = %if.then8, %if.end
  %path.1 = phi ptr [ %call910, %if.then8 ], [ %path.0, %if.end ]
  %cmp12.not = icmp eq ptr %path.1, null
  br i1 %cmp12.not, label %if.else, label %if.then13

if.then13:                                        ; preds = %if.end11
  %call14 = call ptr @objc_msgSend(ptr noundef nonnull %path.1, ptr noundef %0) #8, !clang.arc.no_objc_arc_exceptions !15
  %call15 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef %call14) #8, !clang.arc.no_objc_arc_exceptions !15
  br label %if.end18

if.else:                                          ; preds = %if.end11
  %4 = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_", align 8
  %call1617 = call ptr (ptr, ptr, ptr, ...) @objc_msgSend(ptr noundef %4, ptr noundef %1, ptr noundef nonnull @_unnamed_cfstring_, ptr noundef null) #8 [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ], !clang.arc.no_objc_arc_exceptions !15
  call void (...) @llvm.objc.clang.arc.noop.use(ptr %call1617) #4
  call void @llvm.objc.release(ptr %call1617) #4, !clang.imprecise_release !15
  br label %if.end18

if.end18:                                         ; preds = %if.else, %if.then13
  %.pre-phi = phi ptr [ null, %if.else ], [ %path.1, %if.then13 ]
  call void @llvm.objc.release(ptr %name.0) #4, !clang.imprecise_release !15
  call void @llvm.objc.release(ptr %.pre-phi) #4, !clang.imprecise_release !15
  br label %if.end19

if.end19:                                         ; preds = %if.end18, %for.body
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %personalized) #4
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %persistent) #4
  %inc = add nuw nsw i32 %i.032, 1
  %exitcond.not = icmp eq i32 %inc, %argc
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #2

; Function Attrs: inaccessiblememonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.objc.clang.arc.noop.use(...) #5

declare zeroext i1 @lookupType(ptr noundef, ptr noundef) #2

declare ptr @getnsstr(...) #2

; Function Attrs: nounwind
declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr) #3

; Function Attrs: nounwind
declare void @llvm.objc.release(ptr) #3

declare i32 @printf(ptr noundef, ...) #2

; Function Attrs: nonlazybind
declare ptr @objc_msgSend(ptr, ptr, ...) #4

attributes #0 = { "objc_arc_inert" }

!15 = !{}
