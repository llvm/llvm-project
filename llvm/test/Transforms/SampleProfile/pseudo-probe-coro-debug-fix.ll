; REQUIRES: x86-registered-target
; RUN: opt < %s -passes='pseudo-probe,cgscc(coro-split),coro-cleanup,always-inline' -mtriple=x86_64 -pass-remarks=inline -S -o %t.ll
; RUN: llc -mtriple=x86_64 -stop-after=pseudo-probe-inserter < %t.ll --filetype=asm -o - | FileCheck -check-prefix=MIR %s

; Making sure PseudoProbeInserter is not generating GUID for `_Z3foov.resume`
; MIR-NOT: PSEUDO_PROBE 4448042984153125393,
; Checking the same for `_Z3foov.cleanup`
; MIR-NOT: PSEUDO_PROBE 3532999944647676065,
; Checking the same for `_Z3foov.destroy`
; MIR-NOT: PSEUDO_PROBE -7235034626494519075,

; RUN: llc -mtriple=x86_64 < %t.ll --filetype=obj -o %t.obj
; RUN: obj2yaml %t.obj | FileCheck -check-prefix=OBJ --match-full-lines %s
; OBJ:       - Name:            '.pseudo_probe (1)'{{$}}
; OBJ-NEXT:    Type:            SHT_PROGBITS{{$}}
; OBJ-NEXT:    Flags:           [ SHF_LINK_ORDER ]{{$}}
; OBJ-NEXT:    Link:            .text{{$}}
; OBJ-NEXT:    AddressAlign:    0x1{{$}}

; Making sure `_Z3foov.resume` is not a top-level function except when its a sentinel probe (i.e. 20{GUID})
; OBJ-NOT:     Content:         11921D00879FBA3D{{[0-9A-F]+$}}
; OBJ-NOT:                      {{([013-9A-F][0-9A-F])|(2[1-9A-F])}}11921D00879FBA3D{{[0-9A-F]+$}}
; Checking the same for `_Z3foov.cleanup`
; OBJ-NOT:     Content:         A1103324BBBC0731{{[0-9A-F]+$}}
; OBJ-NOT:                      {{([013-9A-F][0-9A-F])|(2[1-9A-F])}}A1103324BBBC0731{{[0-9A-F]+$}}
; Checking the same for `_Z3foov.destroy`
; OBJ-NOT:     Content:         DDA8240E57FE979B{{[0-9A-F]+$}}
; OBJ-NOT:                      {{([013-9A-F][0-9A-F])|(2[1-9A-F])}}DDA8240E57FE979B{{[0-9A-F]+$}}

; OBJ:       - Name:            .pseudo_probe{{$}}
; OBJ-NEXT:    Type:            SHT_PROGBITS{{$}}
; OBJ-NEXT:    Flags:           [ SHF_LINK_ORDER, SHF_GROUP ]{{$}}
; OBJ-NEXT:    Link:            .text.__clang_call_terminate{{$}}


; Reduced from original source code:
;   clang++ -S -g -O0 SampleProfile/Inputs/pseudo-probe-coro-debug-fix.cpp -emit-llvm -Xclang -disable-llvm-passes -std=c++20 -o SampleProfile/pseudo-probe-coro-debug-fix.ll
;
; #include <coroutine>
; struct co_sleep {
;   co_sleep(int n) : delay{n} {}
;   constexpr bool await_ready() const noexcept { return false; }
;   void await_suspend(std::coroutine_handle<> h) const noexcept {}
;   void await_resume() const noexcept {}
;   int delay;
; };
; struct Task {
;   struct promise_type {
;     promise_type() = default;
;     Task get_return_object() { return {}; }
;     std::suspend_never initial_suspend() { return {}; }
;     std::suspend_always final_suspend() noexcept { return {}; }
;     void unhandled_exception() {}
;   };
; };
; Task foo() noexcept {
;   co_await co_sleep{10};
; }
; 
; int main() {
;   foo();
; }

; ModuleID = 'pseudo-probe-coro-debug-fix.cpp'
source_filename = "pseudo-probe-coro-debug-fix.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

%"struct.Task::promise_type" = type { i8 }
%struct.Task = type { i8 }
%"struct.std::__n4861::suspend_never" = type { i8 }
%struct.co_sleep = type { i32 }
%"struct.std::__n4861::suspend_always" = type { i8 }

$__clang_call_terminate = comdat any

; Function Attrs: mustprogress noinline nounwind optnone presplitcoroutine uwtable
define dso_local void @_Z3foov() #0 personality ptr @__gxx_personality_v0 !dbg !30 {
entry:
  %__promise = alloca %"struct.Task::promise_type", align 1
  %undef.agg.tmp = alloca %struct.Task, align 1
  %ref.tmp = alloca %"struct.std::__n4861::suspend_never", align 1
  %undef.agg.tmp3 = alloca %"struct.std::__n4861::suspend_never", align 1
  %ref.tmp5 = alloca %struct.co_sleep, align 4
  %exn.slot = alloca ptr, align 8
  %ehselector.slot = alloca i32, align 4
  %ref.tmp13 = alloca %"struct.std::__n4861::suspend_always", align 1
  %undef.agg.tmp14 = alloca %"struct.std::__n4861::suspend_always", align 1
  %0 = call token @llvm.coro.id(i32 16, ptr %__promise, ptr null, ptr null), !dbg !33
  %1 = call i1 @llvm.coro.alloc(token %0), !dbg !33
  br i1 %1, label %coro.alloc, label %coro.init, !dbg !33

coro.alloc:                                       ; preds = %entry
  %2 = call i64 @llvm.coro.size.i64(), !dbg !34
  %call = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %2) #12
          to label %invoke.cont unwind label %terminate.lpad, !dbg !34

invoke.cont:                                      ; preds = %coro.alloc
  br label %coro.init, !dbg !33

coro.init:                                        ; preds = %invoke.cont, %entry
  %3 = phi ptr [ null, %entry ], [ %call, %invoke.cont ], !dbg !33
  %4 = call ptr @llvm.coro.begin(token %0, ptr %3), !dbg !33
  call void @llvm.lifetime.start.p0(ptr %__promise) #2, !dbg !34
    #dbg_declare(ptr %__promise, !35, !DIExpression(), !41)
  call void @llvm.lifetime.start.p0(ptr %ref.tmp5) #2, !dbg !42
  %5 = call token @llvm.coro.save(ptr null), !dbg !44
  call void @llvm.coro.await.suspend.void(ptr %ref.tmp5, ptr %4, ptr @_Z3foov.__await_suspend_wrapper__await) #2, !dbg !44
  %6 = call i8 @llvm.coro.suspend(token %5, i1 false), !dbg !44
  switch i8 %6, label %coro.ret [
    i8 0, label %cleanup8
    i8 1, label %await.cleanup
  ], !dbg !44

await.cleanup:                                    ; preds = %coro.init
  br label %cleanup8, !dbg !44

coro.final:                                       ; preds = %cleanup8
  call void @llvm.lifetime.start.p0(ptr %ref.tmp13) #2, !dbg !34
  %7 = call token @llvm.coro.save(ptr null), !dbg !34
  call void @llvm.coro.await.suspend.void(ptr %ref.tmp13, ptr %4, ptr @_Z3foov.__await_suspend_wrapper__final) #2, !dbg !34
  %8 = call i8 @llvm.coro.suspend(token %7, i1 true), !dbg !34
  switch i8 %8, label %coro.ret [
    i8 0, label %cleanup16
    i8 1, label %final.cleanup
  ], !dbg !34

final.cleanup:                                    ; preds = %coro.final
  br label %cleanup16, !dbg !34

cleanup8:                                         ; preds = %await.cleanup, %coro.init
  %cleanup.dest.slot.1 = phi i32 [ 0, %coro.init ], [ 2, %await.cleanup ]
  call void @llvm.lifetime.end.p0(ptr %ref.tmp5) #2, !dbg !44
  switch i32 %cleanup.dest.slot.1, label %cleanup19 [
    i32 0, label %coro.final
  ]

cleanup16:                                        ; preds = %final.cleanup, %coro.final
  %cleanup.dest.slot.2 = phi i32 [ 0, %coro.final ], [ 2, %final.cleanup ]
  call void @llvm.lifetime.end.p0(ptr %ref.tmp13) #2, !dbg !34
  switch i32 %cleanup.dest.slot.2, label %cleanup19 [
    i32 0, label %cleanup.cont18
  ]

cleanup.cont18:                                   ; preds = %cleanup16
  br label %cleanup19, !dbg !34

cleanup19:                                        ; preds = %cleanup.cont18, %cleanup16, %cleanup8
  %cleanup.dest.slot.3 = phi i32 [ %cleanup.dest.slot.1, %cleanup8 ], [ %cleanup.dest.slot.2, %cleanup16 ], [ 0, %cleanup.cont18 ], !dbg !41
  call void @llvm.lifetime.end.p0(ptr %__promise) #2, !dbg !34
  %9 = call ptr @llvm.coro.free(token %0, ptr %4), !dbg !34
  %10 = icmp ne ptr %9, null, !dbg !34
  br i1 %10, label %coro.free, label %after.coro.free, !dbg !34

coro.free:                                        ; preds = %cleanup19
  %11 = call i64 @llvm.coro.size.i64(), !dbg !34
  call void @_ZdlPvm(ptr noundef %9, i64 noundef %11) #2, !dbg !34
  br label %after.coro.free, !dbg !34

after.coro.free:                                  ; preds = %coro.free, %cleanup19
  switch i32 %cleanup.dest.slot.3, label %unreachable [
    i32 0, label %cleanup.cont22
    i32 2, label %coro.ret
  ]

cleanup.cont22:                                   ; preds = %after.coro.free
  br label %coro.ret, !dbg !34

coro.ret:                                         ; preds = %cleanup.cont22, %after.coro.free, %coro.final, %coro.init
  call void @llvm.coro.end(ptr null, i1 false, token none), !dbg !34
  ret void, !dbg !34

terminate.lpad:                                   ; preds = %coro.alloc
  %12 = landingpad { ptr, i32 }
          catch ptr null, !dbg !34
  %13 = extractvalue { ptr, i32 } %12, 0, !dbg !34
  call void @__clang_call_terminate(ptr %13) #13, !dbg !34
  unreachable, !dbg !34

unreachable:                                      ; preds = %after.coro.free
  unreachable
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare token @llvm.coro.id(i32, ptr readnone, ptr readonly captures(none), ptr) #1

; Function Attrs: nounwind
declare i1 @llvm.coro.alloc(token) #2

; Function Attrs: nobuiltin allocsize(0)
declare dso_local noundef nonnull ptr @_Znwm(i64 noundef) #3

; Function Attrs: nounwind memory(none)
declare i64 @llvm.coro.size.i64() #4

declare dso_local i32 @__gxx_personality_v0(...)

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) #5 comdat {
  %2 = call ptr @__cxa_begin_catch(ptr %0) #2
  call void @_ZSt9terminatev() #13
  unreachable
}

declare dso_local ptr @__cxa_begin_catch(ptr)

declare dso_local void @_ZSt9terminatev()

; Function Attrs: nounwind
declare ptr @llvm.coro.begin(token, ptr writeonly) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #6

; Function Attrs: nomerge nounwind
declare token @llvm.coro.save(ptr) #7

; Function Attrs: alwaysinline mustprogress
define internal void @_Z3foov.__await_suspend_wrapper__init(ptr noundef nonnull %0, ptr noundef %1) #8 !dbg !45 {
entry:
  call void @_ZSt9terminatev() #13
  ret void, !dbg !47
}

declare void @llvm.coro.await.suspend.void(ptr, ptr, ptr)

; Function Attrs: nounwind
declare i8 @llvm.coro.suspend(token, i1) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #6

; Function Attrs: alwaysinline mustprogress
define internal void @_Z3foov.__await_suspend_wrapper__await(ptr noundef nonnull %0, ptr noundef %1) #8 !dbg !48 {
entry:
  call void @_ZSt9terminatev() #13
  ret void, !dbg !49
}

declare dso_local void @__cxa_end_catch()

; Function Attrs: alwaysinline mustprogress
define internal void @_Z3foov.__await_suspend_wrapper__final(ptr noundef nonnull %0, ptr noundef %1) #8 !dbg !50 {
entry:
  call void @_ZSt9terminatev() #13
  ret void, !dbg !51
}

; Function Attrs: nobuiltin nounwind
declare dso_local void @_ZdlPvm(ptr noundef, i64 noundef) #9

; Function Attrs: nounwind memory(argmem: read)
declare ptr @llvm.coro.free(token, ptr readonly captures(none)) #10

; Function Attrs: nounwind
declare void @llvm.coro.end(ptr, i1, token) #2

; Function Attrs: mustprogress noinline norecurse nounwind optnone uwtable
define dso_local noundef i32 @main() #11 !dbg !52 {
entry:
  %undef.agg.tmp = alloca %struct.Task, align 1
  call void @_Z3foov() #2, !dbg !55
  ret i32 0, !dbg !56
}

attributes #0 = { mustprogress noinline nounwind optnone presplitcoroutine uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #2 = { nounwind }
attributes #3 = { nobuiltin allocsize(0) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { nounwind memory(none) }
attributes #5 = { noinline noreturn nounwind uwtable "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { nomerge nounwind }
attributes #8 = { alwaysinline mustprogress "min-legal-vector-width"="0" }
attributes #9 = { nobuiltin nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #10 = { nounwind memory(argmem: read) }
attributes #11 = { mustprogress noinline norecurse nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #12 = { allocsize(0) }
attributes #13 = { noreturn nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!24, !25, !26, !27, !28}
!llvm.ident = !{!29}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 22.0.0git (https://github.com/llvm/llvm-project.git fe218aab737d55dfd67f4f84118744003f45958e)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "pseudo-probe-coro-debug-fix.cpp", directory: "/tmp", checksumkind: CSK_MD5, checksum: "601552fedf6b88dcf699ea9d066126eb")
!2 = !{!3, !6, !15, !20}
!3 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "promise_type", scope: !4, file: !1, line: 10, size: 8, flags: DIFlagTypePassByValue, elements: !5, identifier: "_ZTSN4Task12promise_typeE")
!4 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Task", file: !1, line: 9, size: 8, flags: DIFlagTypePassByValue, elements: !5, identifier: "_ZTS4Task")
!5 = !{}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "coroutine_handle<void>", scope: !8, file: !7, line: 87, size: 64, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !10, templateParams: !13, identifier: "_ZTSNSt7__n486116coroutine_handleIvEE")
!7 = !DIFile(filename: "/usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/coroutine", directory: "")
!8 = !DINamespace(name: "__n4861", scope: !9, exportSymbols: true)
!9 = !DINamespace(name: "std", scope: null)
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "_M_fr_ptr", scope: !6, file: !7, line: 131, baseType: !12, size: 64, flags: DIFlagProtected)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!13 = !{!14}
!14 = !DITemplateTypeParameter(name: "_Promise", type: null, defaulted: true)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "coroutine_handle<Task::promise_type>", scope: !8, file: !7, line: 182, size: 64, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !16, templateParams: !18, identifier: "_ZTSNSt7__n486116coroutine_handleIN4Task12promise_typeEEE")
!16 = !{!17}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "_M_fr_ptr", scope: !15, file: !7, line: 244, baseType: !12, size: 64, flags: DIFlagPrivate)
!18 = !{!19}
!19 = !DITemplateTypeParameter(name: "_Promise", type: !3)
!20 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "co_sleep", file: !1, line: 2, size: 32, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !21, identifier: "_ZTS8co_sleep")
!21 = !{!22}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "delay", scope: !20, file: !1, line: 7, baseType: !23, size: 32)
!23 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!24 = !{i32 7, !"Dwarf Version", i32 5}
!25 = !{i32 2, !"Debug Info Version", i32 3}
!26 = !{i32 1, !"wchar_size", i32 4}
!27 = !{i32 7, !"uwtable", i32 2}
!28 = !{i32 7, !"frame-pointer", i32 2}
!29 = !{!"clang version 22.0.0git (https://github.com/llvm/llvm-project.git fe218aab737d55dfd67f4f84118744003f45958e)"}
!30 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 18, type: !31, scopeLine: 18, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !5)
!31 = !DISubroutineType(types: !32)
!32 = !{!4}
!33 = !DILocation(line: 18, column: 21, scope: !30)
!34 = !DILocation(line: 18, column: 6, scope: !30)
!35 = !DILocalVariable(name: "__promise", scope: !30, type: !36, flags: DIFlagArtificial)
!36 = !DIDerivedType(tag: DW_TAG_typedef, name: "promise_type", scope: !37, file: !7, line: 75, baseType: !3)
!37 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__coroutine_traits_impl<Task, void>", scope: !8, file: !7, line: 72, size: 8, flags: DIFlagTypePassByValue, elements: !5, templateParams: !38, identifier: "_ZTSNSt7__n486123__coroutine_traits_implI4TaskvEE")
!38 = !{!39, !40}
!39 = !DITemplateTypeParameter(name: "_Result", type: !4)
!40 = !DITemplateTypeParameter(type: null, defaulted: true)
!41 = !DILocation(line: 0, scope: !30)
!42 = !DILocation(line: 19, column: 12, scope: !43)
!43 = distinct !DILexicalBlock(scope: !30, file: !1, line: 18, column: 21)
!44 = !DILocation(line: 19, column: 3, scope: !43)
!45 = distinct !DISubprogram(linkageName: "_Z3foov.__await_suspend_wrapper__init", scope: !1, file: !1, type: !46, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !5)
!46 = !DISubroutineType(types: !5)
!47 = !DILocation(line: 18, column: 6, scope: !45)
!48 = distinct !DISubprogram(linkageName: "_Z3foov.__await_suspend_wrapper__await", scope: !1, file: !1, type: !46, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !5)
!49 = !DILocation(line: 19, column: 12, scope: !48)
!50 = distinct !DISubprogram(linkageName: "_Z3foov.__await_suspend_wrapper__final", scope: !1, file: !1, type: !46, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !5)
!51 = !DILocation(line: 18, column: 6, scope: !50)
!52 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 22, type: !53, scopeLine: 22, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!53 = !DISubroutineType(types: !54)
!54 = !{!23}
!55 = !DILocation(line: 23, column: 3, scope: !52)
!56 = !DILocation(line: 24, column: 1, scope: !52)
