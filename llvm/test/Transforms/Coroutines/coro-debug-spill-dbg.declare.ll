; Test spilling a temp generates dbg.declare in resume/destroy/cleanup functions.
;
; RUN: opt < %s -passes='cgscc(coro-split)' -S | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators < %s -passes='cgscc(coro-split)' -S | FileCheck %s
;
; The test case simulates a coroutine method in a class.
;
; class Container {
;  public:
;    Container() : field(12) {}
;    Task foo() {
;      co_await std::suspend_always{};
;      auto *copy = this;
;      co_return;
;    }
;    int field;
; };
;
; We want to make sure that the "this" pointer is accessable in debugger before and after the suspension point.
;
; CHECK: define internal fastcc void @foo.resume(ptr noundef nonnull align 8 dereferenceable(32) %[[HDL:.*]])
; CHECK-NEXT: entry.resume:
; CHECK-NEXT:   %[[HDL]].debug = alloca ptr, align 8
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata ptr %[[HDL]].debug, metadata ![[THIS_RESUME:[0-9]+]], metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 24))
;
; CHECK: define internal fastcc void @foo.destroy(ptr noundef nonnull align 8 dereferenceable(32) %[[HDL]])
; CHECK-NEXT: entry.destroy:
; CHECK-NEXT:   %[[HDL]].debug = alloca ptr, align 8
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata ptr %[[HDL]].debug, metadata ![[THIS_DESTROY:[0-9]+]], metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 24))
;
; CHECK: define internal fastcc void @foo.cleanup(ptr noundef nonnull align 8 dereferenceable(32) %[[HDL]])
; CHECK-NEXT: entry.cleanup:
; CHECK-NEXT:   %[[HDL]].debug = alloca ptr, align 8
; CHECK-NEXT:   call void @llvm.dbg.declare(metadata ptr %[[HDL]].debug, metadata ![[THIS_CLEANUP:[0-9]+]], metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 24))
;
; CHECK: ![[THIS_RESUME]] = !DILocalVariable(name: "this"
; CHECK: ![[THIS_DESTROY]] = !DILocalVariable(name: "this"
; CHECK: ![[THIS_CLEANUP]] = !DILocalVariable(name: "this"

; Function Attrs: presplitcoroutine
define ptr @foo(ptr noundef nonnull align 1 dereferenceable(1) %this) #0 !dbg !11 {
entry:
  %this.addr = alloca ptr, align 8
  %__promise = alloca i8, align 1
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !20, metadata !DIExpression()), !dbg !22
  %this1 = load ptr, ptr %this.addr, align 8
  %0 = bitcast ptr %__promise to ptr
  %id = call token @llvm.coro.id(i32 16, ptr %0, ptr null, ptr null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin

dyn.alloc:                                        ; preds = %entry
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  br label %coro.begin

coro.begin:                                       ; preds = %dyn.alloc, %entry
  %phi = phi ptr [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %phi)
  call void @llvm.dbg.declare(metadata ptr %__promise, metadata !23, metadata !DIExpression()), !dbg !22
  %1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %1, label %suspend [
    i8 0, label %resume
    i8 1, label %cleanup
  ]

resume:                                           ; preds = %coro.begin
  call void @bar(ptr %this1)
  br label %cleanup

cleanup:                                          ; preds = %resume, %coro.begin
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:                                          ; preds = %cleanup, %coro.begin
  %2 = call i1 @llvm.coro.end(ptr %hdl, i1 false, token none)
  ret ptr %hdl
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind memory(argmem: read)
declare ptr @llvm.coro.free(token, ptr nocapture readonly) #2

; Function Attrs: nounwind memory(none)
declare i32 @llvm.coro.size.i32() #3

; Function Attrs: nounwind
declare i8 @llvm.coro.suspend(token, i1) #4

declare void @llvm.coro.resume(ptr)

declare void @llvm.coro.destroy(ptr)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr) #5

; Function Attrs: nounwind
declare i1 @llvm.coro.alloc(token) #4

; Function Attrs: nounwind
declare ptr @llvm.coro.begin(token, ptr writeonly) #4

; Function Attrs: nounwind
declare i1 @llvm.coro.end(ptr, i1, token) #4

declare noalias ptr @malloc(i32)

declare void @free(ptr)

declare void @bar(ptr)

attributes #0 = { presplitcoroutine }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nounwind memory(argmem: read) }
attributes #3 = { nounwind memory(none) }
attributes #4 = { nounwind }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }

!llvm.dbg.cu = !{!0}
!llvm.linker.options = !{}
!llvm.module.flags = !{!3, !4, !5, !6, !7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 17.0.20210610", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: ".")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 8, !"PIC Level", i32 2}
!7 = !{i32 7, !"PIE Level", i32 2}
!8 = !{i32 7, !"uwtable", i32 2}
!9 = !{i32 7, !"frame-pointer", i32 2}
!10 = !{!"clang version 17.0.20210610"}
!11 = distinct !DISubprogram(name: "foo", linkageName: "_ZN9Container3fooEv", scope: !1, file: !1, line: 20, type: !12, scopeLine: 20, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !13, retainedNodes: !2)
!12 = !DISubroutineType(types: !2)
!13 = !DISubprogram(name: "foo", linkageName: "_ZN9Container3fooEv", scope: !14, file: !1, line: 20, type: !12, scopeLine: 20, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!14 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "Container", file: !1, line: 17, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !15, identifier: "_ZTS9Container")
!15 = !{!16, !13}
!16 = !DISubprogram(name: "Container", scope: !14, file: !1, line: 19, type: !17, scopeLine: 19, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!20 = !DILocalVariable(name: "this", arg: 1, scope: !11, type: !21, flags: DIFlagArtificial | DIFlagObjectPointer)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!22 = !DILocation(line: 0, scope: !11)
!23 = !DILocalVariable(name: "__promise", scope: !11, type: !24, flags: DIFlagArtificial)
!24 = !DIDerivedType(tag: DW_TAG_typedef, name: "promise_type", scope: !11, file: !1, line: 40, baseType: !25)
!25 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "promise_type", scope: !11, file: !1, line: 6, size: 8, flags: DIFlagTypePassByValue, elements: !2, identifier: "_ZTSN4Task12promise_typeE")
