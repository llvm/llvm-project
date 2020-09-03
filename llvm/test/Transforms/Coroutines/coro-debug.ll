; Tests that debug information is sane after coro-split
; RUN: opt < %s -coro-split -S | FileCheck %s
; RUN: opt < %s -passes=coro-split -S | FileCheck %s

source_filename = "simple-repro.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind
define i8* @f(i32 %x) #0 !dbg !6 {
entry:
  %x.addr = alloca i32, align 4
  %coro_hdl = alloca i8*, align 8
  store i32 %x, i32* %x.addr, align 4
  %0 = call token @llvm.coro.id(i32 0, i8* null, i8* bitcast (i8* (i32)* @f to i8*), i8* null), !dbg !16
  %1 = call i64 @llvm.coro.size.i64(), !dbg !16
  %call = call i8* @malloc(i64 %1), !dbg !16
  %2 = call i8* @llvm.coro.begin(token %0, i8* %call) #7, !dbg !16
  store i8* %2, i8** %coro_hdl, align 8, !dbg !16
  %3 = call i8 @llvm.coro.suspend(token none, i1 false), !dbg !17
  %conv = sext i8 %3 to i32, !dbg !17
  call void @coro.devirt.trigger(i8* null)
  switch i32 %conv, label %sw.default [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
  ], !dbg !17

sw.bb:                                            ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %x.addr, metadata !12, metadata !13), !dbg !14
  call void @llvm.dbg.declare(metadata i8** %coro_hdl, metadata !15, metadata !13), !dbg !16
  br label %sw.epilog, !dbg !18

sw.bb1:                                           ; preds = %entry
  br label %coro_Cleanup, !dbg !18

sw.default:                                       ; preds = %entry
  br label %coro_Suspend, !dbg !18

sw.epilog:                                        ; preds = %sw.bb
  %4 = load i32, i32* %x.addr, align 4, !dbg !20
  %add = add nsw i32 %4, 1, !dbg !21
  store i32 %add, i32* %x.addr, align 4, !dbg !22
  br label %coro_Cleanup, !dbg !23

coro_Cleanup:                                     ; preds = %sw.epilog, %sw.bb1
  %5 = load i8*, i8** %coro_hdl, align 8, !dbg !24
  %6 = call i8* @llvm.coro.free(token %0, i8* %5), !dbg !24
  call void @free(i8* %6), !dbg !24
  br label %coro_Suspend, !dbg !24

coro_Suspend:                                     ; preds = %coro_Cleanup, %sw.default
  %7 = call i1 @llvm.coro.end(i8* null, i1 false) #7, !dbg !24
  %8 = load i8*, i8** %coro_hdl, align 8, !dbg !24
  ret i8* %8, !dbg !24
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind readonly
declare token @llvm.coro.id(i32, i8* readnone, i8* nocapture readonly, i8*) #2

declare i8* @malloc(i64) #3

; Function Attrs: nounwind readnone
declare i64 @llvm.coro.size.i64() #4

; Function Attrs: nounwind
declare i8* @llvm.coro.begin(token, i8* writeonly) #5

; Function Attrs: nounwind
declare i8 @llvm.coro.suspend(token, i1) #5

declare void @free(i8*) #3

; Function Attrs: argmemonly nounwind readonly
declare i8* @llvm.coro.free(token, i8* nocapture readonly) #2

; Function Attrs: nounwind
declare i1 @llvm.coro.end(i8*, i1) #5

; Function Attrs: alwaysinline
define private void @coro.devirt.trigger(i8*) #6 {
entry:
  ret void
}

; Function Attrs: argmemonly nounwind readonly
declare i8* @llvm.coro.subfn.addr(i8* nocapture readonly, i8) #2

attributes #0 = { noinline nounwind "coroutine.presplit"="1" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind readonly }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind readnone }
attributes #5 = { nounwind }
attributes #6 = { alwaysinline }
attributes #7 = { noduplicate }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 5.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "C:\5CGitHub\5Cllvm\5Cbuild\5CDebug\5Cbin")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 5.0.0"}
!6 = distinct !DISubprogram(name: "f", linkageName: "flink", scope: !7, file: !7, line: 55, type: !8, isLocal: false, isDefinition: true, scopeLine: 55, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "simple-repro.c", directory: "C:\5CGitHub\5Cllvm\5Cbuild\5CDebug\5Cbin")
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocalVariable(name: "x", arg: 1, scope: !6, file: !7, line: 55, type: !11)
!13 = !DIExpression()
!14 = !DILocation(line: 55, column: 13, scope: !6)
!15 = !DILocalVariable(name: "coro_hdl", scope: !6, file: !7, line: 56, type: !10)
!16 = !DILocation(line: 56, column: 3, scope: !6)
!17 = !DILocation(line: 58, column: 5, scope: !6)
!18 = !DILocation(line: 58, column: 5, scope: !19)
!19 = distinct !DILexicalBlock(scope: !6, file: !7, line: 58, column: 5)
!20 = !DILocation(line: 59, column: 9, scope: !6)
!21 = !DILocation(line: 59, column: 10, scope: !6)
!22 = !DILocation(line: 59, column: 7, scope: !6)
!23 = !DILocation(line: 59, column: 5, scope: !6)
!24 = !DILocation(line: 62, column: 3, scope: !6)

; CHECK: define i8* @f(i32 %x) #0 !dbg ![[ORIG:[0-9]+]]
; CHECK: define internal fastcc void @f.resume(%f.Frame* noalias nonnull align 8 dereferenceable(32) %FramePtr) #0 !dbg ![[RESUME:[0-9]+]]
; CHECK: entry.resume:
; CHECK-NEXT: call void @coro.devirt.trigger(i8* null)
; CHECK-NEXT: call void @llvm.dbg.declare(metadata i32* %x.addr.reload.addr, metadata ![[RESUME_VAR:[0-9]+]]
; CHECK: define internal fastcc void @f.destroy(%f.Frame* noalias nonnull align 8 dereferenceable(32) %FramePtr) #0 !dbg ![[DESTROY:[0-9]+]]
; CHECK: define internal fastcc void @f.cleanup(%f.Frame* noalias nonnull align 8 dereferenceable(32) %FramePtr) #0 !dbg ![[CLEANUP:[0-9]+]]

; CHECK: ![[ORIG]] = distinct !DISubprogram(name: "f", linkageName: "flink"

; CHECK: ![[RESUME]] = distinct !DISubprogram(name: "f", linkageName: "flink"
; CHECK: ![[RESUME_VAR]] = !DILocalVariable(name: "x", arg: 1, scope: ![[RESUME]]

; CHECK: ![[DESTROY]] = distinct !DISubprogram(name: "f", linkageName: "flink"

; CHECK: ![[CLEANUP]] = distinct !DISubprogram(name: "f", linkageName: "flink"
