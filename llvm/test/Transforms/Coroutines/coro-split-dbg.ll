; Make sure that coro-split correctly deals with debug information.
; The test here is simply that it does not result in bad IR that will crash opt.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -disable-output
source_filename = "coro.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @bar(...) local_unnamed_addr #2

; Function Attrs: nounwind uwtable
define ptr @f() #3 !dbg !16 {
entry:
  %0 = tail call token @llvm.coro.id(i32 0, ptr null, ptr @f, ptr null), !dbg !26
  %1 = tail call i64 @llvm.coro.size.i64(), !dbg !26
  %call = tail call ptr @malloc(i64 %1), !dbg !26
  %2 = tail call ptr @llvm.coro.begin(token %0, ptr %call) #9, !dbg !26
  tail call void @llvm.dbg.value(metadata ptr %2, metadata !21, metadata !12), !dbg !26
  br label %for.cond, !dbg !27

for.cond:                                         ; preds = %for.cond, %entry
  tail call void @llvm.dbg.value(metadata i32 undef, metadata !22, metadata !12), !dbg !28
  tail call void @llvm.dbg.value(metadata i32 undef, metadata !11, metadata !12) #7, !dbg !29
  tail call void (...) @bar() #7, !dbg !33
  %3 = tail call token @llvm.coro.save(ptr null), !dbg !34
  %4 = tail call i8 @llvm.coro.suspend(token %3, i1 false), !dbg !34
  %conv = sext i8 %4 to i32, !dbg !34
  switch i32 %conv, label %coro_Suspend [
    i32 0, label %for.cond
    i32 1, label %coro_Cleanup
  ], !dbg !34

coro_Cleanup:                                     ; preds = %for.cond
  %5 = tail call ptr @llvm.coro.free(token %0, ptr %2), !dbg !35
  tail call void @free(ptr nonnull %5), !dbg !36
  br label %coro_Suspend, !dbg !36

coro_Suspend:                                     ; preds = %for.cond, %if.then, %coro_Cleanup
  tail call i1 @llvm.coro.end(ptr null, i1 false, token none) #9, !dbg !38
  ret ptr %2, !dbg !39
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #4

; Function Attrs: argmemonly nounwind readonly
declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr) #5

; Function Attrs: nounwind
declare noalias ptr @malloc(i64) local_unnamed_addr #6
declare i64 @llvm.coro.size.i64() #1
declare ptr @llvm.coro.begin(token, ptr writeonly) #7
declare token @llvm.coro.save(ptr) #7
declare i8 @llvm.coro.suspend(token, i1) #7
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #4
declare ptr @llvm.coro.free(token, ptr nocapture readonly) #5
declare void @free(ptr nocapture) local_unnamed_addr #6
declare i1 @llvm.coro.end(ptr, i1, token) #7
declare ptr @llvm.coro.subfn.addr(ptr nocapture readonly, i8) #5

declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind uwtable presplitcoroutine "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { argmemonly nounwind }
attributes #5 = { argmemonly nounwind readonly }
attributes #6 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nounwind }
attributes #8 = { alwaysinline nounwind }
attributes #9 = { noduplicate }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "coro.c", directory: "/home/gor/build/bin")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 4.0.0"}
!6 = distinct !DISubprogram(name: "print", scope: !1, file: !1, line: 6, type: !7, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !10)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{!11}
!11 = !DILocalVariable(name: "v", arg: 1, scope: !6, file: !1, line: 6, type: !9)
!12 = !DIExpression()
!13 = !DILocation(line: 6, column: 16, scope: !6)
!14 = !DILocation(line: 6, column: 19, scope: !6)
!15 = !DILocation(line: 6, column: 25, scope: !6)
!16 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 8, type: !17, isLocal: false, isDefinition: true, scopeLine: 8, isOptimized: true, unit: !0, retainedNodes: !20)
!17 = !DISubroutineType(types: !18)
!18 = !{!19}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64, align: 64)
!20 = !{!21, !22, !24}
!21 = !DILocalVariable(name: "coro_hdl", scope: !16, file: !1, line: 9, type: !19)
!22 = !DILocalVariable(name: "i", scope: !23, file: !1, line: 11, type: !9)
!23 = distinct !DILexicalBlock(scope: !16, file: !1, line: 11, column: 3)
!24 = !DILocalVariable(name: "coro_mem", scope: !25, file: !1, line: 16, type: !19)
!25 = distinct !DILexicalBlock(scope: !16, file: !1, line: 16, column: 3)
!26 = !DILocation(line: 9, column: 3, scope: !16)
!27 = !DILocation(line: 11, column: 8, scope: !23)
!28 = !DILocation(line: 11, column: 12, scope: !23)
!29 = !DILocation(line: 6, column: 16, scope: !6, inlinedAt: !30)
!30 = distinct !DILocation(line: 12, column: 5, scope: !31)
!31 = distinct !DILexicalBlock(scope: !32, file: !1, line: 11, column: 25)
!32 = distinct !DILexicalBlock(scope: !23, file: !1, line: 11, column: 3)
!33 = !DILocation(line: 6, column: 19, scope: !6, inlinedAt: !30)
!34 = !DILocation(line: 13, column: 5, scope: !31)
!35 = !DILocation(line: 16, column: 3, scope: !25)
!36 = !DILocation(line: 16, column: 3, scope: !37)
!37 = distinct !DILexicalBlock(scope: !25, file: !1, line: 16, column: 3)
!38 = !DILocation(line: 16, column: 3, scope: !16)
!39 = !DILocation(line: 17, column: 1, scope: !16)
