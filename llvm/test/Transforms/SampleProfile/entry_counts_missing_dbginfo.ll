; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/entry_counts_cold.prof -S -profile-sample-accurate | FileCheck %s
; ModuleID = 'temp.bc'
; This is a test similar to entry_counts_cold.ll. The key differences are:
; - we want profile-sample-accurate. Normally, that would trigger resetting function entry counts to 0, and then update
; them based on the sample profile (SampleProfile::runOnFunction).
; - we don't have debug info for the function. This means we can't match sample counts in the profile to locations in this function.
; When we attempt to update its profile info (at the end of SampleProfile::runOnModule - llvm::updateProfileCallee), we'll call
; updateProfWeight with a 0 priorEntryCount, which will result in a division by 0.
;
; We're using @bar and the call to @baz.
;
; Currently, the test just ensures opt doesn't ICE
source_filename = "temp.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; Function Attrs: nounwind ssp uwtable
; CHECK: define i32 @top({{.*}} !prof [[TOP:![0-9]+]]
define i32 @top(i32* %p) #0 !dbg !8 {
entry:
  %p.addr = alloca i32*, align 8
  store i32* %p, i32** %p.addr, align 8, !tbaa !15
  call void @llvm.dbg.declare(metadata i32** %p.addr, metadata !14, metadata !DIExpression()), !dbg !19
  %0 = load i32*, i32** %p.addr, align 8, !dbg !20, !tbaa !15
  %call = call i32 @foo(i32* %0), !dbg !21
; foo is inlined
; CHECK-NOT: call i32 @foo
; CHECK: call i32 @bar
  %1 = load i32*, i32** %p.addr, align 8, !dbg !22, !tbaa !15
  %2 = load i32, i32* %1, align 4, !dbg !24, !tbaa !25
  %tobool = icmp ne i32 %2, 0, !dbg !24
  br i1 %tobool, label %if.then, label %if.end, !dbg !27

if.then:                                          ; preds = %entry
  %3 = load i32*, i32** %p.addr, align 8, !dbg !28, !tbaa !15
; bar is not inlined
; CHECK: call i32 @bar
  %call1 = call i32 @bar(i32* %3), !dbg !29
  br label %if.end, !dbg !29

if.end:                                           ; preds = %if.then, %entry
  ret i32 0, !dbg !30
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind ssp uwtable
; CHECK: define i32 @foo({{.*}} !prof [[FOO:![0-9]+]]
define i32 @foo(i32* %p) #0 !dbg !31 {
entry:
  %p.addr = alloca i32*, align 8
  %a = alloca i32, align 4
  store i32* %p, i32** %p.addr, align 8, !tbaa !15
  call void @llvm.dbg.declare(metadata i32** %p.addr, metadata !33, metadata !DIExpression()), !dbg !35
  %0 = bitcast i32* %a to i8*, !dbg !36
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #4, !dbg !36
  call void @llvm.dbg.declare(metadata i32* %a, metadata !34, metadata !DIExpression()), !dbg !37
  %1 = load i32*, i32** %p.addr, align 8, !dbg !38, !tbaa !15
  %arrayidx = getelementptr inbounds i32, i32* %1, i64 3, !dbg !38
  %2 = load i32, i32* %arrayidx, align 4, !dbg !38, !tbaa !25
  %3 = load i32*, i32** %p.addr, align 8, !dbg !39, !tbaa !15
  %arrayidx1 = getelementptr inbounds i32, i32* %3, i64 2, !dbg !39
  %4 = load i32, i32* %arrayidx1, align 4, !dbg !40, !tbaa !25
  %add = add nsw i32 %4, %2, !dbg !40
  store i32 %add, i32* %arrayidx1, align 4, !dbg !40, !tbaa !25
  %5 = load i32*, i32** %p.addr, align 8, !dbg !41, !tbaa !15
  %call = call i32 @bar(i32* %5), !dbg !42
  store i32 %call, i32* %a, align 4, !dbg !43, !tbaa !25
  %6 = load i32, i32* %a, align 4, !dbg !44, !tbaa !25
  %add2 = add nsw i32 %6, 1, !dbg !45
  %7 = bitcast i32* %a to i8*, !dbg !46
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %7) #4, !dbg !46
  ret i32 %add2, !dbg !47
}

; Function Attrs: nounwind ssp uwtable
; CHECK: define i32 @bar(i32* %p) #0 !prof [[BAR:![0-9]+]] {
define i32 @bar(i32* %p) #0 {
entry:
  %p.addr = alloca i32*, align 8
  store i32* %p, i32** %p.addr, align 8, !tbaa !15
  call void @llvm.dbg.declare(metadata i32** %p.addr, metadata !50, metadata !DIExpression()), !dbg !51
  ; CHECK: call void (...) @baz(), !dbg !{{[0-9]+}}
  call void (...) @baz(), !dbg !52, !prof !100
  %0 = load i32*, i32** %p.addr, align 8, !dbg !53, !tbaa !15
  %arrayidx = getelementptr inbounds i32, i32* %0, i64 2, !dbg !53
  %1 = load i32, i32* %arrayidx, align 4, !dbg !53, !tbaa !25
  %2 = load i32*, i32** %p.addr, align 8, !dbg !54, !tbaa !15
  %arrayidx1 = getelementptr inbounds i32, i32* %2, i64 1, !dbg !54
  %3 = load i32, i32* %arrayidx1, align 4, !dbg !55, !tbaa !25
  %add = add nsw i32 %3, %1, !dbg !55
  store i32 %add, i32* %arrayidx1, align 4, !dbg !55, !tbaa !25
  %4 = load i32*, i32** %p.addr, align 8, !dbg !56, !tbaa !15
  %arrayidx2 = getelementptr inbounds i32, i32* %4, i64 3, !dbg !56
  %5 = load i32, i32* %arrayidx2, align 4, !dbg !56, !tbaa !25
  ret i32 %5, !dbg !57
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #2

declare void @baz(...) #3

attributes #0 = { nounwind ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" "use-sample-profile" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

; CHECK: [[TOP]] = !{!"function_entry_count", i64 101}
; CHECK: [[FOO]] = !{!"function_entry_count", i64 151}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: GNU)
!1 = !DIFile(filename: "temp.c", directory: "llvm/test/Transforms/SampleProfile")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 8.0.0"}
!8 = distinct !DISubprogram(name: "top", scope: !1, file: !1, line: 5, type: !9, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !12}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!13 = !{!14}
!14 = !DILocalVariable(name: "p", arg: 1, scope: !8, file: !1, line: 5, type: !12)
!15 = !{!16, !16, i64 0}
!16 = !{!"any pointer", !17, i64 0}
!17 = !{!"omnipotent char", !18, i64 0}
!18 = !{!"Simple C/C++ TBAA"}
!19 = !DILocation(line: 5, column: 14, scope: !8)
!20 = !DILocation(line: 6, column: 7, scope: !8)
!21 = !DILocation(line: 6, column: 3, scope: !8)
!22 = !DILocation(line: 7, column: 8, scope: !23)
!23 = distinct !DILexicalBlock(scope: !8, file: !1, line: 7, column: 7)
!24 = !DILocation(line: 7, column: 7, scope: !23)
!25 = !{!26, !26, i64 0}
!26 = !{!"int", !17, i64 0}
!27 = !DILocation(line: 7, column: 7, scope: !8)
!28 = !DILocation(line: 8, column: 9, scope: !23)
!29 = !DILocation(line: 8, column: 5, scope: !23)
!30 = !DILocation(line: 9, column: 3, scope: !8)
!31 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 12, type: !9, scopeLine: 12, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !32)
!32 = !{!33, !34}
!33 = !DILocalVariable(name: "p", arg: 1, scope: !31, file: !1, line: 12, type: !12)
!34 = !DILocalVariable(name: "a", scope: !31, file: !1, line: 13, type: !11)
!35 = !DILocation(line: 12, column: 14, scope: !31)
!36 = !DILocation(line: 13, column: 3, scope: !31)
!37 = !DILocation(line: 13, column: 7, scope: !31)
!38 = !DILocation(line: 14, column: 11, scope: !31)
!39 = !DILocation(line: 14, column: 3, scope: !31)
!40 = !DILocation(line: 14, column: 8, scope: !31)
!41 = !DILocation(line: 15, column: 11, scope: !31)
!42 = !DILocation(line: 15, column: 7, scope: !31)
!43 = !DILocation(line: 15, column: 5, scope: !31)
!44 = !DILocation(line: 16, column: 10, scope: !31)
!45 = !DILocation(line: 16, column: 11, scope: !31)
!46 = !DILocation(line: 17, column: 1, scope: !31)
!47 = !DILocation(line: 16, column: 3, scope: !31)
!48 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 19, type: !9, scopeLine: 19, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !49)
!49 = !{!50}
!50 = !DILocalVariable(name: "p", arg: 1, scope: !48, file: !1, line: 19, type: !12)
!51 = !DILocation(line: 19, column: 15, scope: !48)
!52 = !DILocation(line: 20, column: 3, scope: !48)
!53 = !DILocation(line: 21, column: 11, scope: !48)
!54 = !DILocation(line: 21, column: 3, scope: !48)
!55 = !DILocation(line: 21, column: 8, scope: !48)
!56 = !DILocation(line: 22, column: 10, scope: !48)
!57 = !DILocation(line: 22, column: 3, scope: !48)
!100 = !{!"branch_weights", i32 5}
