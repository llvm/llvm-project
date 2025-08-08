; Just run it through opt, no passes needed.
; RUN: opt < %s -S | FileCheck %s

; ModuleID = 'various_ir_values.c'
source_filename = "various_ir_values.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local void @foo(ptr %A) #0 !dbg !7 {
entry:
  %A.addr = alloca ptr, align 8, !DIAssignID !16
  %i = alloca i32, align 4
    #dbg_assign(i1 undef, !13, !DIExpression(), !16, ptr %A.addr, !DIExpression(), !17)
  store ptr %A, ptr %A.addr, align 8, !tbaa !18
    #dbg_declare(ptr %A.addr, !13, !DIExpression(), !17)
  call void @llvm.lifetime.start.p0(ptr %i) #2, !dbg !22
    #dbg_declare(ptr %i, !14, !DIExpression(), !23)
  store i32 0, ptr %i, align 4, !dbg !23, !tbaa !24
  br label %for.cond, !dbg !22

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4, !dbg !26, !tbaa !24
  %1 = load ptr, ptr %A.addr, align 8, !dbg !28, !tbaa !18
  %2 = load i32, ptr %1, align 4, !dbg !29, !tbaa !24
  %cmp = icmp slt i32 %0, %2, !dbg !30
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !31, !prof !32

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0(ptr %i) #2, !dbg !33
  br label %for.end

for.body:                                         ; preds = %for.cond
  %3 = load ptr, ptr %A.addr, align 8, !dbg !34, !tbaa !18
  %4 = load i32, ptr %i, align 4, !dbg !35, !tbaa !24
  %idxprom = sext i32 %4 to i64, !dbg !34
  %arrayidx = getelementptr inbounds i32, ptr %3, i64 %idxprom, !dbg !34
  store i32 0, ptr %arrayidx, align 4, !dbg !36, !tbaa !24
  br label %for.inc, !dbg !34

for.inc:                                          ; preds = %for.body
  %5 = load i32, ptr %i, align 4, !dbg !37, !tbaa !24
  %inc = add nsw i32 %5, 1, !dbg !37
  store i32 %inc, ptr %i, align 4, !dbg !37, !tbaa !24
  br label %for.cond, !dbg !33, !llvm.loop !38

for.end:                                          ; preds = %for.cond.cleanup
  ret void, !dbg !40
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr nocapture) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr nocapture) #1

; Function Attrs: nounwind uwtable
define dso_local void @bar(ptr %A) #0 !dbg !41 {
entry:
  %A.addr = alloca ptr, align 8
  %i = alloca i32, align 4
  store ptr %A, ptr %A.addr, align 8, !tbaa !18
    #dbg_declare(ptr %A.addr, !43, !DIExpression(), !46)
  call void @llvm.lifetime.start.p0(ptr %i) #2, !dbg !47
    #dbg_declare(ptr %i, !44, !DIExpression(), !48)
  store i32 0, ptr %i, align 4, !dbg !48, !tbaa !24
  br label %for.cond, !dbg !47

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4, !dbg !49, !tbaa !24
  %1 = load ptr, ptr %A.addr, align 8, !dbg !51, !tbaa !18
  %2 = load i32, ptr %1, align 4, !dbg !52, !tbaa !24
  %cmp = icmp slt i32 %0, %2, !dbg !53
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !54

for.cond.cleanup:                                 ; preds = %for.cond
  call void @llvm.lifetime.end.p0(ptr %i) #2, !dbg !55
  br label %for.end

for.body:                                         ; preds = %for.cond
  %3 = load ptr, ptr %A.addr, align 8, !dbg !56, !tbaa !18
  %4 = load i32, ptr %i, align 4, !dbg !57, !tbaa !24
  %idxprom = sext i32 %4 to i64, !dbg !56
  %arrayidx = getelementptr inbounds i32, ptr %3, i64 %idxprom, !dbg !56
  store i32 0, ptr %arrayidx, align 4, !dbg !58, !tbaa !24
  br label %for.inc, !dbg !56

for.inc:                                          ; preds = %for.body
  %5 = load i32, ptr %i, align 4, !dbg !59, !tbaa !24
  %inc = add nsw i32 %5, 1, !dbg !59
  store i32 %inc, ptr %i, align 4, !dbg !59, !tbaa !24
  br label %for.cond, !dbg !55, !llvm.loop !60

for.end:                                          ; preds = %for.cond.cleanup
  ret void, !dbg !62
}

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "denormal-fp-math"="ieee,ieee" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0 (git@github.com:llvm/llvm-project.git 1d5da8cd30fce1c0a2c2fa6ba656dbfaa36192c8)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "various_ir_values.c", directory: "/data/build/llvm-project")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 11.0.0 (git@github.com:llvm/llvm-project.git 1d5da8cd30fce1c0a2c2fa6ba656dbfaa36192c8)"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13, !14}
!13 = !DILocalVariable(name: "A", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!14 = !DILocalVariable(name: "i", scope: !15, file: !1, line: 3, type: !11)
!15 = distinct !DILexicalBlock(scope: !7, file: !1, line: 3, column: 3)
!16 = distinct !DIAssignID()
!17 = !DILocation(line: 1, column: 15, scope: !7)
!18 = !{!19, !19, i64 0}
!19 = !{!"any pointer", !20, i64 0}
!20 = !{!"omnipotent char", !21, i64 0}
!21 = !{!"Simple C/C++ TBAA"}
!22 = !DILocation(line: 3, column: 8, scope: !15)
!23 = !DILocation(line: 3, column: 12, scope: !15)
!24 = !{!25, !25, i64 0}
!25 = !{!"int", !20, i64 0}
!26 = !DILocation(line: 3, column: 19, scope: !27)
!27 = distinct !DILexicalBlock(scope: !15, file: !1, line: 3, column: 3)
!28 = !DILocation(line: 3, column: 24, scope: !27)
!29 = !DILocation(line: 3, column: 23, scope: !27)
!30 = !DILocation(line: 3, column: 21, scope: !27)
!31 = !DILocation(line: 3, column: 3, scope: !15)
!32 = !{!"branch_weights", i32 1, i32 1048575}
!33 = !DILocation(line: 3, column: 3, scope: !27)
!34 = !DILocation(line: 4, column: 5, scope: !27)
!35 = !DILocation(line: 4, column: 7, scope: !27)
!36 = !DILocation(line: 4, column: 10, scope: !27)
!37 = !DILocation(line: 3, column: 27, scope: !27)
!38 = distinct !{!38, !31, !39}
!39 = !DILocation(line: 4, column: 12, scope: !15)
!40 = !DILocation(line: 5, column: 1, scope: !7)
!41 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 7, type: !8, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !42)
!42 = !{!43, !44}
!43 = !DILocalVariable(name: "A", arg: 1, scope: !41, file: !1, line: 7, type: !10)
!44 = !DILocalVariable(name: "i", scope: !45, file: !1, line: 9, type: !11)
!45 = distinct !DILexicalBlock(scope: !41, file: !1, line: 9, column: 3)
!46 = !DILocation(line: 7, column: 15, scope: !41)
!47 = !DILocation(line: 9, column: 8, scope: !45)
!48 = !DILocation(line: 9, column: 12, scope: !45)
!49 = !DILocation(line: 9, column: 19, scope: !50)
!50 = distinct !DILexicalBlock(scope: !45, file: !1, line: 9, column: 3)
!51 = !DILocation(line: 9, column: 24, scope: !50)
!52 = !DILocation(line: 9, column: 23, scope: !50)
!53 = !DILocation(line: 9, column: 21, scope: !50)
!54 = !DILocation(line: 9, column: 3, scope: !45)
!55 = !DILocation(line: 9, column: 3, scope: !50)
!56 = !DILocation(line: 10, column: 5, scope: !50)
!57 = !DILocation(line: 10, column: 7, scope: !50)
!58 = !DILocation(line: 10, column: 10, scope: !50)
!59 = !DILocation(line: 9, column: 27, scope: !50)
!60 = distinct !{!60, !54, !61}
!61 = !DILocation(line: 10, column: 12, scope: !45)
!62 = !DILocation(line: 11, column: 1, scope: !41)
