; RUN: opt -mtriple=aarch64-unknown-linux-gnu -S %s -passes=sroa -o - | FileCheck %s

; In this test we want to ensure that the location of phi instruction merged from locations
; of %mul3 and %mul9 belongs to the correct scope (DILexicalBlockFile), so that line
; number of that location belongs to the corresponding file.

; Generated with clang from
;  1 # 1 "1.c" 1
;  2 # 1 "1.c" 2
;  3 int foo(int a) {
;  4   int i = 0;
;  5   if ((a & 1) == 1) {
;  6     a -= 1;
;  7 # 1 "m.c" 1
;  8 # 40 "m.c"
;  9 i += a;
; 10 i -= 10*a;
; 11 i *= a*a;
; 12 # 6 "1.c" 2
; 13  } else {
; 14     a += 3;
; 15 # 1 "m.c" 1
; 16 # 40 "m.c"
; 17 i += a;
; 18 i -= 10*a;
; 19 i *= a*a;
; 20 # 9 "1.c" 2
; 21  }
; 22   return i;
; 23 }

; ModuleID = 'repro.c'
source_filename = "repro.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local i32 @foo(i32 noundef %a) #0 !dbg !9 {
; CHECK:    phi i32 {{.*}}, !dbg [[PHILOC:![0-9]+]]
;
entry:
  %a.addr = alloca i32, align 4, !DIAssignID !17
  #dbg_assign(i1 undef, !15, !DIExpression(), !17, ptr %a.addr, !DIExpression(), !18)
  %i = alloca i32, align 4, !DIAssignID !19
  #dbg_assign(i1 undef, !16, !DIExpression(), !19, ptr %i, !DIExpression(), !18)
  store i32 %a, ptr %a.addr, align 4, !tbaa !20, !DIAssignID !24
  #dbg_assign(i32 %a, !15, !DIExpression(), !24, ptr %a.addr, !DIExpression(), !18)
  call void @llvm.lifetime.start.p0(i64 4, ptr %i) #2, !dbg !25
  store i32 0, ptr %i, align 4, !dbg !26, !tbaa !20, !DIAssignID !27
  #dbg_assign(i32 0, !16, !DIExpression(), !27, ptr %i, !DIExpression(), !18)
  %0 = load i32, ptr %a.addr, align 4, !dbg !28, !tbaa !20
  %and = and i32 %0, 1, !dbg !30
  %cmp = icmp eq i32 %and, 1, !dbg !31
  br i1 %cmp, label %if.then, label %if.else, !dbg !31

if.then:                                          ; preds = %entry
  %1 = load i32, ptr %a.addr, align 4, !dbg !32, !tbaa !20
  %sub = sub nsw i32 %1, 1, !dbg !32
  store i32 %sub, ptr %a.addr, align 4, !dbg !32, !tbaa !20, !DIAssignID !34
  #dbg_assign(i32 %sub, !15, !DIExpression(), !34, ptr %a.addr, !DIExpression(), !18)
  %2 = load i32, ptr %a.addr, align 4, !dbg !35, !tbaa !20
  %3 = load i32, ptr %i, align 4, !dbg !38, !tbaa !20
  %add = add nsw i32 %3, %2, !dbg !38
  store i32 %add, ptr %i, align 4, !dbg !38, !tbaa !20, !DIAssignID !39
  #dbg_assign(i32 %add, !16, !DIExpression(), !39, ptr %i, !DIExpression(), !18)
  %4 = load i32, ptr %a.addr, align 4, !dbg !40, !tbaa !20
  %mul = mul nsw i32 10, %4, !dbg !41
  %5 = load i32, ptr %i, align 4, !dbg !42, !tbaa !20
  %sub1 = sub nsw i32 %5, %mul, !dbg !42
  store i32 %sub1, ptr %i, align 4, !dbg !42, !tbaa !20, !DIAssignID !43
  #dbg_assign(i32 %sub1, !16, !DIExpression(), !43, ptr %i, !DIExpression(), !18)
  %6 = load i32, ptr %a.addr, align 4, !dbg !44, !tbaa !20
  %7 = load i32, ptr %a.addr, align 4, !dbg !45, !tbaa !20
  %mul2 = mul nsw i32 %6, %7, !dbg !46
  %8 = load i32, ptr %i, align 4, !dbg !47, !tbaa !20
  %mul3 = mul nsw i32 %8, %mul2, !dbg !47
  store i32 %mul3, ptr %i, align 4, !dbg !47, !tbaa !20, !DIAssignID !48
  #dbg_assign(i32 %mul3, !16, !DIExpression(), !48, ptr %i, !DIExpression(), !18)
  br label %if.end, !dbg !49

if.else:                                          ; preds = %entry
  %9 = load i32, ptr %a.addr, align 4, !dbg !51, !tbaa !20
  %add4 = add nsw i32 %9, 3, !dbg !51
  store i32 %add4, ptr %a.addr, align 4, !dbg !51, !tbaa !20, !DIAssignID !53
  #dbg_assign(i32 %add4, !15, !DIExpression(), !53, ptr %a.addr, !DIExpression(), !18)
  %10 = load i32, ptr %a.addr, align 4, !dbg !54, !tbaa !20
  %11 = load i32, ptr %i, align 4, !dbg !56, !tbaa !20
  %add5 = add nsw i32 %11, %10, !dbg !56
  store i32 %add5, ptr %i, align 4, !dbg !56, !tbaa !20, !DIAssignID !57
  #dbg_assign(i32 %add5, !16, !DIExpression(), !57, ptr %i, !DIExpression(), !18)
  %12 = load i32, ptr %a.addr, align 4, !dbg !58, !tbaa !20
  %mul6 = mul nsw i32 10, %12, !dbg !59
  %13 = load i32, ptr %i, align 4, !dbg !60, !tbaa !20
  %sub7 = sub nsw i32 %13, %mul6, !dbg !60
  store i32 %sub7, ptr %i, align 4, !dbg !60, !tbaa !20, !DIAssignID !61
  #dbg_assign(i32 %sub7, !16, !DIExpression(), !61, ptr %i, !DIExpression(), !18)
  %14 = load i32, ptr %a.addr, align 4, !dbg !62, !tbaa !20
  %15 = load i32, ptr %a.addr, align 4, !dbg !63, !tbaa !20
  %mul8 = mul nsw i32 %14, %15, !dbg !64
  %16 = load i32, ptr %i, align 4, !dbg !65, !tbaa !20
  %mul9 = mul nsw i32 %16, %mul8, !dbg !65
  store i32 %mul9, ptr %i, align 4, !dbg !65, !tbaa !20, !DIAssignID !66
  #dbg_assign(i32 %mul9, !16, !DIExpression(), !66, ptr %i, !DIExpression(), !18)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %17 = load i32, ptr %i, align 4, !dbg !67, !tbaa !20
  call void @llvm.lifetime.end.p0(i64 4, ptr %i) #2, !dbg !68
  ret i32 %17, !dbg !69
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 20.0.0git (git@github.com:llvm/llvm-project.git c8ee1164bd6ae2f0a603c53d1d29ad5a3225c5cd)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "repro.c", directory: "", checksumkind: CSK_MD5, checksum: "51454d2babc57d5ea92df6734236bd39")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 2}
!6 = !{i32 7, !"frame-pointer", i32 1}
!7 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!8 = !{!"clang version 20.0.0git (git@github.com:llvm/llvm-project.git c8ee1164bd6ae2f0a603c53d1d29ad5a3225c5cd)"}
!9 = distinct !DISubprogram(name: "foo", scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!10 = !DIFile(filename: "1.c", directory: "")
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15, !16}
!15 = !DILocalVariable(name: "a", arg: 1, scope: !9, file: !10, line: 1, type: !13)
!16 = !DILocalVariable(name: "i", scope: !9, file: !10, line: 2, type: !13)
!17 = distinct !DIAssignID()
!18 = !DILocation(line: 0, scope: !9)
!19 = distinct !DIAssignID()
!20 = !{!21, !21, i64 0}
!21 = !{!"int", !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C/C++ TBAA"}
!24 = distinct !DIAssignID()
!25 = !DILocation(line: 2, column: 3, scope: !9)
!26 = !DILocation(line: 2, column: 7, scope: !9)
!27 = distinct !DIAssignID()
!28 = !DILocation(line: 3, column: 8, scope: !29)
!29 = distinct !DILexicalBlock(scope: !9, file: !10, line: 3, column: 7)
!30 = !DILocation(line: 3, column: 10, scope: !29)
!31 = !DILocation(line: 3, column: 15, scope: !29)
!32 = !DILocation(line: 4, column: 7, scope: !33)
!33 = distinct !DILexicalBlock(scope: !29, file: !10, line: 3, column: 21)
!34 = distinct !DIAssignID()
!35 = !DILocation(line: 40, column: 6, scope: !36)
!36 = !DILexicalBlockFile(scope: !33, file: !37, discriminator: 0)
!37 = !DIFile(filename: "m.c", directory: "")
!38 = !DILocation(line: 40, column: 3, scope: !36)
!39 = distinct !DIAssignID()
!40 = !DILocation(line: 41, column: 9, scope: !36)
!41 = !DILocation(line: 41, column: 8, scope: !36)
!42 = !DILocation(line: 41, column: 3, scope: !36)
!43 = distinct !DIAssignID()
!44 = !DILocation(line: 42, column: 6, scope: !36)
!45 = !DILocation(line: 42, column: 8, scope: !36)
!46 = !DILocation(line: 42, column: 7, scope: !36)
!47 = !DILocation(line: 42, column: 3, scope: !36)
!48 = distinct !DIAssignID()
!49 = !DILocation(line: 6, column: 2, scope: !50)
!50 = !DILexicalBlockFile(scope: !33, file: !10, discriminator: 0)
!51 = !DILocation(line: 7, column: 7, scope: !52)
!52 = distinct !DILexicalBlock(scope: !29, file: !10, line: 6, column: 9)
!53 = distinct !DIAssignID()
!54 = !DILocation(line: 40, column: 6, scope: !55)
!55 = !DILexicalBlockFile(scope: !52, file: !37, discriminator: 0)
!56 = !DILocation(line: 40, column: 3, scope: !55)
!57 = distinct !DIAssignID()
!58 = !DILocation(line: 41, column: 9, scope: !55)
!59 = !DILocation(line: 41, column: 8, scope: !55)
!60 = !DILocation(line: 41, column: 3, scope: !55)
!61 = distinct !DIAssignID()
!62 = !DILocation(line: 42, column: 6, scope: !55)
!63 = !DILocation(line: 42, column: 8, scope: !55)
!64 = !DILocation(line: 42, column: 7, scope: !55)
!65 = !DILocation(line: 42, column: 3, scope: !55)
!66 = distinct !DIAssignID()
!67 = !DILocation(line: 10, column: 10, scope: !9)
!68 = !DILocation(line: 11, column: 1, scope: !9)
!69 = !DILocation(line: 10, column: 3, scope: !9)

;.
; CHECK: [[SP:![0-9]+]] = distinct !DISubprogram(name: "foo", scope: [[FILE1:![0-9]+]], file: [[FILE1]], line: 1
; CHECK: [[FILE1]] = !DIFile(filename: "1.c", directory: "")
; CHECK: [[LB1:![0-9]+]] = distinct !DILexicalBlock(scope: [[SP]], file: [[FILE1]], line: 3, column: 7)
; CHECK: [[LB2:![0-9]+]] = distinct !DILexicalBlock(scope: [[LB1]], file: [[FILE1]], line: 3, column: 21)
; CHECK: [[LBF:![0-9]+]] = !DILexicalBlockFile(scope: [[LB2]], file: [[FILE2:![0-9]+]], discriminator: 0)
; CHECK: [[FILE2]] = !DIFile(filename: "m.c", directory: "")
; CHECK: [[PHILOC]] = !DILocation(line: 42, column: 3, scope: [[LBF]])
