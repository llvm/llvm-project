; RUN: opt -S -passes=argpromotion < %s | FileCheck %s
;
; Source code:
;   __attribute__((noinline)) static int is_absolute_path(const char *path)
;   {
;     return path[0] == '/';
;   }
;
;   void quit(char *buf);
;   const char *make_nonrelative_path(char *buf, const char *path)
;   {
;     if (is_absolute_path(path))
;       quit(buf);
;     return buf;
;   }

define dso_local ptr @make_nonrelative_path(ptr noundef %buf, ptr noundef %path) local_unnamed_addr #0 !dbg !10 {
    #dbg_value(ptr %buf, !18, !DIExpression(), !20)
    #dbg_value(ptr %path, !19, !DIExpression(), !20)
  %x = call fastcc i32 @is_absolute_path(ptr noundef %path), !dbg !21
  %y = icmp eq i32 %x, 0, !dbg !21
  br i1 %y, label %to_ret, label %to_quit, !dbg !21

to_quit:
  call void @quit(ptr noundef %buf), !dbg !23
  br label %to_ret, !dbg !23

to_ret:
  ret ptr %buf, !dbg !24
}

; Function Attrs: noinline nounwind uwtable
define internal fastcc range(i32 0, 2) i32 @is_absolute_path(ptr noundef %path) unnamed_addr #1 !dbg !25 {
    #dbg_value(ptr %path, !30, !DIExpression(), !31)
  %x = load i8, ptr %path, align 1, !dbg !32, !tbaa !33
  %y = icmp eq i8 %x, 47, !dbg !36
  %z = zext i1 %y to i32, !dbg !36
  ret i32 %z, !dbg !37
}

; CHECK: define internal fastcc range(i32 0, 2) i32 @is_absolute_path(i8 {{.*}})

declare !dbg !38 void @quit(ptr noundef) local_unnamed_addr

attributes #0 = { nounwind }
attributes #1 = { noinline nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 21.0.0git (git@github.com:yonghong-song/llvm-project.git 25cfee009e78194d1f7ca70779d63ef1936cc7b9)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/tests/sig-change/prom", checksumkind: CSK_MD5, checksum: "bcc8cf18726713f5d2ab6d82e8ff459d")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 21.0.0git (git@github.com:yonghong-song/llvm-project.git 25cfee009e78194d1f7ca70779d63ef1936cc7b9)"}
!10 = distinct !DISubprogram(name: "make_nonrelative_path", scope: !1, file: !1, line: 7, type: !11, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !17)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !16, !13}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !15)
!15 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!17 = !{!18, !19}
!18 = !DILocalVariable(name: "buf", arg: 1, scope: !10, file: !1, line: 7, type: !16)
!19 = !DILocalVariable(name: "path", arg: 2, scope: !10, file: !1, line: 7, type: !13)
!20 = !DILocation(line: 0, scope: !10)
!21 = !DILocation(line: 9, column: 7, scope: !22)
!22 = distinct !DILexicalBlock(scope: !10, file: !1, line: 9, column: 7)
!23 = !DILocation(line: 10, column: 5, scope: !22)
!24 = !DILocation(line: 11, column: 3, scope: !10)
!25 = distinct !DISubprogram(name: "is_absolute_path", scope: !1, file: !1, line: 1, type: !26, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !29)

; CHECK: distinct !DISubprogram(name: "is_absolute_path", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[#]], scopeLine: [[#]], flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized | DISPFlagArgChanged, unit: ![[#]], retainedNodes: ![[#]])

!26 = !DISubroutineType(types: !27)
!27 = !{!28, !13}
!28 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!29 = !{!30}
!30 = !DILocalVariable(name: "path", arg: 1, scope: !25, file: !1, line: 1, type: !13)
!31 = !DILocation(line: 0, scope: !25)
!32 = !DILocation(line: 3, column: 10, scope: !25)
!33 = !{!34, !34, i64 0}
!34 = !{!"omnipotent char", !35, i64 0}
!35 = !{!"Simple C/C++ TBAA"}
!36 = !DILocation(line: 3, column: 18, scope: !25)
!37 = !DILocation(line: 3, column: 3, scope: !25)
!38 = !DISubprogram(name: "quit", scope: !1, file: !1, line: 6, type: !39, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!39 = !DISubroutineType(types: !40)
!40 = !{null, !16}
