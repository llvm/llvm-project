; RUN: llc -filetype=asm %s -o - | FileCheck %s

; Test the extension of debug ranges from predecessors.
; Generated from the source file LiveDebugValues.c:
; #include <stdio.h>
; int m;
; extern int inc(int n); 
; extern int change(int n); 
; extern int modify(int n); 
; int main(int argc, char **argv) {
;   int n;
;   if (argc != 2)
;     n = 2;
;   else
;     n = atoi(argv[1]);
;   n = change(n);
;   if (n > 10) {
;     m = modify(n);
;     m = m + n;  // var `m' doesn't has a dbg.value
;   }
;   else
;     m = inc(n); // var `m' doesn't has a dbg.value
;   printf("m(main): %d\n", m); 
;   return 0;
; }
; with clang -g -O3 -emit-llvm -c LiveDebugValues.c -S -o live-debug-values.ll
; This case will also produce multiple locations but only the debug range
; extension is tested here.

; DBG_VALUE for variable "n" is extended into %bb.5 from its predecessors %bb.3
; and %bb.4.
; CHECK:       .LBB0_5:
; CHECK-NEXT:  #DEBUG_VALUE: main:n <- $ebx
;   Other register values have been clobbered.
; CHECK-NOT:   #DEBUG_VALUE:
; CHECK:         movl    %ecx, m(%rip)

; ModuleID = 'LiveDebugValues.c'
source_filename = "test/DebugInfo/X86/live-debug-values.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@m = common global i32 0, align 4, !dbg !0
@.str = private unnamed_addr constant [13 x i8] c"m(main): %d\0A\00", align 1

; Function Attrs: nounwind uwtable
define i32 @main(i32 %argc, i8** nocapture readonly %argv) #0 !dbg !10 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %argc, metadata !17, metadata !20), !dbg !21
  tail call void @llvm.dbg.value(metadata i8** %argv, metadata !18, metadata !20), !dbg !22
  %cmp = icmp eq i32 %argc, 2, !dbg !24
  br i1 %cmp, label %if.else, label %if.end, !dbg !26

if.else:                                          ; preds = %entry
  %arrayidx = getelementptr inbounds i8*, i8** %argv, i64 1, !dbg !27
  %0 = load i8*, i8** %arrayidx, align 8, !dbg !27, !tbaa !28
  %call = tail call i32 (i8*, ...) bitcast (i32 (...)* @atoi to i32 (i8*, ...)*)(i8* %0) #1, !dbg !32
  tail call void @llvm.dbg.value(metadata i32 %call, metadata !19, metadata !20), !dbg !33
  br label %if.end

if.end:                                           ; preds = %if.else, %entry
  %n.0 = phi i32 [ %call, %if.else ], [ 2, %entry ]
  %call1 = tail call i32 @change(i32 %n.0) #1, !dbg !34
  tail call void @llvm.dbg.value(metadata i32 %call1, metadata !19, metadata !20), !dbg !33
  %cmp2 = icmp sgt i32 %call1, 10, !dbg !35
  br i1 %cmp2, label %if.then.3, label %if.else.5, !dbg !37

if.then.3:                                        ; preds = %if.end
  %call4 = tail call i32 @modify(i32 %call1) #1, !dbg !38
  %add = add nsw i32 %call4, %call1, !dbg !40
  br label %if.end.7, !dbg !41

if.else.5:                                        ; preds = %if.end
  %call6 = tail call i32 @inc(i32 %call1) #1, !dbg !42
  br label %if.end.7

if.end.7:                                         ; preds = %if.else.5, %if.then.3
  %storemerge = phi i32 [ %call6, %if.else.5 ], [ %add, %if.then.3 ]
  store i32 %storemerge, i32* @m, align 4, !dbg !43, !tbaa !44
  %call8 = tail call i32 (i8*, ...) @printf(i8* nonnull getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i64 0, i64 0), i32 %storemerge) #1, !dbg !46
  ret i32 0, !dbg !47
}

; Function Attrs: nounwind
declare i32 @atoi(...) #1

; Function Attrs: nounwind
declare i32 @change(i32) #1

; Function Attrs: nounwind
declare i32 @modify(i32) #1

; Function Attrs: nounwind
declare i32 @inc(i32) #1

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "m", scope: !2, file: !3, line: 2, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 3.8.0 (trunk 253049) ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "LiveDebugValues.c", directory: "/home/vt/julia/test/tvvikram")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.8.0 (trunk 253049) "}
!10 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 6, type: !11, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !16)
!11 = !DISubroutineType(types: !12)
!12 = !{!6, !6, !13}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64, align: 64)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64, align: 64)
!15 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!16 = !{!17, !18, !19}
!17 = !DILocalVariable(name: "argc", arg: 1, scope: !10, file: !3, line: 6, type: !6)
!18 = !DILocalVariable(name: "argv", arg: 2, scope: !10, file: !3, line: 6, type: !13)
!19 = !DILocalVariable(name: "n", scope: !10, file: !3, line: 7, type: !6)
!20 = !DIExpression()
!21 = !DILocation(line: 6, column: 14, scope: !10)
!22 = !DILocation(line: 6, column: 27, scope: !23)
!23 = !DILexicalBlockFile(scope: !10, file: !3, discriminator: 1)
!24 = !DILocation(line: 8, column: 12, scope: !25)
!25 = distinct !DILexicalBlock(scope: !10, file: !3, line: 8, column: 7)
!26 = !DILocation(line: 8, column: 7, scope: !10)
!27 = !DILocation(line: 11, column: 14, scope: !25)
!28 = !{!29, !29, i64 0}
!29 = !{!"any pointer", !30, i64 0}
!30 = !{!"omnipotent char", !31, i64 0}
!31 = !{!"Simple C/C++ TBAA"}
!32 = !DILocation(line: 11, column: 9, scope: !25)
!33 = !DILocation(line: 7, column: 7, scope: !23)
!34 = !DILocation(line: 12, column: 7, scope: !10)
!35 = !DILocation(line: 13, column: 9, scope: !36)
!36 = distinct !DILexicalBlock(scope: !10, file: !3, line: 13, column: 7)
!37 = !DILocation(line: 13, column: 7, scope: !10)
!38 = !DILocation(line: 14, column: 9, scope: !39)
!39 = distinct !DILexicalBlock(scope: !36, file: !3, line: 13, column: 15)
!40 = !DILocation(line: 15, column: 11, scope: !39)
!41 = !DILocation(line: 16, column: 3, scope: !39)
!42 = !DILocation(line: 18, column: 9, scope: !36)
!43 = !DILocation(line: 15, column: 7, scope: !39)
!44 = !{!45, !45, i64 0}
!45 = !{!"int", !30, i64 0}
!46 = !DILocation(line: 19, column: 3, scope: !10)
!47 = !DILocation(line: 20, column: 3, scope: !10)

