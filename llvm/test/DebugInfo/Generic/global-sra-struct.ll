; RUN: opt -S -passes=globalopt < %s | FileCheck %s
source_filename = "test.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

%struct.mystruct = type { i32, i64 }
; Generated at -Os from:
;
; static struct mystruct {
;     int a;
;     long long int b;
; } static_struct;
; void __attribute__((nodebug)) foo(int in) { static_struct.a = in; }
; void __attribute__((nodebug)) bar(int in) { static_struct.b = in; }
; int main(int argc, char **argv)
; {
;     foo(argv[0][1]);
;     bar(argv[0][1]);
;     return static_struct.a + static_struct.b;
; }

; CHECK: @static_struct.0 = internal unnamed_addr global i32 0, align 8, !dbg ![[EL0:.*]]
; CHECK: @static_struct.1 = internal unnamed_addr global i64 0, align 8, !dbg ![[EL1:.*]]

; CHECK: ![[EL0]] = !DIGlobalVariableExpression(var: ![[VAR:.*]], expr: !DIExpression(DW_OP_LLVM_fragment, 0, 32))
; CHECK: ![[VAR]] = distinct !DIGlobalVariable(name: "static_struct"
; CHECK: ![[EL1]] = !DIGlobalVariableExpression(var: ![[VAR]], expr: !DIExpression(DW_OP_LLVM_fragment, 64, 64))

@static_struct = internal global %struct.mystruct zeroinitializer, align 8, !dbg !0

; Function Attrs: nounwind optsize ssp uwtable
define void @foo(i32 %in) #0 {
entry:
  store i32 %in, ptr @static_struct, align 8, !tbaa !17
  ret void
}

; Function Attrs: nounwind optsize ssp uwtable
define void @bar(i32 %in) #0 {
entry:
  %conv = sext i32 %in to i64
  store i64 %conv, ptr getelementptr inbounds (%struct.mystruct, ptr @static_struct, i32 0, i32 1), align 8, !tbaa !23
  ret void
}

; Function Attrs: nounwind optsize ssp uwtable
define i32 @main(i32 %argc, ptr %argv) #0 !dbg !24 {
entry:
  call void @llvm.dbg.value(metadata i32 %argc, metadata !31, metadata !33), !dbg !34
  call void @llvm.dbg.value(metadata ptr %argv, metadata !32, metadata !33), !dbg !35
  %0 = load ptr, ptr %argv, align 8, !dbg !36, !tbaa !37
  %arrayidx1 = getelementptr inbounds i8, ptr %0, i64 1, !dbg !36
  %1 = load i8, ptr %arrayidx1, align 1, !dbg !36, !tbaa !39
  %conv = sext i8 %1 to i32, !dbg !36
  call void @foo(i32 %conv) #2, !dbg !40
  %2 = load ptr, ptr %argv, align 8, !dbg !41, !tbaa !37
  %arrayidx3 = getelementptr inbounds i8, ptr %2, i64 1, !dbg !41
  %3 = load i8, ptr %arrayidx3, align 1, !dbg !41, !tbaa !39
  %conv4 = sext i8 %3 to i32, !dbg !41
  call void @bar(i32 %conv4) #2, !dbg !42
  %4 = load i32, ptr @static_struct, align 8, !dbg !43, !tbaa !17
  %conv5 = sext i32 %4 to i64, !dbg !44
  %5 = load i64, ptr getelementptr inbounds (%struct.mystruct, ptr @static_struct, i32 0, i32 1), align 8, !dbg !45, !tbaa !23
  %add = add nsw i64 %conv5, %5, !dbg !46
  %conv6 = trunc i64 %add to i32, !dbg !44
  ret i32 %conv6, !dbg !47
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind optsize ssp uwtable }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { optsize }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "static_struct", scope: !2, file: !3, line: 4, type: !6, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 6.0.0 (trunk 309852) (llvm/trunk 309850)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "test.c", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "mystruct", file: !3, line: 1, size: 128, elements: !7)
!7 = !{!8, !10}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !6, file: !3, line: 2, baseType: !9, size: 32)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !6, file: !3, line: 3, baseType: !11, size: 64, offset: 64)
!11 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{i32 7, !"PIC Level", i32 2}
!16 = !{!"clang version 6.0.0 (trunk 309852) (llvm/trunk 309850)"}
!17 = !{!18, !19, i64 0}
!18 = !{!"mystruct", !19, i64 0, !22, i64 8}
!19 = !{!"int", !20, i64 0}
!20 = !{!"omnipotent char", !21, i64 0}
!21 = !{!"Simple C/C++ TBAA"}
!22 = !{!"long long", !20, i64 0}
!23 = !{!18, !22, i64 8}
!24 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 7, type: !25, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !30)
!25 = !DISubroutineType(types: !26)
!26 = !{!9, !9, !27}
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !28, size: 64)
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !29, size: 64)
!29 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!30 = !{!31, !32}
!31 = !DILocalVariable(name: "argc", arg: 1, scope: !24, file: !3, line: 7, type: !9)
!32 = !DILocalVariable(name: "argv", arg: 2, scope: !24, file: !3, line: 7, type: !27)
!33 = !DIExpression()
!34 = !DILocation(line: 7, column: 14, scope: !24)
!35 = !DILocation(line: 7, column: 27, scope: !24)
!36 = !DILocation(line: 9, column: 9, scope: !24)
!37 = !{!38, !38, i64 0}
!38 = !{!"any pointer", !20, i64 0}
!39 = !{!20, !20, i64 0}
!40 = !DILocation(line: 9, column: 5, scope: !24)
!41 = !DILocation(line: 10, column: 9, scope: !24)
!42 = !DILocation(line: 10, column: 5, scope: !24)
!43 = !DILocation(line: 11, column: 26, scope: !24)
!44 = !DILocation(line: 11, column: 12, scope: !24)
!45 = !DILocation(line: 11, column: 44, scope: !24)
!46 = !DILocation(line: 11, column: 28, scope: !24)
!47 = !DILocation(line: 11, column: 5, scope: !24)
