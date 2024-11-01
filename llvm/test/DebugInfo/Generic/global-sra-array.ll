; RUN: opt -S -passes=globalopt < %s | FileCheck %s
source_filename = "test.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

%struct.mystruct = type { i32, i64 }
; Generated at -Os from:
;
; static struct { int a; char b; } array[2];
; void __attribute__((nodebug)) foo(int in) { array[0].a = in; }
; void __attribute__((nodebug)) bar(int in) { array[1].a = in; }
; int main(int argc, char **argv)
; {
;   foo(argv[0][1]);
;   bar(argv[0][1]);
;   return array[0].a + array[1].a;
; }

%struct.anon = type { i32, i8 }

; This array is first split into two struct, which are then split into their
; elements, of which only .a survives.
@array = internal global [2 x %struct.anon] zeroinitializer, align 16, !dbg !0
; CHECK: @array.0 = internal unnamed_addr global i32 0, align 16, !dbg ![[EL0:.*]]
; CHECK: @array.1 = internal unnamed_addr global i32 0, align 8, !dbg ![[EL1:.*]]

; CHECK: ![[EL0]] = !DIGlobalVariableExpression(var: ![[VAR:.*]], expr: !DIExpression(DW_OP_LLVM_fragment, 0, 32))
; CHECK: ![[VAR]] = distinct !DIGlobalVariable(name: "array"
; CHECK: ![[EL1]] = !DIGlobalVariableExpression(var: ![[VAR]], expr: !DIExpression(DW_OP_LLVM_fragment, 64, 32))


; Function Attrs: nounwind optsize ssp uwtable
define void @foo(i32 %in) #0 {
entry:
  store i32 %in, i32* getelementptr inbounds ([2 x %struct.anon], [2 x %struct.anon]* @array, i64 0, i64 0, i32 0), align 16, !tbaa !20
  ret void
}

; Function Attrs: nounwind optsize ssp uwtable
define void @bar(i32 %in) #0 {
entry:
  store i32 %in, i32* getelementptr inbounds ([2 x %struct.anon], [2 x %struct.anon]* @array, i64 0, i64 1, i32 0), align 8, !tbaa !20
  ret void
}

; Function Attrs: nounwind optsize ssp uwtable
define i32 @main(i32 %argc, i8** %argv) #0 !dbg !25 {
entry:
  call void @llvm.dbg.value(metadata i32 %argc, metadata !31, metadata !33), !dbg !34
  call void @llvm.dbg.value(metadata i8** %argv, metadata !32, metadata !33), !dbg !35
  %0 = load i8*, i8** %argv, align 8, !dbg !36, !tbaa !37
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 1, !dbg !36
  %1 = load i8, i8* %arrayidx1, align 1, !dbg !36, !tbaa !39
  %conv = sext i8 %1 to i32, !dbg !36
  call void @foo(i32 %conv) #2, !dbg !40
  %2 = load i8*, i8** %argv, align 8, !dbg !41, !tbaa !37
  %arrayidx3 = getelementptr inbounds i8, i8* %2, i64 1, !dbg !41
  %3 = load i8, i8* %arrayidx3, align 1, !dbg !41, !tbaa !39
  %conv4 = sext i8 %3 to i32, !dbg !41
  call void @bar(i32 %conv4) #2, !dbg !42
  %4 = load i32, i32* getelementptr inbounds ([2 x %struct.anon], [2 x %struct.anon]* @array, i64 0, i64 0, i32 0), align 16, !dbg !43, !tbaa !20
  %5 = load i32, i32* getelementptr inbounds ([2 x %struct.anon], [2 x %struct.anon]* @array, i64 0, i64 1, i32 0), align 8, !dbg !44, !tbaa !20
  %add = add nsw i32 %4, %5, !dbg !45
  ret i32 %add, !dbg !46
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind optsize ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { optsize }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15, !16, !17, !18}
!llvm.ident = !{!19}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "array", scope: !2, file: !3, line: 1, type: !6, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 6.0.0 (trunk 309960) (llvm/trunk 309961)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "test.c", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 128, elements: !13)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 1, size: 64, elements: !8)
!8 = !{!9, !11}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !7, file: !3, line: 1, baseType: !10, size: 32)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !7, file: !3, line: 1, baseType: !12, size: 8, offset: 32)
!12 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!13 = !{!14}
!14 = !DISubrange(count: 2)
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"wchar_size", i32 4}
!18 = !{i32 7, !"PIC Level", i32 2}
!19 = !{!"clang version 6.0.0 (trunk 309960) (llvm/trunk 309961)"}
!20 = !{!21, !22, i64 0}
!21 = !{!"", !22, i64 0, !23, i64 4}
!22 = !{!"int", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C/C++ TBAA"}
!25 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 4, type: !26, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !30)
!26 = !DISubroutineType(types: !27)
!27 = !{!10, !10, !28}
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !29, size: 64)
!29 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!30 = !{!31, !32}
!31 = !DILocalVariable(name: "argc", arg: 1, scope: !25, file: !3, line: 4, type: !10)
!32 = !DILocalVariable(name: "argv", arg: 2, scope: !25, file: !3, line: 4, type: !28)
!33 = !DIExpression()
!34 = !DILocation(line: 4, column: 14, scope: !25)
!35 = !DILocation(line: 4, column: 27, scope: !25)
!36 = !DILocation(line: 6, column: 7, scope: !25)
!37 = !{!38, !38, i64 0}
!38 = !{!"any pointer", !23, i64 0}
!39 = !{!23, !23, i64 0}
!40 = !DILocation(line: 6, column: 3, scope: !25)
!41 = !DILocation(line: 7, column: 7, scope: !25)
!42 = !DILocation(line: 7, column: 3, scope: !25)
!43 = !DILocation(line: 8, column: 19, scope: !25)
!44 = !DILocation(line: 8, column: 32, scope: !25)
!45 = !DILocation(line: 8, column: 21, scope: !25)
!46 = !DILocation(line: 8, column: 3, scope: !25)
