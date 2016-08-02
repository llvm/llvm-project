; RUN: llc -mtriple x86_64-pc-linux -O0 < %s | FileCheck %s

; Make sure that the sequence of debug locations for function foo is correctly
; generated. More specifically, .loc entries for lines 4,5,6,7 must appear in
; the correct sequence.

; $ clang -emit-llvm -S -g dbg-combine.c
; 1.  int foo()
; 2.  {
; 3.     int elems = 3;
; 4.     int array1[elems];
; 5.     array1[0]=0;
; 6.     array1[1]=1;
; 7.     array1[2]=2;
; 8.     int array2[elems];
; 9.     array2[0]=1;
; 10.    return array2[0];
; 11. }

; CHECK: .loc    1 4
; CHECK: .loc    1 5
; CHECK: .loc    1 6
; CHECK: .loc    1 7

; ModuleID = 'dbg-combine.c'
; Function Attrs: nounwind uwtable
define i32 @foo() #0 !dbg !4 {
entry:
  %elems = alloca i32, align 4
  %saved_stack = alloca i8*
  %cleanup.dest.slot = alloca i32
  call void @llvm.dbg.declare(metadata i32* %elems, metadata !12, metadata !13), !dbg !14
  store i32 3, i32* %elems, align 4, !dbg !14
  %0 = load i32, i32* %elems, align 4, !dbg !15
  %1 = zext i32 %0 to i64, !dbg !16
  %2 = call i8* @llvm.stacksave(), !dbg !16
  store i8* %2, i8** %saved_stack, !dbg !16
  %vla = alloca i32, i64 %1, align 16, !dbg !16
  call void @llvm.dbg.declare(metadata i32* %vla, metadata !17, metadata !21), !dbg !22
  %arrayidx = getelementptr inbounds i32, i32* %vla, i64 0, !dbg !23
  store i32 0, i32* %arrayidx, align 4, !dbg !24
  %arrayidx1 = getelementptr inbounds i32, i32* %vla, i64 1, !dbg !25
  store i32 1, i32* %arrayidx1, align 4, !dbg !26
  %arrayidx2 = getelementptr inbounds i32, i32* %vla, i64 2, !dbg !27
  store i32 2, i32* %arrayidx2, align 4, !dbg !28
  %3 = load i32, i32* %elems, align 4, !dbg !29
  %4 = zext i32 %3 to i64, !dbg !30
  %vla3 = alloca i32, i64 %4, align 16, !dbg !30
  call void @llvm.dbg.declare(metadata i32* %vla3, metadata !31, metadata !21), !dbg !32
  %arrayidx4 = getelementptr inbounds i32, i32* %vla3, i64 0, !dbg !33
  store i32 1, i32* %arrayidx4, align 4, !dbg !34
  %arrayidx5 = getelementptr inbounds i32, i32* %vla3, i64 0, !dbg !35
  %5 = load i32, i32* %arrayidx5, align 4, !dbg !35
  store i32 1, i32* %cleanup.dest.slot
  %6 = load i8*, i8** %saved_stack, !dbg !36
  call void @llvm.stackrestore(i8* %6), !dbg !36
  ret i32 %5, !dbg !36
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare i8* @llvm.stacksave() #2

; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*) #2

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.7.0 (trunk 227074)", isOptimized: false, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "dbg-combine.c", directory: "/home/probinson/projects/scratch")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, isOptimized: false, scopeLine: 2, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "dbg-combine.c", directory: "/home/probinson/projects/scratch")
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{!"clang version 3.7.0 (trunk 227074)"}
!12 = !DILocalVariable(name: "elems", line: 3, scope: !4, file: !5, type: !8)
!13 = !DIExpression()
!14 = !DILocation(line: 3, column: 8, scope: !4)
!15 = !DILocation(line: 4, column: 15, scope: !4)
!16 = !DILocation(line: 4, column: 4, scope: !4)
!17 = !DILocalVariable(name: "array1", line: 4, scope: !4, file: !5, type: !18)
!18 = !DICompositeType(tag: DW_TAG_array_type, align: 32, baseType: !8, elements: !19)
!19 = !{!20}
!20 = !DISubrange(count: -1)
!21 = !DIExpression(DW_OP_deref)
!22 = !DILocation(line: 4, column: 8, scope: !4)
!23 = !DILocation(line: 5, column: 4, scope: !4)
!24 = !DILocation(line: 5, column: 13, scope: !4)
!25 = !DILocation(line: 6, column: 4, scope: !4)
!26 = !DILocation(line: 6, column: 13, scope: !4)
!27 = !DILocation(line: 7, column: 4, scope: !4)
!28 = !DILocation(line: 7, column: 13, scope: !4)
!29 = !DILocation(line: 8, column: 15, scope: !4)
!30 = !DILocation(line: 8, column: 4, scope: !4)
!31 = !DILocalVariable(name: "array2", line: 8, scope: !4, file: !5, type: !18)
!32 = !DILocation(line: 8, column: 8, scope: !4)
!33 = !DILocation(line: 9, column: 4, scope: !4)
!34 = !DILocation(line: 9, column: 13, scope: !4)
!35 = !DILocation(line: 10, column: 11, scope: !4)
!36 = !DILocation(line: 11, column: 1, scope: !4)
