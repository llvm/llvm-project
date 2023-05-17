; RUN: rm -rf %t && mkdir -p %t
; RUN: sed -e 's|PATTERN|%/t|g' %s | opt -passes=insert-gcov-profiling -disable-output

; This is a crash test.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.vector = type { i8 }

; Function Attrs: nounwind
define i32 @_Z4testv() #0 !dbg !15 {
entry:
  %retval = alloca i32, align 4
  %__range = alloca ptr, align 8
  %ref.tmp = alloca %struct.vector, align 1
  %undef.agg.tmp = alloca %struct.vector, align 1
  %__begin = alloca ptr, align 8
  %__end = alloca ptr, align 8
  %spec = alloca i8, align 1
  call void @llvm.dbg.declare(metadata ptr %__range, metadata !27, metadata !DIExpression()), !dbg !30
  br label %0

; <label>:0                                       ; preds = %entry
  call void @_Z13TagFieldSpecsv(), !dbg !31
  store ptr %ref.tmp, ptr %__range, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %__begin, metadata !32, metadata !DIExpression()), !dbg !30
  %1 = load ptr, ptr %__range, align 8, !dbg !31
  %call = call ptr @_ZN6vector5beginEv(ptr %1), !dbg !31
  store ptr %call, ptr %__begin, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata ptr %__end, metadata !33, metadata !DIExpression()), !dbg !30
  %2 = load ptr, ptr %__range, align 8, !dbg !31
  %call1 = call ptr @_ZN6vector3endEv(ptr %2), !dbg !31
  store ptr %call1, ptr %__end, align 8, !dbg !31
  br label %for.cond, !dbg !31

for.cond:                                         ; preds = %for.inc, %0
  %3 = load ptr, ptr %__begin, align 8, !dbg !34
  %4 = load ptr, ptr %__end, align 8, !dbg !34
  %cmp = icmp ne ptr %3, %4, !dbg !34
  br i1 %cmp, label %for.body, label %for.end, !dbg !34

for.body:                                         ; preds = %for.cond
  call void @llvm.dbg.declare(metadata ptr %spec, metadata !37, metadata !DIExpression()), !dbg !31
  %5 = load ptr, ptr %__begin, align 8, !dbg !38
  %6 = load i8, ptr %5, align 1, !dbg !38
  store i8 %6, ptr %spec, align 1, !dbg !38
  br label %for.inc, !dbg !38

for.inc:                                          ; preds = %for.body
  %7 = load ptr, ptr %__begin, align 8, !dbg !40
  %incdec.ptr = getelementptr inbounds i8, ptr %7, i32 1, !dbg !40
  store ptr %incdec.ptr, ptr %__begin, align 8, !dbg !40
  br label %for.cond, !dbg !40

for.end:                                          ; preds = %for.cond
  call void @llvm.trap(), !dbg !42
  unreachable, !dbg !42

return:                                           ; No predecessors!
  %8 = load i32, ptr %retval, !dbg !44
  ret i32 %8, !dbg !44
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @_Z13TagFieldSpecsv() #2

declare ptr @_ZN6vector5beginEv(ptr) #2

declare ptr @_ZN6vector3endEv(ptr) #2

; Function Attrs: noreturn nounwind
declare void @llvm.trap() #3

; Function Attrs: nounwind
define void @_Z2f1v() #0 !dbg !20 {
entry:
  br label %0

; <label>:0                                       ; preds = %entry
  ret void, !dbg !45
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noreturn nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!23, !24}
!llvm.gcov = !{!25}
!llvm.ident = !{!26}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 (trunk 209871)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "<stdin>", directory: "PATTERN")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "vector", line: 21, size: 8, align: 8, file: !5, elements: !6, identifier: "_ZTS6vector")
!5 = !DIFile(filename: "linezero.cc", directory: "PATTERN")
!6 = !{!7, !13}
!7 = !DISubprogram(name: "begin", linkageName: "_ZN6vector5beginEv", line: 25, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 25, file: !5, scope: !4, type: !8)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !12}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !11)
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !4)
!13 = !DISubprogram(name: "end", linkageName: "_ZN6vector3endEv", line: 26, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 26, file: !5, scope: !4, type: !8)
!15 = distinct !DISubprogram(name: "test", linkageName: "_Z4testv", line: 50, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 50, file: !5, scope: !16, type: !17, retainedNodes: !2)
!16 = !DIFile(filename: "linezero.cc", directory: "PATTERN")
!17 = !DISubroutineType(types: !18)
!18 = !{!19}
!19 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!20 = distinct !DISubprogram(name: "f1", linkageName: "_Z2f1v", line: 54, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 54, file: !5, scope: !16, type: !21, retainedNodes: !2)
!21 = !DISubroutineType(types: !22)
!22 = !{null}
!23 = !{i32 2, !"Dwarf Version", i32 4}
!24 = !{i32 2, !"Debug Info Version", i32 3}
!25 = !{!"PATTERN/linezero.o", !0}
!26 = !{!"clang version 3.5.0 (trunk 209871)"}
!27 = !DILocalVariable(name: "__range", flags: DIFlagArtificial, scope: !28, type: !29)
!28 = distinct !DILexicalBlock(line: 51, column: 0, file: !5, scope: !15)
!29 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: !4)
!30 = !DILocation(line: 0, scope: !28)
!31 = !DILocation(line: 51, scope: !28)
!32 = !DILocalVariable(name: "__begin", flags: DIFlagArtificial, scope: !28, type: !10)
!33 = !DILocalVariable(name: "__end", flags: DIFlagArtificial, scope: !28, type: !10)
!34 = !DILocation(line: 51, scope: !35)
!35 = distinct !DILexicalBlock(line: 51, column: 0, file: !5, scope: !36)
!36 = distinct !DILexicalBlock(line: 51, column: 0, file: !5, scope: !28)
!37 = !DILocalVariable(name: "spec", line: 51, scope: !28, file: !16, type: !11)
!38 = !DILocation(line: 51, scope: !39)
!39 = distinct !DILexicalBlock(line: 51, column: 0, file: !5, scope: !28)
!40 = !DILocation(line: 51, scope: !41)
!41 = distinct !DILexicalBlock(line: 51, column: 0, file: !5, scope: !28)
!42 = !DILocation(line: 51, scope: !43)
!43 = distinct !DILexicalBlock(line: 51, column: 0, file: !5, scope: !28)
!44 = !DILocation(line: 52, scope: !15)
!45 = !DILocation(line: 54, scope: !20)
