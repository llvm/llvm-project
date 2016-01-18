; RUN: llc -mtriple thumbv7 -print-before=post-RA-sched -print-after=post-RA-sched %s -o /dev/null 2>&1 | FileCheck %s

; ModuleID = '<stdin>'
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7"

%struct.s = type opaque

; Function Attrs: nounwind
define arm_aapcscc i32 @f(%struct.s* %s, i32 %u, i8* %b, i32 %n) #0 !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata %struct.s* %s, i64 0, metadata !19, metadata !28), !dbg !29
  tail call void @llvm.dbg.value(metadata i32 %u, i64 0, metadata !20, metadata !28), !dbg !29
  tail call void @llvm.dbg.value(metadata i8* %b, i64 0, metadata !21, metadata !28), !dbg !29
  tail call void @llvm.dbg.value(metadata i32 %n, i64 0, metadata !22, metadata !28), !dbg !29
  %cmp = icmp ult i32 %n, 4, !dbg !30
  br i1 %cmp, label %return, label %if.end, !dbg !32

if.end:                                           ; preds = %entry
  tail call arm_aapcscc void @g(%struct.s* %s, i8* %b, i32 %n) #3, !dbg !33
  br label %return, !dbg !34

return:                                           ; preds = %entry, %if.end
  %retval.0 = phi i32 [ 0, %if.end ], [ -1, %entry ]
  ret i32 %retval.0, !dbg !35
}

; NOTE: This is checking that the register in the DEBUG_VALUE node is not
; accidentally being marked as KILL.  The DBG_VALUE node gets introduced in
; If-Conversion, and gets bundled into the IT block.  The Post RA Scheduler
; attempts to schedule the Machine Instr, and tries to tag the register in the
; debug value as KILL'ed, resulting in a DEBUG_VALUE node changing codegen!  (or
; hopefully, triggering an assert).

; CHECK: BUNDLE %ITSTATE<imp-def,dead>
; CHECK:  * DBG_VALUE %R1, %noreg, !"u"
; CHECK-NOT:  * DBG_VALUE %R1<kill>, %noreg, !"u"

declare arm_aapcscc void @g(%struct.s*, i8*, i32) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!23, !24, !25, !26}
!llvm.ident = !{!27}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.7.0  (llvm/trunk 237059)", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "<stdin>", directory: "/Users/compnerd/Source/llvm")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "f", scope: !5, file: !5, line: 9, type: !6, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: true, variables: !18)
!5 = !DIFile(filename: "<stdin>", directory: "/Users/compnerd/Source/llvm")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9, !12, !13, !17}
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 32, align: 32)
!10 = !DIDerivedType(tag: DW_TAG_typedef, name: "s", file: !5, line: 5, baseType: !11)
!11 = !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !5, line: 5, flags: DIFlagFwdDecl)
!12 = !DIBasicType(name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 32, align: 32)
!14 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !15)
!15 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint8_t", file: !5, line: 2, baseType: !16)
!16 = !DIBasicType(name: "unsigned char", size: 8, align: 8, encoding: DW_ATE_unsigned_char)
!17 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !5, line: 3, baseType: !12)
!18 = !{!19, !20, !21, !22}
!19 = !DILocalVariable(name: "s", arg: 1, scope: !4, file: !5, line: 9, type: !9)
!20 = !DILocalVariable(name: "u", arg: 2, scope: !4, file: !5, line: 9, type: !12)
!21 = !DILocalVariable(name: "b", arg: 3, scope: !4, file: !5, line: 9, type: !13)
!22 = !DILocalVariable(name: "n", arg: 4, scope: !4, file: !5, line: 9, type: !17)
!23 = !{i32 2, !"Dwarf Version", i32 4}
!24 = !{i32 2, !"Debug Info Version", i32 3}
!25 = !{i32 1, !"wchar_size", i32 4}
!26 = !{i32 1, !"min_enum_size", i32 4}
!27 = !{!"clang version 3.7.0  (llvm/trunk 237059)"}
!28 = !DIExpression()
!29 = !DILocation(line: 9, scope: !4)
!30 = !DILocation(line: 10, scope: !31)
!31 = distinct !DILexicalBlock(scope: !4, file: !5, line: 10)
!32 = !DILocation(line: 10, scope: !4)
!33 = !DILocation(line: 13, scope: !4)
!34 = !DILocation(line: 14, scope: !4)
!35 = !DILocation(line: 15, scope: !4)
