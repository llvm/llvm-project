; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source code:
;   extern void bla1;
;   void *test1() {
;     void *x = &bla1;
;     return x;
;   }
;
;   extern const void bla2;
;   const void *test2() {
;     const void *x = &bla2;
;     return x;
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

@bla1 = external dso_local global i8, align 1, !dbg !0
@bla2 = external dso_local constant i8, align 1, !dbg !6

; Function Attrs: norecurse nounwind readnone
define dso_local nonnull i8* @test1() local_unnamed_addr #0 !dbg !13 {
entry:
  call void @llvm.dbg.value(metadata i8* @bla1, metadata !18, metadata !DIExpression()), !dbg !19
  ret i8* @bla1, !dbg !20
}

; Function Attrs: norecurse nounwind readnone
define dso_local nonnull i8* @test2() local_unnamed_addr #0 !dbg !21 {
entry:
  call void @llvm.dbg.value(metadata i8* @bla2, metadata !26, metadata !DIExpression()), !dbg !27
  ret i8* @bla2, !dbg !28
}

; CHECK:        .quad bla1
; CHECK-NEXT:   DW_TAG_variable
;
; CHECK:        .quad   bla2
; CHECK-NEXT:   DW_TAG_const_type
; CHECK-NEXT:   DW_TAG_subprogram

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "bla1", scope: !2, file: !3, line: 1, isLocal: false, isDefinition: false)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git 8a8c6913a931e8bbd119012f4badd81155a0f48a)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/home/yhs/tmp3")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "bla2", scope: !2, file: !3, line: 7, type: !8, isLocal: false, isDefinition: false)
!8 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!9 = !{i32 7, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 8a8c6913a931e8bbd119012f4badd81155a0f48a)"}
!13 = distinct !DISubprogram(name: "test1", scope: !3, file: !3, line: 2, type: !14, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !17)
!14 = !DISubroutineType(types: !15)
!15 = !{!16}
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!17 = !{!18}
!18 = !DILocalVariable(name: "x", scope: !13, file: !3, line: 3, type: !16)
!19 = !DILocation(line: 0, scope: !13)
!20 = !DILocation(line: 4, column: 3, scope: !13)
!21 = distinct !DISubprogram(name: "test2", scope: !3, file: !3, line: 8, type: !22, scopeLine: 8, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !25)
!22 = !DISubroutineType(types: !23)
!23 = !{!24}
!24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!25 = !{!26}
!26 = !DILocalVariable(name: "x", scope: !21, file: !3, line: 9, type: !24)
!27 = !DILocation(line: 0, scope: !21)
!28 = !DILocation(line: 10, column: 3, scope: !21)
