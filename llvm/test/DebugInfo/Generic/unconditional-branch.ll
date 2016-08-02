; REQUIRES: object-emission
; PR 19261

; RUN: %llc_dwarf -fast-isel=false -O0 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; CHECK: {{0x[0-9a-f]+}}      1      0      1   0             0  is_stmt
; CHECK: {{0x[0-9a-f]+}}      2      0      1   0             0  is_stmt
; CHECK: {{0x[0-9a-f]+}}      4      0      1   0             0  is_stmt

; IR generated from clang -O0 -g with the following source:
;void foo(int i){
;  switch(i){
;  default:
;    break;
;  }
;  return;
;}

; Function Attrs: nounwind
define void @foo(i32 %i) #0 !dbg !4 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %i.addr, metadata !12, metadata !DIExpression()), !dbg !13
  %0 = load i32, i32* %i.addr, align 4, !dbg !14
  switch i32 %0, label %sw.default [
  ], !dbg !14

sw.epilog:                                        ; preds = %sw.default
  ret void, !dbg !17

sw.default:                                       ; preds = %entry
  br label %sw.epilog, !dbg !15

}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 (204712)", isOptimized: false, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "test.c", directory: "D:\5Cwork\5CEPRs\5C396363")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 1, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "test.c", directory: "D:CworkCEPRsC396363")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 1, !"Debug Info Version", i32 3}
!11 = !{!"clang version 3.5.0 (204712)"}
!12 = !DILocalVariable(name: "i", line: 1, arg: 1, scope: !4, file: !5, type: !8)
!13 = !DILocation(line: 1, scope: !4)
!14 = !DILocation(line: 2, scope: !4)
!15 = !DILocation(line: 4, scope: !16)
!16 = distinct !DILexicalBlock(line: 2, column: 0, file: !1, scope: !4)
!17 = !DILocation(line: 6, scope: !4)
