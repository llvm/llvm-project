; RUN: llc %s -filetype=obj -o - | llvm-dwarfdump --debug-dump=info - | FileCheck %s
; CHECK: DW_TAG_variable
;                                                      DW_OP_fbreg
; CHECK-NEXT: DW_AT_location [DW_FORM_exprloc]      (<0x2> 91 00 )
; CHECK-NEXT: DW_AT_name {{.*}}"i"

target datalayout = "e-m:e-p:32:32-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

; Function Attrs: nounwind uwtable
declare i32 @foo(i32 %i) #0

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define i32 @main() #0 !dbg !9 {
entry:
  %retval = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata i32* %i, metadata !20, metadata !16), !dbg !21
  store i32 20, i32* %i, align 4, !dbg !21
  %0 = load i32, i32* %i, align 4, !dbg !22
  %call = call i32 @foo(i32 %0), !dbg !23
  ret i32 %call, !dbg !24
}

attributes #0 = { nounwind uwtable  }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.7.0", isOptimized: false, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "x.c", directory: "")
!2 = !{}
!3 = !{!4, !9}
!4 = distinct !DISubprogram(name: "foo", line: 2, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 3, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "x.c", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = distinct !DISubprogram(name: "main", line: 8, isLocal: false, isDefinition: true, isOptimized: false, scopeLine: 9, file: !1, scope: !5, type: !10, variables: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!8}
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{!"clang version 3.7.0"}
!15 = !DILocalVariable(name: "i", line: 2, arg: 1, scope: !4, file: !5, type: !8)
!16 = !DIExpression()
!17 = !DILocation(line: 2, column: 10, scope: !4)
!18 = !DILocation(line: 4, column: 10, scope: !4)
!19 = !DILocation(line: 4, column: 3, scope: !4)
!20 = !DILocalVariable(name: "i", line: 10, scope: !9, file: !5, type: !8)
!21 = !DILocation(line: 10, column: 7, scope: !9)
!22 = !DILocation(line: 11, column: 15, scope: !9)
!23 = !DILocation(line: 11, column: 10, scope: !9)
!24 = !DILocation(line: 11, column: 3, scope: !9)
