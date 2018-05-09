; RUN: opt < %s -add-discriminators -S | FileCheck %s
; RUN: opt < %s -passes=add-discriminators -S | FileCheck %s

; We should not generate discriminators for DWARF versions prior to 4.
;
; Original code:
;
; int foo(long i) {
;   if (i < 5) return 2; else return 90;
; }
;
; None of the !dbg nodes associated with the if() statement should be
; altered. If they are, it means that the discriminators pass added a
; new lexical scope.

define i32 @foo(i64 %i) #0 !dbg !4 {
entry:
  %retval = alloca i32, align 4
  %i.addr = alloca i64, align 8
  store i64 %i, i64* %i.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %i.addr, metadata !13, metadata !DIExpression()), !dbg !14
  %0 = load i64, i64* %i.addr, align 8, !dbg !15
; CHECK:  %0 = load i64, i64* %i.addr, align 8, !dbg ![[ENTRY:[0-9]+]]
  %cmp = icmp slt i64 %0, 5, !dbg !15
; CHECK:  %cmp = icmp slt i64 %0, 5, !dbg ![[ENTRY:[0-9]+]]
  br i1 %cmp, label %if.then, label %if.else, !dbg !15
; CHECK:  br i1 %cmp, label %if.then, label %if.else, !dbg ![[ENTRY:[0-9]+]]

if.then:                                          ; preds = %entry
  store i32 2, i32* %retval, !dbg !15
  br label %return, !dbg !15

if.else:                                          ; preds = %entry
  store i32 90, i32* %retval, !dbg !15
  br label %return, !dbg !15

return:                                           ; preds = %if.else, %if.then
  %1 = load i32, i32* %retval, !dbg !17
  ret i32 %1, !dbg !17
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

; We should be able to add discriminators even in the absence of llvm.dbg.cu.
; When using sample profiles, the front end will generate line tables but it
; does not generate llvm.dbg.cu to prevent codegen from emitting debug info
; to the final binary.
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

; CHECK: !{i32 2, !"Dwarf Version", i32 2}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "no-discriminators", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, retainedNodes: !2)
; CHECK: ![[FOO:[0-9]+]] = distinct !DISubprogram(name: "foo"
!5 = !DIFile(filename: "no-discriminators", directory: ".")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "long int", size: 64, align: 64, encoding: DW_ATE_signed)
!10 = !{i32 2, !"Dwarf Version", i32 2}
!11 = !{i32 1, !"Debug Info Version", i32 3}
!12 = !{!"clang version 3.5.0 "}
!13 = !DILocalVariable(name: "i", line: 1, arg: 1, scope: !4, file: !5, type: !9)
!14 = !DILocation(line: 1, scope: !4)
!15 = !DILocation(line: 2, scope: !16)
; CHECK: ![[ENTRY]] = !DILocation(line: 2, scope: ![[BLOCK:[0-9]+]])
!16 = distinct !DILexicalBlock(line: 2, column: 0, file: !1, scope: !4)
; CHECK: ![[BLOCK]] = distinct !DILexicalBlock(scope: ![[FOO]],{{.*}} line: 2)
!17 = !DILocation(line: 3, scope: !4)
