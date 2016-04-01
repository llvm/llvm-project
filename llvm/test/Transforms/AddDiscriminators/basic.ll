; RUN: opt < %s -add-discriminators -S | FileCheck %s

; Basic DWARF discriminator test. All the instructions in block
; 'if.then' should have a different discriminator value than
; the conditional branch at the end of block 'entry'.
;
; Original code:
;
;       void foo(int i) {
;         int x;
;         if (i < 10) x = i;
;       }

define void @foo(i32 %i) #0 !dbg !4 {
entry:
  %i.addr = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4, !dbg !10
  %cmp = icmp slt i32 %0, 10, !dbg !10
  br i1 %cmp, label %if.then, label %if.end, !dbg !10

if.then:                                          ; preds = %entry
  %1 = load i32, i32* %i.addr, align 4, !dbg !10
; CHECK:  %1 = load i32, i32* %i.addr, align 4, !dbg ![[THEN:[0-9]+]]

  store i32 %1, i32* %x, align 4, !dbg !10
; CHECK:  store i32 %1, i32* %x, align 4, !dbg ![[THEN]]

  br label %if.end, !dbg !10
; CHECK:   br label %if.end, !dbg ![[THEN]]

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !12
; CHECK:   ret void, !dbg ![[END:[0-9]+]]
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

; We should be able to add discriminators even in the absence of llvm.dbg.cu.
; When using sample profiles, the front end will generate line tables but it
; does not generate llvm.dbg.cu to prevent codegen from emitting debug info
; to the final binary.
; !llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "basic.c", directory: ".")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 1, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "basic.c", directory: ".")
!6 = !DISubroutineType(types: !2)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 1, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.5 "}
!10 = !DILocation(line: 3, scope: !11)
!11 = distinct !DILexicalBlock(line: 3, column: 0, file: !1, scope: !4)
!12 = !DILocation(line: 4, scope: !4)

; CHECK: ![[FOO:[0-9]+]] = distinct !DISubprogram(name: "foo"
; CHECK: ![[BLOCK:[0-9]+]] = distinct !DILexicalBlock(scope: ![[FOO]],{{.*}} line: 3)
; CHECK: ![[THEN]] = !DILocation(line: 3, scope: ![[BLOCKFILE:[0-9]+]])
; CHECK: ![[BLOCKFILE]] = !DILexicalBlockFile(scope: ![[BLOCK]],{{.*}} discriminator: 1)
; CHECK: ![[END]] = !DILocation(line: 4, scope: ![[FOO]])
