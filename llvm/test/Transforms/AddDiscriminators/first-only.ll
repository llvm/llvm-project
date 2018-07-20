; RUN: opt < %s -add-discriminators -S | FileCheck %s
; RUN: opt < %s -passes=add-discriminators -S | FileCheck %s

; Test that the only instructions that receive a new discriminator in
; the block 'if.then' are those that share the same line number as
; the branch in 'entry'.
;
; Original code:
;
;       void foo(int i) {
;         int x, y;
;         if (i < 10) { x = i;
;             y = -i;
;         }
;       }

define void @foo(i32 %i) #0 !dbg !4 {
entry:
  %i.addr = alloca i32, align 4
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4, !dbg !10
  %cmp = icmp slt i32 %0, 10, !dbg !10
  br i1 %cmp, label %if.then, label %if.end, !dbg !10

if.then:                                          ; preds = %entry
  %1 = load i32, i32* %i.addr, align 4, !dbg !12
  store i32 %1, i32* %x, align 4, !dbg !12

  %2 = load i32, i32* %i.addr, align 4, !dbg !14
; CHECK:  %2 = load i32, i32* %i.addr, align 4, !dbg ![[THEN:[0-9]+]]

  %sub = sub nsw i32 0, %2, !dbg !14
; CHECK:  %sub = sub nsw i32 0, %2, !dbg ![[THEN]]

  store i32 %sub, i32* %y, align 4, !dbg !14
; CHECK:  store i32 %sub, i32* %y, align 4, !dbg ![[THEN]]

  br label %if.end, !dbg !15
; CHECK:  br label %if.end, !dbg ![[BR:[0-9]+]]

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !16
; CHECK:  ret void, !dbg ![[END:[0-9]+]]
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5 (trunk 199750) (llvm/trunk 199751)", isOptimized: false, emissionKind: NoDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "first-only.c", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "first-only.c", directory: ".")
!6 = !DISubroutineType(types: !{null})
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 1, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.5 (trunk 199750) (llvm/trunk 199751)"}
!10 = !DILocation(line: 3, scope: !11)

!11 = distinct !DILexicalBlock(line: 3, column: 0, file: !1, scope: !4)
; CHECK: ![[FOO:[0-9]+]] = distinct !DISubprogram(name: "foo"
; CHECK: ![[BLOCK1:[0-9]+]] = distinct !DILexicalBlock(scope: ![[FOO]],{{.*}} line: 3)

!12 = !DILocation(line: 3, scope: !13)

!13 = distinct !DILexicalBlock(line: 3, column: 0, file: !1, scope: !11)
; CHECK: !DILexicalBlockFile(scope: ![[BLOCK2:[0-9]+]],{{.*}} discriminator: 2)

!14 = !DILocation(line: 4, scope: !13)
; CHECK: ![[BLOCK2]] = distinct !DILexicalBlock(scope: ![[BLOCK1]],{{.*}} line: 3)

!15 = !DILocation(line: 5, scope: !13)
; CHECK: ![[THEN]] = !DILocation(line: 4, scope: ![[BLOCK2]])

!16 = !DILocation(line: 6, scope: !4)
; CHECK: ![[BR]] = !DILocation(line: 5, scope: ![[BLOCK2]])
; CHECK: ![[END]] = !DILocation(line: 6, scope: ![[FOO]])

