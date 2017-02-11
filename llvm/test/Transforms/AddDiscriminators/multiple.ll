; RUN: opt < %s -add-discriminators -S | FileCheck %s
; RUN: opt < %s -passes=add-discriminators -S | FileCheck %s

; Discriminator support for multiple CFG paths on the same line.
;
;       void foo(int i) {
;         int x;
;         if (i < 10) x = i; else x = -i;
;       }
;
; The two stores inside the if-then-else line must have different discriminator
; values.

define void @foo(i32 %i) #0 !dbg !4 {
entry:
  %i.addr = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4, !dbg !10
  %cmp = icmp slt i32 %0, 10, !dbg !10
  br i1 %cmp, label %if.then, label %if.else, !dbg !10

if.then:                                          ; preds = %entry
  %1 = load i32, i32* %i.addr, align 4, !dbg !10
; CHECK:  %1 = load i32, i32* %i.addr, align 4, !dbg ![[THEN:[0-9]+]]

  store i32 %1, i32* %x, align 4, !dbg !10
; CHECK:  store i32 %1, i32* %x, align 4, !dbg ![[THEN]]

  br label %if.end, !dbg !10
; CHECK:  br label %if.end, !dbg ![[THEN]]

if.else:                                          ; preds = %entry
  %2 = load i32, i32* %i.addr, align 4, !dbg !10
; CHECK:  %2 = load i32, i32* %i.addr, align 4, !dbg ![[ELSE:[0-9]+]]

  %sub = sub nsw i32 0, %2, !dbg !10
; CHECK:  %sub = sub nsw i32 0, %2, !dbg ![[ELSE]]

  store i32 %sub, i32* %x, align 4, !dbg !10
; CHECK:  store i32 %sub, i32* %x, align 4, !dbg ![[ELSE]]

  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void, !dbg !12
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5 (trunk 199750) (llvm/trunk 199751)", isOptimized: false, emissionKind: NoDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "multiple.c", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "multiple.c", directory: ".")
!6 = !DISubroutineType(types: !{null, !13})
!13 = !DIBasicType(encoding: DW_ATE_signed, name: "int", size: 32, align: 32)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 1, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.5 (trunk 199750) (llvm/trunk 199751)"}
!10 = !DILocation(line: 3, scope: !11)
!11 = distinct !DILexicalBlock(line: 3, column: 0, file: !1, scope: !4)
!12 = !DILocation(line: 4, scope: !4)

; CHECK: ![[THEN]] = !DILocation(line: 3, scope: ![[THENBLOCK:[0-9]+]])
; CHECK: ![[THENBLOCK]] = !DILexicalBlockFile(scope: ![[SCOPE:[0-9]+]],{{.*}} discriminator: 2)
; CHECK: ![[ELSE]] = !DILocation(line: 3, scope: ![[ELSEBLOCK:[0-9]+]])
; CHECK: ![[ELSEBLOCK]] = !DILexicalBlockFile(scope: ![[SCOPE]],{{.*}} discriminator: 4)
