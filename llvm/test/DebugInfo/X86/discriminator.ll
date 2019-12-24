; RUN: llc -mtriple=i386-unknown-unknown -mcpu=core2 %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-line %t | FileCheck %s
;
; Generated from:
;
;   int foo(int i) {
;     if (i < 10) return i - 1;
;     return 0;
;   }
;
; Manually generated debug nodes !14 and !15 to incorporate an
; arbitrary discriminator with value 42.

define i32 @foo(i32 %i) #0 !dbg !4 {
entry:
  %retval = alloca i32, align 4
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4, !dbg !10
  %cmp = icmp slt i32 %0, 10, !dbg !10
  br i1 %cmp, label %if.then, label %if.end, !dbg !10

if.then:                                          ; preds = %entry
  %1 = load i32, i32* %i.addr, align 4, !dbg !14
  %sub = sub nsw i32 %1, 1, !dbg !14
  store i32 %sub, i32* %retval, !dbg !14
  br label %return, !dbg !14

if.end:                                           ; preds = %entry
  store i32 0, i32* %retval, !dbg !12
  br label %return, !dbg !12

return:                                           ; preds = %if.end, %if.then
  %2 = load i32, i32* %retval, !dbg !13
  ret i32 %2, !dbg !13
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "discriminator.c", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "discriminator.c", directory: ".")
!6 = !DISubroutineType(types: !2)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 1, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.5 "}
!10 = !DILocation(line: 2, scope: !11)
!11 = distinct !DILexicalBlock(line: 2, column: 0, file: !1, scope: !4)
!12 = !DILocation(line: 3, scope: !4)
!13 = !DILocation(line: 4, scope: !4)
!14 = !DILocation(line: 2, scope: !15)
!15 = !DILexicalBlockFile(discriminator: 42, file: !1, scope: !4)

; CHECK: Address            Line   Column File   ISA Discriminator Flags
; CHECK: ------------------ ------ ------ ------ --- ------------- -------------
; CHECK: 0x000000000000000a      2      0      1   0            42 {{$}}
