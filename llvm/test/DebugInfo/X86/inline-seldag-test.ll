; RUN: llc -mtriple=x86_64-linux-gnu -fast-isel=false -filetype=obj < %s -o - | llvm-dwarfdump -debug-info - | FileCheck %s
; RUN: llc -mtriple=x86_64-linux-gnu -fast-isel=false -filetype=asm < %s -o - | FileCheck --check-prefix=ASM %s

; Generated from:
; clang-tot -c -S -emit-llvm -g inline-seldag-test.c
; inline int __attribute__((always_inline)) f(int y) {
;   return y ? 4 : 7;
; }
; void func() {
;   volatile int x;
;   x = f(x);
; }

; CHECK: DW_TAG_inlined_subroutine
; CHECK-NEXT: DW_AT_abstract_origin {{.*}} "f"


; Make sure the condition test is attributed to the inline function, not the
; location of the test's operands within the caller.

; ASM: # inline-seldag-test.c:2:0
; ASM-NOT: .loc
; ASM: testl

; Function Attrs: nounwind uwtable
define void @func() #0 !dbg !4 {
entry:
  %y.addr.i = alloca i32, align 4
  %x = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %x, metadata !15, metadata !DIExpression()), !dbg !17
  %0 = load volatile i32, i32* %x, align 4, !dbg !18
  store i32 %0, i32* %y.addr.i, align 4
  call void @llvm.dbg.declare(metadata i32* %y.addr.i, metadata !19, metadata !DIExpression()), !dbg !20
  %1 = load i32, i32* %y.addr.i, align 4, !dbg !21
  %tobool.i = icmp ne i32 %1, 0, !dbg !21
  %cond.i = select i1 %tobool.i, i32 4, i32 7, !dbg !21
  store volatile i32 %cond.i, i32* %x, align 4, !dbg !18
  ret void, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "inline-seldag-test.c", directory: "/tmp/dbginfo")
!2 = !{}
!4 = distinct !DISubprogram(name: "func", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !0, scopeLine: 4, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "inline-seldag-test.c", directory: "/tmp/dbginfo")
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = distinct !DISubprogram(name: "f", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !9, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 1, !"Debug Info Version", i32 3}
!14 = !{!"clang version 3.5.0 "}
!15 = !DILocalVariable(name: "x", line: 5, scope: !4, file: !5, type: !16)
!16 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !11)
!17 = !DILocation(line: 5, scope: !4)
!18 = !DILocation(line: 6, column: 7, scope: !4)
!19 = !DILocalVariable(name: "y", line: 1, arg: 1, scope: !8, file: !5, type: !11)
!20 = !DILocation(line: 1, scope: !8, inlinedAt: !18)
!21 = !DILocation(line: 2, scope: !8, inlinedAt: !18)
!22 = !DILocation(line: 7, scope: !4)
