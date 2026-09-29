; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: llc -filetype=obj a.ll -o a.o
; RUN: llvm-readelf -r a.o 2>err | FileCheck %s --check-prefix=RELOC
; RUN: llvm-objdump -Sl --no-show-raw-insn a.o 2>err | FileCheck %s
;; Test that ULEB128 relocs do not lead to spurious warnings.
;; https://github.com/llvm/llvm-project/issues/101544
; RUN: count 0 < err

; RELOC:      Relocation section '.rela.debug_loclists'
; RELOC:      R_RISCV_SET_ULEB128
; RELOC-NEXT: R_RISCV_SUB_ULEB128

; CHECK:      ; foo():
; CHECK-NEXT: a.c:2
; CHECK-NEXT: ; int foo(int x) {
; CHECK-NEXT:   0: addi    sp, sp, -0x10
; CHECK-NEXT:   2: sd      ra, 0x8(sp)
; CHECK-NEXT: a.c:3
; CHECK-NEXT: ; ext();
; CHECK-NEXT:   4: auipc   ra, 0x0

;--- a.c
int ext(void);
int foo(int x) {
  ext();
  return 0;
}

int ext(void);
void foo2() {
  {
    int ret = ext();
    if (__builtin_expect(ret, 0))
      ext();
  }
}
;--- gen
clang --target=riscv64-linux -S -emit-llvm -g -O1 -fdebug-compilation-dir=. a.c -o - | sed -E '/^attribute/s/,-[-0-9a-z]+//g'
;--- a.ll
; ModuleID = 'a.c'
source_filename = "a.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-linux"

; Function Attrs: nounwind uwtable
define dso_local noundef signext i32 @foo(i32 noundef signext %x) local_unnamed_addr #0 !dbg !13 {
entry:
    #dbg_value(i32 %x, !18, !DIExpression(), !19)
  %call = tail call signext i32 @ext() #2, !dbg !20
  ret i32 0, !dbg !21
}

declare !dbg !22 signext i32 @ext() local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define dso_local void @foo2() local_unnamed_addr #0 !dbg !25 {
entry:
  %call = tail call signext i32 @ext() #2, !dbg !31
    #dbg_value(i32 %call, !29, !DIExpression(), !32)
  %tobool.not = icmp eq i32 %call, 0, !dbg !33
  br i1 %tobool.not, label %if.end, label %if.then, !dbg !35, !prof !36

if.then:                                          ; preds = %entry
  %call1 = tail call signext i32 @ext() #2, !dbg !37
  br label %if.end, !dbg !37

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !38
}

attributes #0 = { nounwind uwtable "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+m,+relax,+zicsr,+zmmul" }
attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+m,+relax,+zicsr,+zmmul" }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !8, !9, !10, !11, !12}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "a.c", directory: ".", checksumkind: CSK_MD5, checksum: "4791066d0b0e4fd9c4b4df1c56f349cb")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 1, !"target-abi", !"lp64d"}
!6 = !{i32 6, !"riscv-isa", !7}
!7 = !{!"rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zmmul1p0"}
!8 = !{i32 8, !"PIC Level", i32 2}
!9 = !{i32 7, !"PIE Level", i32 2}
!10 = !{i32 7, !"uwtable", i32 2}
!11 = !{i32 8, !"SmallDataLimit", i32 8}
!12 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!13 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !14, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !17)
!14 = !DISubroutineType(types: !15)
!15 = !{!16, !16}
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !{!18}
!18 = !DILocalVariable(name: "x", arg: 1, scope: !13, file: !1, line: 2, type: !16)
!19 = !DILocation(line: 0, scope: !13)
!20 = !DILocation(line: 3, column: 3, scope: !13)
!21 = !DILocation(line: 4, column: 3, scope: !13)
!22 = !DISubprogram(name: "ext", scope: !1, file: !1, line: 1, type: !23, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!23 = !DISubroutineType(types: !24)
!24 = !{!16}
!25 = distinct !DISubprogram(name: "foo2", scope: !1, file: !1, line: 8, type: !26, scopeLine: 8, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !28)
!26 = !DISubroutineType(types: !27)
!27 = !{null}
!28 = !{!29}
!29 = !DILocalVariable(name: "ret", scope: !30, file: !1, line: 10, type: !16)
!30 = distinct !DILexicalBlock(scope: !25, file: !1, line: 9, column: 3)
!31 = !DILocation(line: 10, column: 15, scope: !30)
!32 = !DILocation(line: 0, scope: !30)
!33 = !DILocation(line: 11, column: 9, scope: !34)
!34 = distinct !DILexicalBlock(scope: !30, file: !1, line: 11, column: 9)
!35 = !DILocation(line: 11, column: 9, scope: !30)
!36 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!37 = !DILocation(line: 12, column: 7, scope: !34)
!38 = !DILocation(line: 14, column: 1, scope: !25)
