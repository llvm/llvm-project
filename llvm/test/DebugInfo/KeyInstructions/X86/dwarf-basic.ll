; RUN: llc %s --filetype=obj -o - --dwarf-use-key-instructions \
; RUN: | llvm-objdump -d - --no-show-raw-insn \
; RUN: | FileCheck %s --check-prefix=OBJ

; RUN: llc %s --filetype=obj -o - --dwarf-use-key-instructions \
; RUN: | llvm-dwarfdump - --debug-line \
; RUN: | FileCheck %s --check-prefix=DBG

;; 1. int f(int a) {
;; 2.   int x = a + 1;
;; 3.   return x;
;; 4. }
;; 5. int g(int b) {
;; 6.   return f(b);
;; 7. }
;;
;; Both functions contain 2 instructions in unique atom groups. In f we see
;; groups 1 and 3, and in g we see {!18, 1} and 1. All of these instructions
;; should get is_stmt.

; OBJ: 0000000000000000 <_Z1fi>:
; OBJ-NEXT: 0: leal    0x1(%rdi), %eax
; OBJ-NEXT: 3: retq
; OBJ: 0000000000000010 <_Z1gi>:
; OBJ-NEXT: 10: leal    0x1(%rdi), %eax
; OBJ-NEXT: 13: retq

; DBG:      Address            Line   Column File   ISA Discriminator OpIndex Flags
; DBG-NEXT: ------------------ ------ ------ ------ --- ------------- ------- -------------
; DBG-NEXT: 0x0000000000000000      2      0      0   0             0       0  is_stmt prologue_end
; DBG-NEXT: 0x0000000000000003      3      0      0   0             0       0  is_stmt
; DBG-NEXT: 0x0000000000000010      2      0      0   0             0       0  is_stmt prologue_end
; DBG-NEXT: 0x0000000000000013      6      0      0   0             0       0  is_stmt

target triple = "x86_64-unknown-linux-gnu"

define hidden noundef i32 @_Z1fi(i32 noundef %a) local_unnamed_addr !dbg !11 {
entry:
  %add = add nsw i32 %a, 1,   !dbg !DILocation(line: 2, scope: !11, atomGroup: 1, atomRank: 2)
  ret i32 %add,               !dbg !DILocation(line: 3, scope: !11, atomGroup: 3, atomRank: 1)
}

define hidden noundef i32 @_Z1gi(i32 noundef %b) local_unnamed_addr !dbg !16 {
entry:
  %add.i = add nsw i32 %b, 1, !dbg !DILocation(line: 2, scope: !11, inlinedAt: !18, atomGroup: 1, atomRank: 2)
  ret i32 %add.i,             !dbg !DILocation(line: 6, scope: !16, atomGroup: 1, atomRank: 1)
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_17, file: !1, producer: "clang version 19.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 19.0.0"}
!11 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !12, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DISubroutineType(types: !13)
!13 = !{}
!16 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 5, type: !12, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!18 = distinct !DILocation(line: 6, scope: !16)
