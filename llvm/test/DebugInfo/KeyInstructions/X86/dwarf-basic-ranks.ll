; RUN: llc %s --filetype=obj -o - --dwarf-use-key-instructions \
; RUN: | llvm-objdump -d - --no-show-raw-insn \
; RUN: | FileCheck %s --check-prefix=OBJ

; RUN: llc %s --filetype=obj -o - --dwarf-use-key-instructions \
; RUN: | llvm-dwarfdump - --debug-line \
; RUN: | FileCheck %s --check-prefix=DBG

; OBJ: 0000000000000000 <_Z1fPiii>:
; OBJ-NEXT:  0: pushq   %rbp
; OBJ-NEXT:  1: pushq   %r14
; OBJ-NEXT:  3: pushq   %rbx
; OBJ-NEXT:  4: movl    %edx, %ebx
; OBJ-NEXT:  6: movl    %esi, %ebp
; OBJ-NEXT:  8: movq    %rdi, %r14
; OBJ-NEXT:  b: callq   0x10 <_Z1fPiii+0x10>
; OBJ-NEXT: 10: addl    %ebx, %ebp
; OBJ-NEXT: 12: movl    %ebp, (%r14)
; OBJ-NEXT: 15: movl    %ebp, %eax
; OBJ-NEXT: 17: popq    %rbx
; OBJ-NEXT: 18: popq    %r14
; OBJ-NEXT: 1a: popq    %rbp

; DBG:      Address            Line   Column File   ISA Discriminator OpIndex Flags
; DBG-NEXT: ------------------ ------ ------ ------ --- ------------- ------- -------------
; DBG-NEXT: 0x0000000000000000      3      0      0   0             0       0  is_stmt
; DBG-NEXT: 0x000000000000000b      4      0      0   0             0       0  is_stmt prologue_end
; DBG-NEXT: 0x0000000000000010      6      0      0   0             0       0
; DBG-NEXT: 0x0000000000000012      5      0      0   0             0       0  is_stmt
; DBG-NEXT: 0x0000000000000015      7      0      0   0             0       0  is_stmt
; DBG-NEXT: 0x0000000000000017      7      0      0   0             0       0  epilogue_begin
; DBG-NEXT: 0x000000000000001c      7      0      0   0             0       0  end_sequence

;; 1. [[gnu::nodebug]] void prologue_end();
;; 2.
;; 3. int f(int *a, int b, int c) {
;; 4.   prologue_end();
;; 5.   *a = 
;; 6.     b + c;
;; 7.   return *a;
;; 8. }

;; The add and store are in the same goup (1). The add (line 6) has lower
;; precedence (rank 2) so should not get is_stmt applied.
target triple = "x86_64-unknown-linux-gnu"

define hidden noundef i32 @_Z1fPiii(ptr %a, i32 %b, i32 %c) local_unnamed_addr !dbg !11 {
entry:
  tail call void @_Z12prologue_endv(), !dbg !DILocation(line: 4, scope: !11)
  %add = add nsw i32 %c, %b,           !dbg !DILocation(line: 6, scope: !11, atomGroup: 1, atomRank: 2)
  store i32 %add, ptr %a, align 4,     !dbg !DILocation(line: 5, scope: !11, atomGroup: 1, atomRank: 1)
  ret i32 %add,                        !dbg !DILocation(line: 7, scope: !11, atomGroup: 2, atomRank: 1)
}

declare void @_Z12prologue_endv() local_unnamed_addr #1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_17, file: !1, producer: "clang version 19.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 19.0.0"}
!11 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 3, type: !12, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DISubroutineType(types: !13)
!13 = !{}
