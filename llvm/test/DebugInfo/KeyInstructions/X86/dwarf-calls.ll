; RUN: llc %s --filetype=obj -o - --dwarf-use-key-instructions \
; RUN: | llvm-objdump -d - --no-show-raw-insn \
; RUN: | FileCheck %s --check-prefix=OBJ

; RUN: llc %s --filetype=obj -o - --dwarf-use-key-instructions \
; RUN: | llvm-dwarfdump - --debug-line \
; RUN: | FileCheck %s --check-prefix=DBG

; OBJ:0000000000000000 <fun>:
; OBJ-NEXT:  0:       pushq   %rbp
; OBJ-NEXT:  1:       pushq   %r14
; OBJ-NEXT:  3:       pushq   %rbx
; OBJ-NEXT:  4:       movq    (%rip), %rax
; OBJ-NEXT:  b:       movl    (%rax), %ebp
; OBJ-NEXT:  d:       callq   0x12 <fun+0x12>
; OBJ-NEXT: 12:       callq   0x17 <fun+0x17>
; OBJ-NEXT: 17:       movl    %eax, %ebx
; OBJ-NEXT: 19:       addl    %ebp, %ebx
; OBJ-NEXT: 1b:       movq    (%rip), %r14
; OBJ-NEXT: 22:       movl    $0x1, (%r14)
; OBJ-NEXT: 29:       callq   0x2e <fun+0x2e>
; OBJ-NEXT: 2e:       movl    $0x2, (%r14)
; OBJ-NEXT: 35:       callq   0x3a <fun+0x3a>
; OBJ-NEXT: 3a:       movl    $0x3, (%r14)
; OBJ-NEXT: 41:       callq   0x46 <fun+0x46>
; OBJ-NEXT: 46:       movl    $0x4, (%r14)
; OBJ-NEXT: 4d:       callq   0x52 <fun+0x52>
; OBJ-NEXT: 52:       movl    %ebx, %eax
; OBJ-NEXT: 54:       popq    %rbx
; OBJ-NEXT: 55:       popq    %r14
; OBJ-NEXT: 57:       popq    %rbp
; OBJ-NEXT: 58:       retq

; DBG:      Address            Line   Column File   ISA Discriminator OpIndex Flags
; DBG-NEXT: ------------------ ------ ------ ------ --- ------------- ------- -------------
; DBG-NEXT: 0x0000000000000000      1      0      0   0             0       0  is_stmt
; DBG-NEXT: 0x0000000000000004      2      0      0   0             0       0  is_stmt prologue_end

;; Test A:
;; Check the 1st call (line 3) gets is_stmt despite having no atom group.
; DBG-NEXT: 0x000000000000000d      3      0      0   0             0       0  is_stmt

;; Test B:
;; Check the 2nd call (line 4) gets is_stmt applied despite being part of group
;; 1 and having lower precedence than the add. Check that the add stil gets
;; is_stmt applied.
;; There are two is_stmt line 4 entries because we don't float
;; is_stmts up on the same line past other key instructions. The call is
;; key, so the add's is_stmt floats up to the movl on the same line, but
;; not past the call.
; DBG-NEXT: 0x0000000000000012      4      0      0   0             0       0  is_stmt
; DBG-NEXT: 0x0000000000000017      4      0      0   0             0       0  is_stmt

;; Test C:
;; Check that is_stmt floats up from the call (0x29) to the store (0x1b).
; DBG-NEXT: 0x000000000000001b      5      0      0   0             0       0  is_stmt

;; Test D:
;; Check the is_stmt is not applied to the lower ranking instruction (0x2e).
; DBG-NEXT: 0x000000000000002e      6      0      0   0             0       0
; DBG-NEXT: 0x0000000000000035      7      0      0   0             0       0  is_stmt

;; Test E:
;; Check the is_stmt floats up to an instruction in the same group of the same
;; or lower precedence (from call, 0x41, to `store 3`, 0x3a).
; DBG-NEXT: 0x000000000000003a      8      0      0   0             0       0  is_stmt
; DBG-NEXT: 0x0000000000000046      9      0      0   0             0       0  is_stmt

; DBG-NEXT: 0x0000000000000052     10      0      0   0             0       0
; DBG-NEXT: 0x0000000000000054     10      0      0   0             0       0  epilogue_begin
; DBG-NEXT: 0x0000000000000059     10      0      0   0             0       0  end_sequence

target triple = "x86_64-unknown-linux-gnu"

@a = global i32 0
@z = global i32 0

define hidden i32 @fun() local_unnamed_addr !dbg !11 {
entry:
  %b = load i32, ptr @a,   !dbg !DILocation(line: 2, scope: !11)
;; Test A:
  tail call void @f(),     !dbg !DILocation(line: 3, scope: !11)
;; Test B:
  %x = tail call i32 @g(), !dbg !DILocation(line: 4, scope: !11, atomGroup: 1, atomRank: 2)
  %y = add i32 %x, %b,     !dbg !DILocation(line: 4, scope: !11, atomGroup: 1, atomRank: 1)
;; Test C:
  store i32 1, ptr @z,     !dbg !DILocation(line: 5, scope: !11, atomGroup: 2, atomRank: 2)
  tail call void @f(),     !dbg !DILocation(line: 5, scope: !11, atomGroup: 2, atomRank: 1)
;; Test D:
  store i32 2, ptr @z,     !dbg !DILocation(line: 6, scope: !11, atomGroup: 3, atomRank: 2)
  tail call void @f(),     !dbg !DILocation(line: 7, scope: !11, atomGroup: 3, atomRank: 1)
;; Test E:
  store i32 3, ptr @z,     !dbg !DILocation(line: 8, scope: !11, atomGroup: 4, atomRank: 2)
  tail call void @f(),     !dbg !DILocation(line: 8, scope: !11, atomGroup: 4, atomRank: 1)
  store i32 4, ptr @z,     !dbg !DILocation(line: 9, scope: !11, atomGroup: 5, atomRank: 1)
  tail call void @f(),     !dbg !DILocation(line: 9, scope: !11, atomGroup: 5, atomRank: 1)
  ret i32 %y,              !dbg !DILocation(line: 10, scope: !11)
}

declare void @f() local_unnamed_addr
declare i32  @g() local_unnamed_addr

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_17, file: !1, producer: "clang version 19.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 19.0.0"}
!11 = distinct !DISubprogram(name: "fun", scope: !1, file: !1, line: 1, type: !12, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DISubroutineType(types: !13)
!13 = !{}
