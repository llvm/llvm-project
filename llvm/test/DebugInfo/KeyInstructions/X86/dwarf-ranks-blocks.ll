; RUN: llc %s --filetype=obj -o - --dwarf-use-key-instructions \
; RUN: | llvm-objdump -d - --no-show-raw-insn \
; RUN: | FileCheck %s --check-prefix=OBJ

; RUN: llc %s --filetype=obj -o - --dwarf-use-key-instructions \
; RUN: | llvm-dwarfdump - --debug-line \
; RUN: | FileCheck %s --check-prefix=DBG

;; Hand written. The stores and add are in the same atom group. Check that,
;; despite each instruction belonging to a separate block, the stores (line 3)
;; get is_stmt (both rank 1) and the add (line 2) does not (it's rank 2).

; OBJ:      16: addl    %ebp, %ebx
; OBJ-NEXT: 18: testb   $0x1, %r15b
; OBJ-NEXT: 1c: je      0x23 <_Z1fPiii+0x23>
; OBJ-NEXT: 1e: movl    %ebx, (%r14)
; OBJ-NEXT: 21: jmp     0x26 <_Z1fPiii+0x26>
; OBJ-NEXT: 23: movl    %ebp, (%r14)

; DBG:      Address            Line   Column File   ISA Discriminator OpIndex Flags
; DBG-NEXT: ------------------ ------ ------ ------ --- ------------- ------- -------------
; DBG-NEXT: 0x0000000000000000      1      0      0   0             0       0  is_stmt
; DBG-NEXT: 0x0000000000000011      1      0      0   0             0       0  is_stmt prologue_end
;; The add: no is_stmt
; DBG-NEXT: 0x0000000000000016      2      0      0   0             0       0
; DBG-NEXT: 0x0000000000000018      1      0      0   0             0       0
;; Both stores: is_stmt
; DBG-NEXT: 0x000000000000001e      3      0      0   0             0       0  is_stmt
; DBG-NEXT: 0x0000000000000023      3      0      0   0             0       0  is_stmt
; DBG-NEXT: 0x0000000000000026      1      0      0   0             0       0
; DBG-NEXT: 0x0000000000000028      1      0      0   0             0       0  epilogue_begin
; DBG-NEXT: 0x0000000000000033      1      0      0   0             0       0  end_sequence

target triple = "x86_64-unknown-linux-gnu"

define hidden noundef i32 @_Z1fPiii(ptr %a, i32 %b, i32 %c, i1 %cond) local_unnamed_addr !dbg !11 {
entry:
  tail call void @_Z12prologue_endv(), !dbg !DILocation(line: 1, scope: !11)
  %add = add nsw i32 %c, %b,           !dbg !DILocation(line: 2, scope: !11, atomGroup: 1, atomRank: 2)
  br i1 %cond, label %bb1, label %bb2, !dbg !DILocation(line: 1, scope: !11)

bb1:
  store i32 %add, ptr %a, align 4,     !dbg !DILocation(line: 3, scope: !11, atomGroup: 1, atomRank: 1)
  ret i32 %add,                        !dbg !DILocation(line: 1, scope: !11)

bb2:
  store i32 %b, ptr %a, align 4,       !dbg !DILocation(line: 3, scope: !11, atomGroup: 1, atomRank: 1)
  store i32 %c, ptr %a, align 4,       !dbg !DILocation(line: 3, scope: !11, atomGroup: 1, atomRank: 1)
  ret i32 %add,                        !dbg !DILocation(line: 1, scope: !11)
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
!11 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !12, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DISubroutineType(types: !13)
!13 = !{}
