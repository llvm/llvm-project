; RUN: llc -O2 %s -o - -mtriple=x86_64-unknown-linux-gnu | FileCheck %s --check-prefix=OPTS
; RUN: llc -O0 %s -o - -mtriple=x86_64-unknown-linux-gnu | FileCheck %s --check-prefix=UNOPT

;; Test that, even though there are no source locations attached to the foo
;; function, we still give it the start-of-function source location of the
;; definition line. Otherwise, this function would have no entry in the
;; line table at all.

; OPTS-LABEL: foo:
; OPTS-NEXT:   .Lfunc_begin0:
; OPTS-NEXT:   .file   0 "." "foobar.c"
; OPTS-NEXT:   .cfi_startproc
; OPTS-NEXT:   # %bb.0:
; OPTS-NEXT:   .loc    0 1 0 prologue_end
; OPTS-LABEL: bar:

define dso_local noundef i32 @foo(ptr nocapture noundef writeonly %bar) local_unnamed_addr !dbg !10 {
entry:
  store i32 0, ptr %bar, align 4
  ret i32 0
}

;; In a function with no source location, but multiple blocks, there will be
;; an opening scope-line. Test for this behaviour, and preserve the
;; unconditional branch by compiling -O0.

; UNOPT-LABEL: bar:
; UNOPT-NEXT:   .Lfunc_begin1:
; UNOPT-NEXT:   .cfi_startproc
; UNOPT-LABEL: %bb.0:
; UNOPT-NEXT:   .loc    0 11 0 prologue_end
; UNOPT-NEXT:    movq    %rdi, -8(%rsp)
; UNOPT-NEXT:    jmp     .LBB1_1
; UNOPT-LABEL: .LBB1_1:
; UNOPT-NEXT:    movq    -8(%rsp), %rax

define dso_local noundef i32 @bar(ptr nocapture noundef writeonly %baz) local_unnamed_addr !dbg !20 {
entry:
  br label %bb1
bb1:
  store i32 0, ptr %baz, align 4
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "foobar.c", directory: ".")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang"}
!10 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !15)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !14}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!15 = !{!16}
!16 = !DILocalVariable(name: "bar", arg: 1, scope: !10, file: !1, line: 1, type: !14)
!17 = !DILocation(line: 0, scope: !10)
!18 = !DILocation(line: 2, column: 8, scope: !10)
!20 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 11, type: !11, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !25)
!25 = !{!26}
!26 = !DILocalVariable(name: "bar", arg: 1, scope: !20, file: !1, line: 11, type: !14)
!27 = !DILocation(line: 0, scope: !20)
!28 = !DILocation(line: 12, column: 8, scope: !20)

