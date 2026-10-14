; RUN: opt %s -S -passes=dse -o - | FileCheck %s --implicit-check-not="#dbg_"

;; The IR is clang's, with only the cleanup listed below. The variable fills its
;; alloca, so the source byte offsets can be checked directly against the test.
;;
;; $ cat shorten.c
;; void esc(char *);
;; void shortenBeginPartial(void) {
;;   char local[80];
;;   __builtin_memset(local + 8, 0, 72);
;;   __builtin_memset(local + 4, 8, 64);
;;   esc(local);
;; }
;;
;; $ clang -O2 -g -gno-key-instructions -c \
;;       -Xclang -fexperimental-assignment-tracking=forced \
;;       -mllvm -print-before=dse -mllvm -print-module-scope shorten.c \
;;       -o /dev/null
;;
;; and then, by hand: dropped the target triple and datalayout so this runs
;; anywhere, dropped the tbaa metadata, llvm.ident, the producer/checksum/
;; sysroot strings, the lifetime intrinsics and the function attributes, and
;; renumbered the metadata. The remaining non-debug instructions keep clang's
;; order and operands, and the assignment offsets are unchanged.

;; 'local' fills its 80-byte alloca, so an offset into the alloca is also an
;; offset into the variable.
;;
;; Check that the dead fragment describes exactly what DSE removes from the
;; first memset. That memset writes 72 bytes starting eight bytes into 'local'.
;; The second memset overwrites the first 60 of those bytes (bytes 8 through
;; 67), so DSE keeps the final 12 bytes, starting at byte 68. The removed part
;; starts at bit 64 of the variable and is 480 bits long.
;;
;; DW_OP_LLVM_fragment stores the starting bit followed by the size, so the
;; CHECK expects (64, 480). Make sure the starting bit is 64 so that we know
;; the dead range begins at byte 8 of 'local', where the overwritten part
;; starts.

; CHECK: @shortenBeginPartial
; CHECK:      #dbg_assign({{.*}}, ![[VAR:[0-9]+]], !DIExpression(), {{.*}}, ptr %local, !DIExpression(),
; CHECK:      call void @llvm.memset{{.*}}, !DIAssignID ![[ID:[0-9]+]]
; CHECK-NEXT: #dbg_assign(i8 0, ![[VAR]], !DIExpression(DW_OP_LLVM_fragment, 64, 576), ![[ID]], ptr %add.ptr, !DIExpression(),
; CHECK-NEXT: #dbg_assign(i8 0, ![[VAR]], !DIExpression(DW_OP_LLVM_fragment, 64, 480), ![[UniqueID:[0-9]+]], ptr poison, !DIExpression(),
; CHECK:      call void @llvm.memset{{.*}}, !DIAssignID ![[ID2:[0-9]+]]
; CHECK-NEXT: #dbg_assign(i1 poison, ![[VAR]], !DIExpression(DW_OP_LLVM_fragment, 32, 512), ![[ID2]], ptr %add.ptr2, !DIExpression(),

; CHECK-DAG: ![[ID]] = distinct !DIAssignID()
; CHECK-DAG: ![[ID2]] = distinct !DIAssignID()
; CHECK-DAG: ![[UniqueID]] = distinct !DIAssignID()

define void @shortenBeginPartial() !dbg !7 {
entry:
  %local = alloca [80 x i8], align 1, !DIAssignID !13
    #dbg_assign(i1 poison, !11, !DIExpression(), !13, ptr %local, !DIExpression(), !14)
  %add.ptr = getelementptr inbounds nuw i8, ptr %local, i64 8, !dbg !15
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(72) %add.ptr, i8 0, i64 72, i1 false), !dbg !15, !DIAssignID !16
    #dbg_assign(i8 0, !11, !DIExpression(DW_OP_LLVM_fragment, 64, 576), !16, ptr %add.ptr, !DIExpression(), !14)
  %add.ptr2 = getelementptr inbounds nuw i8, ptr %local, i64 4, !dbg !17
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(64) %add.ptr2, i8 8, i64 64, i1 false), !dbg !17, !DIAssignID !18
    #dbg_assign(i1 poison, !11, !DIExpression(DW_OP_LLVM_fragment, 32, 512), !18, ptr %add.ptr2, !DIExpression(), !14)
  call void @esc(ptr noundef nonnull %local), !dbg !19
  ret void, !dbg !20
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare void @esc(ptr noundef)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !21}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false)
!1 = !DIFile(filename: "shorten.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "shortenBeginPartial", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "local", scope: !7, file: !1, line: 3, type: !12)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, size: 640, elements: !5)
!4 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!5 = !{!6}
!6 = !DISubrange(count: 80)
!13 = distinct !DIAssignID()
!14 = !DILocation(line: 0, scope: !7)
!15 = !DILocation(line: 4, column: 3, scope: !7)
!16 = distinct !DIAssignID()
!17 = !DILocation(line: 5, column: 3, scope: !7)
!18 = distinct !DIAssignID()
!19 = !DILocation(line: 6, column: 3, scope: !7)
!20 = !DILocation(line: 7, column: 1, scope: !7)
!21 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
