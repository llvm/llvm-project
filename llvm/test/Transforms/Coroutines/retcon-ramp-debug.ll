; REQUIRES: aarch64-registered-target
;
; Splitting directly is the control: both the original entry and its
; continuation have source-level debug information. Separating the entry from
; the body must preserve the entry's subprogram when the body is inlined away.
; Check the emitted DWARF, not just a surviving continuation's IR metadata.
;
; RUN: opt -verify-each -passes='cgscc(coro-split),coro-cleanup,globaldce' %s -o %t.control.bc
; RUN: llc -filetype=obj %t.control.bc -o %t.control.o
; RUN: llvm-dwarfdump --debug-info %t.control.o | FileCheck %s --check-prefix=DWARF
; RUN: opt -verify-each -passes='coro-early,cgscc(coro-split),coro-cleanup,globaldce' %s -o %t.split.bc
; RUN: llc -filetype=obj %t.split.bc -o %t.split.o
; RUN: llvm-dwarfdump --debug-info %t.split.o | FileCheck %s --check-prefix=DWARF
; RUN: opt -verify-each -passes='default<O2>' %s -o %t.o2.bc
; RUN: llc -filetype=obj %t.o2.bc -o %t.o2.o
; RUN: llvm-dwarfdump --debug-info %t.o2.o | FileCheck %s --check-prefix=DWARF
;
; DWARF: DW_AT_linkage_name ("yield_once")
; DWARF-NEXT: DW_AT_name ("yield_once")
; DWARF-NEXT: DW_AT_decl_file ("{{.*}}retcon-ramp-debug.swift")
; DWARF-NEXT: DW_AT_decl_line (1)

source_filename = "retcon-ramp-debug.swift"
target triple = "aarch64-unknown-linux-gnu"

define { ptr, i32 } @yield_once(ptr %buffer) presplitcoroutine !dbg !5 {
entry:
  %id = call token @llvm.coro.id.retcon.once(i32 16, i32 8, ptr %buffer,
      ptr @prototype, ptr @malloc, ptr @free), !dbg !9
    #dbg_value(ptr %buffer, !12, !DIExpression(), !9)
  %handle = call ptr @llvm.coro.begin(token %id, ptr null), !dbg !9
  %unwind = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 7), !dbg !10
  br i1 %unwind, label %cleanup, label %cleanup, !dbg !10

cleanup:
  call void @llvm.coro.end(ptr %handle, i1 false, token none), !dbg !11
  unreachable, !dbg !11
}

declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare void @llvm.coro.end(ptr, i1, token)
declare void @prototype(ptr, i1)
declare ptr @malloc(i64)
declare void @free(ptr)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!0 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !1, producer: "regression", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "retcon-ramp-debug.swift", directory: "/test")
!3 = !{null, !13}
!4 = !DISubroutineType(types: !3)
!5 = distinct !DISubprogram(name: "yield_once", linkageName: "yield_once", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !DILocation(line: 2, column: 1, scope: !5)
!10 = !DILocation(line: 3, column: 1, scope: !5)
!11 = !DILocation(line: 4, column: 1, scope: !5)
!12 = !DILocalVariable(name: "buffer", arg: 1, scope: !5, file: !1, line: 1, type: !13)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!14 = !{!12}
