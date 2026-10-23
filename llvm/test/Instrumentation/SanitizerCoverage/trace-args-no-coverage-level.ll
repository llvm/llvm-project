; Argument and return value tracing instrument function boundaries rather than
; blocks, so they stand on their own and do not need a coverage level. A direct
; pass invocation has no level (the clang driver supplies one when it parses
; -fsanitize-coverage=), and the module pass used to bail out on SCK_None
; before reaching them -- producing a cleanly linking but entirely
; uninstrumented module.
;
; Without a level, the boundary callbacks must still be emitted, and no block
; or edge instrumentation may appear.
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-trace-args -sanitizer-coverage-trace-ret -S | FileCheck %s
;
; Asking for no mode at all still instruments nothing.
; RUN: opt < %s -passes='module(sancov-module)' -S | FileCheck %s --check-prefix=NONE
;
; Block coverage requested alongside the boundary modes still works. Note that
; a level alone is not enough: trace-args/trace-ret count as a requested mode,
; so they suppress the implicit trace-pc-guard default and it has to be asked
; for explicitly.
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-pc-guard -sanitizer-coverage-trace-args -S | FileCheck %s --check-prefix=WITHLEVEL

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i32, i64 }

; int f(struct S *s, int x)
define i32 @f(ptr %0, i32 %1) !dbg !8 {
entry:
  %s = ptrtoint ptr %0 to i64
    #dbg_value(ptr %0, !16, !DIExpression(), !19)
    #dbg_value(i32 %1, !17, !DIExpression(), !19)
  %a = load i32, ptr %0, align 8, !dbg !19
  %sum = add nsw i32 %a, %1, !dbg !19
  ret i32 %sum, !dbg !19
}

; Both arguments and the return are reported by value.
; CHECK-LABEL: define i32 @f(
; CHECK: call void @__sanitizer_cov_trace_args(i32 0, i32 8, i64 %{{[0-9]+}}, ptr null, i32 0)
; CHECK: call void @__sanitizer_cov_trace_args(i32 1, i32 4, i64 %{{[0-9]+}}, ptr null, i32 0)
; CHECK: call void @__sanitizer_cov_trace_ret(i32 4, i64 %{{[0-9]+}}, ptr null, i32 0)

; No block instrumentation came along with it.
; CHECK-NOT: call void @__sanitizer_cov_trace_pc
; CHECK-NOT: @__sancov_gen_
; CHECK-NOT: __sancov_guards

; NONE-NOT: __sanitizer_cov
; NONE-NOT: __sancov

; WITHLEVEL-LABEL: define i32 @f(
; WITHLEVEL: call void @__sanitizer_cov_trace_args(
; WITHLEVEL: call void @__sanitizer_cov_trace_pc_guard

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "a.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!6 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !10, file: !1, baseType: !4, size: 32)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !10, file: !1, baseType: !5, size: 64, offset: 64)
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !13, scopeLine: 2, unit: !0, retainedNodes: !18)
!9 = !{!6, !7}
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, line: 1, size: 128, elements: !9)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!13 = !DISubroutineType(types: !14)
!14 = !{!4, !11, !4}
!16 = !DILocalVariable(name: "s", arg: 1, scope: !8, file: !1, line: 2, type: !11)
!17 = !DILocalVariable(name: "x", arg: 2, scope: !8, file: !1, line: 2, type: !4)
!18 = !{!16, !17}
!19 = !DILocation(line: 2, column: 1, scope: !8)
