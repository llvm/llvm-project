; RUN: llc -emit-call-site-info -O1 -filetype=obj -o - %s | llvm-dwarfdump - | FileCheck %s

; Verify that the 64-bit call site immediates are not truncated.
;
; Reproducer for PR43525.

; Based on the following C program:
;
; #include <stdint.h>
;
; extern void foo(int64_t);
;
; int main() {
;   foo(INT64_C(0x1122334455667788));
;   foo(INT32_C(-100));
; }

; CHECK: DW_AT_GNU_call_site_value (DW_OP_constu 0x1122334455667788)
; CHECK: DW_AT_GNU_call_site_value (DW_OP_constu 0xffffffffffffff9c)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @main() !dbg !12 {
entry:
  tail call void @foo(i64 1234605616436508552), !dbg !16
  tail call void @foo(i64 -100), !dbg !17
  ret i32 0, !dbg !18
}

declare !dbg !4 void @foo(i64)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "dbgcall-site-long-imms.c", directory: "/")
!2 = !{}
!3 = !{!4}
!4 = !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !5, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{!"clang version 10.0.0"}
!12 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 5, type: !13, scopeLine: 5, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!13 = !DISubroutineType(types: !14)
!14 = !{!15}
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !DILocation(line: 6, scope: !12)
!17 = !DILocation(line: 7, scope: !12)
!18 = !DILocation(line: 8, scope: !12)
