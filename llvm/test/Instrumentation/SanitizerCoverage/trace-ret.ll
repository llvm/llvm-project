; Return value tracing: one __sanitizer_cov_trace_ret call before every return.
; A scalar is reported as its value, a pointer to a struct as the address of
; that struct together with its field offset table, and a struct returned
; indirectly through the caller's buffer.
;
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-ret -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i32, i64 }
%struct.Big = type { i64, i64, i64 }

; CHECK-NOT: alloca

; A pointer return whose pointee fields are known is reported as the address of
; the object, so a consumer can read the fields out of it.
define ptr @ret_struct_ptr(ptr %s) !dbg !13 {
entry:
  ret ptr %s
}
; CHECK-LABEL: define ptr @ret_struct_ptr(
; CHECK: %[[ADDR:[0-9]+]] = ptrtoint ptr %s to i64
; CHECK: call void @__sanitizer_cov_trace_ret(i64 ptrtoint (ptr @ret_struct_ptr to i64), i32 16, i64 %[[ADDR]], ptr @__sancov_offsets_, i32 2)
; CHECK: ret ptr %s

; A scalar return is reported as its value, once per return.
define i32 @ret_scalar(i1 %c, i32 %x) !dbg !16 {
entry:
  br i1 %c, label %yes, label %no
yes:
  ret i32 %x
no:
  ret i32 0
}
; CHECK-LABEL: define i32 @ret_scalar(
; CHECK: %[[X:[0-9]+]] = zext i32 %x to i64
; CHECK: call void @__sanitizer_cov_trace_ret(i64 ptrtoint (ptr @ret_scalar to i64), i32 4, i64 %[[X]], ptr null, i32 0)
; CHECK: ret i32 %x
; CHECK: call void @__sanitizer_cov_trace_ret(i64 ptrtoint (ptr @ret_scalar to i64), i32 4, i64 0, ptr null, i32 0)
; CHECK: ret i32 0

; A struct returned by value may be lowered to an indirect return: the IR
; function returns void and writes the result through a hidden struct-return
; pointer. That buffer is reported as the return value, with the size and the
; fields of the source struct, so the return is not dropped.
define void @ret_sret(ptr sret(%struct.Big) %0) !dbg !19 {
entry:
  ret void
}
; CHECK-LABEL: define void @ret_sret(
; CHECK: %[[BUF:[0-9]+]] = ptrtoint ptr %0 to i64
; CHECK: call void @__sanitizer_cov_trace_ret(i64 ptrtoint (ptr @ret_sret to i64), i32 24, i64 %[[BUF]], ptr @__sancov_offsets_.1, i32 3)
; CHECK: ret void

; A struct small enough to come back in registers has no address, so it is
; reported as one call per register, the same way the argument side reports a
; struct the ABI split across registers.
define { i64, i64 } @ret_in_registers(i64 %a, i64 %b) !dbg !30 {
entry:
  %0 = insertvalue { i64, i64 } poison, i64 %a, 0
  %1 = insertvalue { i64, i64 } %0, i64 %b, 1
  ret { i64, i64 } %1
}
; CHECK-LABEL: define { i64, i64 } @ret_in_registers(
; CHECK: %[[LO:[0-9]+]] = extractvalue { i64, i64 } %1, 0
; CHECK: %[[HI:[0-9]+]] = extractvalue { i64, i64 } %1, 1
; CHECK: call void @__sanitizer_cov_trace_ret(i64 ptrtoint (ptr @ret_in_registers to i64), i32 8, i64 %[[LO]], ptr null, i32 0)
; CHECK: call void @__sanitizer_cov_trace_ret(i64 ptrtoint (ptr @ret_in_registers to i64), i32 8, i64 %[[HI]], ptr null, i32 0)

; A void return has nothing to report.
define void @ret_void() !dbg !22 {
entry:
  ret void
}
; CHECK-LABEL: define void @ret_void(
; CHECK: call void @__sanitizer_cov_trace_ret(i64 ptrtoint (ptr @ret_void to i64), i32 0, i64 0, ptr null, i32 0)
; CHECK: ret void

; A musttail call has to stay adjacent to the return that forwards it, so there
; is nowhere to put the call and the return is left alone.
declare i32 @tail_callee(i32)
define i32 @ret_musttail(i32 %x) !dbg !24 {
entry:
  %r = musttail call i32 @tail_callee(i32 %x)
  ret i32 %r
}
; CHECK-LABEL: define i32 @ret_musttail(
; CHECK-NOT: call void @__sanitizer_cov_trace_ret(i64 ptrtoint (ptr @ret_musttail to i64)
; CHECK: ret i32 %r

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, isOptimized: true, emissionKind: FullDebug)
!1 = !{i32 2, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "ret.c", directory: "/")
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)

; struct S { int a; long b; }
!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, size: 128, elements: !7)
!7 = !{!8, !9}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !6, file: !3, baseType: !4, size: 32)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !6, file: !3, baseType: !5, size: 64, offset: 64)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)

; struct Big { long a; long b; long c; }
!11 = !DICompositeType(tag: DW_TAG_structure_type, name: "Big", file: !3, size: 192, elements: !12)
!12 = !{!27, !28, !29}

!13 = distinct !DISubprogram(name: "ret_struct_ptr", scope: !3, file: !3, line: 1, type: !14, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !31)
!14 = !DISubroutineType(types: !15)
!15 = !{!10, !10}

!16 = distinct !DISubprogram(name: "ret_scalar", scope: !3, file: !3, line: 5, type: !17, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !31)
!17 = !DISubroutineType(types: !18)
!18 = !{!4, !4, !4}

!19 = distinct !DISubprogram(name: "ret_sret", scope: !3, file: !3, line: 9, type: !20, scopeLine: 9, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !31)
!20 = !DISubroutineType(types: !21)
!21 = !{!11}

!22 = distinct !DISubprogram(name: "ret_void", scope: !3, file: !3, line: 13, type: !23, scopeLine: 13, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !31)
!23 = !DISubroutineType(types: !{null})

!24 = distinct !DISubprogram(name: "ret_musttail", scope: !3, file: !3, line: 17, type: !25, scopeLine: 17, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !31)
!25 = !DISubroutineType(types: !26)
!26 = !{!4, !4}

!27 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !11, file: !3, baseType: !5, size: 64)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !11, file: !3, baseType: !5, size: 64, offset: 64)
!29 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !11, file: !3, baseType: !5, size: 64, offset: 128)

!30 = distinct !DISubprogram(name: "ret_in_registers", scope: !3, file: !3, line: 21, type: !32, scopeLine: 21, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !31)
!31 = !{}
!32 = !DISubroutineType(types: !{!6, !5, !5})
