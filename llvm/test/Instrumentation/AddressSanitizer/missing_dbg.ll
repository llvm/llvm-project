; RUN: opt < %s -passes=asan -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define i32 @with_dbg(ptr %a, ptr %b) sanitize_address !dbg !3 {
entry:
  %tmp1 = load i32, ptr %a, align 4
  store i32 32, ptr %b
  ret i32 %tmp1
}
; CHECK-LABEL: @with_dbg
; CHECK-NEXT:  entry:
; CHECK:       call void @__asan_report_load4(i64 %0) #3, !dbg [[DBG:![0-9]+]]
; CHECK:       call void @__asan_report_store4(i64 %13) #3, !dbg [[DBG]]

define i32 @without_dbg(ptr %a, ptr %b) sanitize_address {
entry:
  %tmp1 = load i32, ptr %a, align 4
  store i32 32, ptr %b
  ret i32 %tmp1
}
; CHECK-LABEL: @without_dbg
; CHECK-NEXT:  entry:
; CHECK:       call void @__asan_report_load4(i64 %0) #3
; CHECK:       call void @__asan_report_store4(i64 %13) #3

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C89, file: !1, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "foo.c", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 190, type: !4, scopeLine: 192, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}

; CHECK:       [[DBG]] = !DILocation(line: 0, scope: !4)
