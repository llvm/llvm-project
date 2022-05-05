; RUN: opt < %s -passes=tsan -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define i32 @with_dbg(i32* %a) sanitize_thread !dbg !3 {
entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}
; CHECK-LABEL: @with_dbg
; CHECK-NEXT:  entry:
; CHECK:       call void @__tsan_func_entry(i8* %0), !dbg [[DBG:![0-9]+]]
; CHECK:       call void @__tsan_read4(i8* %1), !dbg [[DBG]]
; CHECK:       call void @__tsan_func_exit(), !dbg [[DBG]]

define i32 @without_dbg(i32* %a) sanitize_thread {
entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}
; CHECK-LABEL: @without_dbg
; CHECK-NEXT:  entry:
; CHECK-NOT:   call void @__tsan_func_entry(i8* %0), !dbg
; CHECK-NOT:   call void @__tsan_read4(i8* %1), !dbg
; CHECK-NOT:   call void @__tsan_func_exit(), !dbg
; CHECK:       call void @__tsan_func_entry(i8* %0)
; CHECK:       call void @__tsan_read4(i8* %1)
; CHECK:       call void @__tsan_func_exit()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C89, file: !1, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "foo.c", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 190, type: !4, scopeLine: 192, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}

; CHECK:       [[DBG]] = !DILocation(line: 0, scope: !3)
