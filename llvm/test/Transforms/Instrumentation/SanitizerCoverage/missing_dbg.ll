; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=2 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define i32 @with_dbg(ptr %a, ptr %b) !dbg !3 {
entry:
  %tmp1 = load i32, ptr %a, align 4
  %cmp = icmp eq i32 %tmp1, 42
  br i1 %cmp, label %0, label %1
0:
  store i32 %tmp1, ptr %b
  br label %1
1:
  ret i32 %tmp1
}
; CHECK-LABEL: @with_dbg
; CHECK-NEXT:  entry:
; CHECK:       call void @__sanitizer_cov_trace_pc_guard(ptr @__sancov_gen_) #1, !dbg [[DBG1:![0-9]+]]
; CHECK:       call void @__sanitizer_cov_trace_pc_guard(ptr inttoptr (i64 add (i64 ptrtoint (ptr @__sancov_gen_ to i64), i64 4) to ptr)) #1, !dbg [[DBG2:![0-9]+]]

define i32 @without_dbg(ptr %a, ptr %b) {
entry:
  %tmp1 = load i32, ptr %a, align 4
  %cmp = icmp eq i32 %tmp1, 42
  br i1 %cmp, label %0, label %1
0:
  store i32 %tmp1, ptr %b
  br label %1
1:
  ret i32 %tmp1
}
; CHECK-LABEL: @without_dbg
; CHECK-NEXT:  entry:
; CHECK:       call void @__sanitizer_cov_trace_pc_guard(ptr @__sancov_gen_.1) #1
; CHECK:       call void @__sanitizer_cov_trace_pc_guard(ptr inttoptr (i64 add (i64 ptrtoint (ptr @__sancov_gen_.1 to i64), i64 4) to ptr)) #1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C89, file: !1, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "foo.c", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 190, type: !4, scopeLine: 192, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !DILocation(line: 192, scope: !3)
!7 = !DILocation(line: 0, scope: !3)

; CHECK:       [[DBG1]] = !DILocation(line: 192, scope: !3)
; CHECK:       [[DBG2]] = !DILocation(line: 0, scope: !3)
