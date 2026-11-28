; RUN: opt -passes=attributor -S %s | FileCheck %s

; The callbacks' return types differ from the indirect calls in dispatch.
; Replacing those calls with direct calls must preserve their debug locations,
; both when there is one possible callee and when there are multiple callees.

declare i32 @callback1()
declare i32 @callback2()

define internal void @dispatch(ptr %fn1, ptr %fn2) !dbg !2 {
; CHECK-LABEL: define internal void @dispatch(
; CHECK-SAME: !dbg ![[SP:[0-9]+]]
; CHECK: call void @callback1(), !dbg ![[LOC:[0-9]+]]
; CHECK-DAG: call void @callback1(), !dbg ![[LOC2:[0-9]+]]
; CHECK-DAG: call void @callback2(), !dbg ![[LOC2]]
entry:
  call void %fn1(), !dbg !3
  call void %fn2(), !dbg !4
  ret void
}

define void @foo() {
entry:
  call void @dispatch(ptr @callback1, ptr @callback1)
  call void @dispatch(ptr @callback1, ptr @callback2)
  ret void
}

; CHECK: ![[LOC]] = !DILocation(line: 1, scope: ![[SP]])
; CHECK: ![[LOC2]] = !DILocation(line: 2, scope: ![[SP]])

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1)
!1 = !DIFile(filename: "indirect-call-debug-location.ll", directory: ".")
!2 = distinct !DISubprogram(name: "dispatch", file: !1, type: !DISubroutineType(types: !{}), unit: !0)
!3 = !DILocation(line: 1, scope: !2)
!4 = !DILocation(line: 2, scope: !2)
!5 = !{i32 2, !"Debug Info Version", i32 3}
