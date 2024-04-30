; RUN: opt -S %s -passes=sample-profile -sample-profile-file=%S/Inputs/direct-call-accurate-count.prof -salvage-stale-profile | FileCheck %s
; RUN: llvm-profdata merge --sample --extbinary --use-md5 -output=%t %S/Inputs/direct-call-accurate-count.prof
; RUN: opt -S %s -passes=sample-profile -sample-profile-file=%t -salvage-stale-profile | FileCheck %s

declare void @callee() #0

; CHECK-LABEL: @test
define dso_local void @test() #1 !dbg !3 {
  call void @callee(), !dbg !4
; CHECK: call void @callee(), !dbg !{{[0-9]+}}, !prof ![[BRANCH_WEIGHT1:[0-9]+]]
  ret void
}

; With stale profile
; CHECK-LABEL: @test2
define dso_local void @test2() #1 !dbg !5 {
  call void @callee(), !dbg !6
; CHECK: call void @callee(), !dbg !{{[0-9]+}}, !prof ![[BRANCH_WEIGHT2:[0-9]+]]
  ret void
}

attributes #0 = { "use-sample-profile" }
attributes #1 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, unit: !0)
!4 = !DILocation(line: 3, column: 4, scope: !3)
!5 = distinct !DISubprogram(name: "test2", scope: !1, file: !1, line: 11, unit: !0)
!6 = !DILocation(line: 15, column: 4, scope: !5)
; CHECK-DAG: ![[BRANCH_WEIGHT1]] = !{!"branch_weights", i32 123}
; CHECK-DAG: ![[BRANCH_WEIGHT2]] = !{!"branch_weights", i32 30}
