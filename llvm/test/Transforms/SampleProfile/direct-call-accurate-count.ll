; RUN: opt -S %s -passes=sample-profile -sample-profile-file=%S/Inputs/direct-call-accurate-count.prof -salvage-stale-profile | FileCheck %s
; RUN: llvm-profdata merge --sample --extbinary --use-md5 -output=%t %S/Inputs/direct-call-accurate-count.prof
; RUN: opt -S %s -passes=sample-profile -sample-profile-file=%t -salvage-stale-profile | FileCheck %s

declare void @callee() #0

; CHECK-LABEL: @test1
define dso_local void @test1(i1 %0) #1 !dbg !3 {
; Add a branch here to prevent the head sample from being unconditionally
; propagated to the entire block overriding the line sample count.
  br i1 %0, label %if.then, label %if.end
if.then:
  call void @callee(), !dbg !4
; CHECK: call void @callee(), !dbg !{{[0-9]+}}, !prof ![[BRANCH_WEIGHT1:[0-9]+]]
  br label %if.end
if.end:
  ret void
}

; With stale profile
; CHECK-LABEL: @test2
define dso_local void @test2(i1 %0) #1 !dbg !5 {
  br i1 %0, label %if.then, label %if.end
if.then:
  call void @callee(), !dbg !6
; CHECK: call void @callee(), !dbg !{{[0-9]+}}, !prof ![[BRANCH_WEIGHT2:[0-9]+]]
  br label %if.end
if.end:
  ret void
}

; Call target is not matched in profile, use sample count.
; CHECK-LABEL: @test3
define dso_local void @test3(i1 %0) #1 !dbg !7 {
  br i1 %0, label %if.then, label %if.end
if.then:
  call void @callee(), !dbg !8
; CHECK: call void @callee(), !dbg !{{[0-9]+}}, !prof ![[BRANCH_WEIGHT3:[0-9]+]]
  br label %if.end
if.end:
  ret void
}

attributes #0 = { "use-sample-profile" }
attributes #1 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "test1", scope: !1, file: !1, line: 1, unit: !0)
!4 = !DILocation(line: 3, column: 4, scope: !3)
!5 = distinct !DISubprogram(name: "test2", scope: !1, file: !1, line: 11, unit: !0)
!6 = !DILocation(line: 15, column: 4, scope: !5)
!7 = distinct !DISubprogram(name: "test3", scope: !1, file: !1, line: 21, unit: !0)
!8 = !DILocation(line: 23, column: 4, scope: !7)
; CHECK-DAG: ![[BRANCH_WEIGHT1]] = !{!"branch_weights", i32 123}
; CHECK-DAG: ![[BRANCH_WEIGHT2]] = !{!"branch_weights", i32 30}
; CHECK-DAG: ![[BRANCH_WEIGHT3]] = !{!"branch_weights", i32 101}
