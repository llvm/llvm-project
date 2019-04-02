; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/cold-indirect-call.prof -S | FileCheck %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/cold-indirect-call.prof -S | FileCheck %s

define i32 @foo(i32 ()* %func) !dbg !3 {
; CHECK: icmp {{.*}} @bar
; CHECK-NOT: icmp {{.*}} @baz
  %call = call i32 %func(), !dbg !4
  ret i32 %call
}

define i32 @bar() !dbg !5 {
  ret i32 41, !dbg !6
}

define i32 @baz() !dbg !7 {
  ret i32 42, !dbg !8
}


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1)
!1 = !DIFile(filename: "foo.cc", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 4, unit: !0)
!4 = !DILocation(line: 5, scope: !3)
!5 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 8, unit: !0)
!6 = !DILocation(line: 9, scope: !5)
!7 = distinct !DISubprogram(name: "baz", scope: !1, file: !1, line: 12, unit: !0)
!8 = !DILocation(line: 13, scope: !7)
