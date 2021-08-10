; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/indirect-call-csspgo.prof -sample-profile-icp-relative-hotness=1 -pass-remarks=sample-profile -S -o /dev/null 2>&1 | FileCheck -check-prefix=ICP-ALL %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/indirect-call-csspgo.prof -sample-profile-icp-relative-hotness=1  -pass-remarks=sample-profile -S -o /dev/null 2>&1 | FileCheck -check-prefix=ICP-ALL %s
; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/indirect-call-csspgo.prof -sample-profile-icp-relative-hotness=1  -pass-remarks=sample-profile -sample-profile-inline-size=0 -S -o /dev/null 2>&1 | FileCheck -check-prefix=ICP-HOT %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/indirect-call-csspgo.prof -sample-profile-icp-relative-hotness=1  -pass-remarks=sample-profile -sample-profile-inline-size=0 -S -o /dev/null 2>&1 | FileCheck -check-prefix=ICP-HOT %s

define void @test(void ()*) #0 !dbg !3 {
;; Add two direct call to force top-down order for sample profile loader
  call void @_Z3foov(), !dbg !7
  call void @_Z3barv(), !dbg !7
  call void @_Z3bazv(), !dbg !7
  %2 = alloca void ()*
  store void ()* %0, void ()** %2
  %3 = load void ()*, void ()** %2
  call void %3(), !dbg !4
  %4 = alloca void ()*
  store void ()* %0, void ()** %4
  %5 = load void ()*, void ()** %4
  call void %5(), !dbg !5
  ret void
}

define void @_Z3foov() #0 !dbg !8 {
  ret void
}

define void @_Z3barv() #0 !dbg !9 {
  ret void
}

define void @_Z3bazv() #0 !dbg !10 {
  ret void
}

define void @_Z3zoov() #0 !dbg !11 {
  ret void
}

attributes #0 = {"use-sample-profile"}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1)
!1 = !DIFile(filename: "test.cc", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 3, unit: !0)
!4 = !DILocation(line: 4, scope: !3)
!5 = !DILocation(line: 5, scope: !3)
!6 = !DILocation(line: 6, scope: !3)
!7 = !DILocation(line: 7, scope: !3)
!8 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 29, unit: !0)
!9 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 32, unit: !0)
!10 = distinct !DISubprogram(name: "baz", linkageName: "_Z3bazv", scope: !1, file: !1, line: 24, unit: !0)
!11 = distinct !DISubprogram(name: "zoo", linkageName: "_Z3zoov", scope: !1, file: !1, line: 24, unit: !0)


; ICP-ALL: remark: test.cc:5:0: '_Z3bazv' inlined into 'test'
; ICP-ALL-NEXT: remark: test.cc:4:0: '_Z3foov' inlined into 'test'
; ICP-ALL-NEXT: remark: test.cc:4:0: '_Z3barv' inlined into 'test'
; ICP-ALL-NOT: remark

; ICP-HOT: remark: test.cc:4:0: '_Z3foov' inlined into 'test'
; ICP-HOT-NOT: remark
