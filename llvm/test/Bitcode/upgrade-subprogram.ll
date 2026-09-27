; RUN: llvm-dis < %s.bc | FileCheck %s
; RUN: verify-uselistorder < %s.bc

; CHECK: define void @fooCLS() !dbg [[CLSSP:![0-9]+]]
define void @fooCLS() {
  ret void
}

; CHECK: define void @fooCSL() !dbg [[CSLSP:![0-9]+]]
define void @fooCSL() {
  ret void
}

; CHECK: define void @fooLCS() !dbg [[LCSSP:![0-9]+]]
define void @fooLCS() {
  ret void
}

; CHECK: define void @fooLSC() !dbg [[LSCSP:![0-9]+]]
define void @fooLSC() {
  ret void
}

; CHECK: define void @fooSCL() !dbg [[SCLSP:![0-9]+]]
define void @fooSCL() {
  ret void
}

; CHECK: define void @fooSLC() !dbg [[SLCSP:![0-9]+]]
define void @fooSLC() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}

!llvm.dbg.cu = !{!3, !6, !10, !14, !16, !20}
!1 = !DIFile(filename: "foo.c", directory: "/path/to/dir")
!2 = !DISubroutineType(types: !{})

; CHECK-DAG: [[CLSSP]] = distinct !DISubprogram{{.*}}, unit: [[CLSCU:![0-9]+]]
; CHECK-DAG: [[CLSCU]] = distinct !DICompileUnit
!3 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, subprograms: !4, emissionKind: 1)
!4 = !{!5}
!5 = !DISubprogram(file: !1, scope: !2, line: 51, name: "fooCLS", function: void ()* @fooCLS, type: !2)

; CHECK-DAG: [[CSLSP]] = distinct !DISubprogram{{.*}}, unit: [[CSLCU:![0-9]+]]
; CHECK-DAG: [[CSLCU]] = distinct !DICompileUnit
!6 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, subprograms: !8, emissionKind: 1)
!7 = !DISubprogram(file: !1, scope: !2, line: 51, name: "fooCSL", function: void ()* @fooCSL, type: !2)
!8 = !{!7}

; CHECK-DAG: [[LCSSP]] = distinct !DISubprogram{{.*}}, unit: [[LCSCU:![0-9]+]]
; CHECK-DAG: [[LCSCU]] = distinct !DICompileUnit
!9 = !{!11}
!10 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, subprograms: !9, emissionKind: 1)
!11 = !DISubprogram(file: !1, scope: !2, line: 51, name: "fooLCS", function: void ()* @fooLCS, type: !2)

; CHECK-DAG: [[LSCSP]] = distinct !DISubprogram{{.*}}, unit: [[LSCCU:![0-9]+]]
; CHECK-DAG: [[LSCCU]] = distinct !DICompileUnit
!12 = !{!13}
!13 = !DISubprogram(file: !1, scope: !2, line: 51, name: "fooLSC", function: void ()* @fooLSC, type: !2)
!14 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, subprograms: !12, emissionKind: 1)

; CHECK-DAG: [[SCLSP]] = distinct !DISubprogram{{.*}}, unit: [[SCLCU:![0-9]+]]
; CHECK-DAG: [[SCLCU]] = distinct !DICompileUnit
!15 = !DISubprogram(file: !1, scope: !2, line: 51, name: "fooSCL", function: void ()* @fooSCL, type: !2)
!16 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, subprograms: !17, emissionKind: 1)
!17 = !{!15}

; CHECK-DAG: [[SLCSP]] = distinct !DISubprogram{{.*}}, unit: [[SLCCU:![0-9]+]]
; CHECK-DAG: [[SLCCU]] = distinct !DICompileUnit
!18 = !DISubprogram(file: !1, scope: !2, line: 51, name: "fooSLC", function: void ()* @fooSLC, type: !2)
!19 = !{!18}
!20 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, subprograms: !19, emissionKind: 1)
