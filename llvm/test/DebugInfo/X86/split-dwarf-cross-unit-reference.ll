; RUN: llc -mtriple=x86_64-linux -split-dwarf-file=foo.dwo -filetype=obj -o - < %s | llvm-objdump -r - | FileCheck %s

; CHECK-NOT: .rel{{a?}}.debug_info.dwo
; CHECK: RELOCATION RECORDS FOR [.rel{{a?}}.debug_info]:
; CHECK-NOT: RELOCATION RECORDS
; Expect one relocation in debug_info, between f3 and f1.
; CHECK: R_X86_64_32 .debug_info
; CHECK-NOT: .debug_info
; CHECK: RELOCATION RECORDS
; CHECK-NOT: .rel{{a?}}.debug_info.dwo


; Function Attrs: noinline nounwind optnone uwtable
define void @_Z2f1v() !dbg !7 {
entry:
  ret void, !dbg !10
}

; Function Attrs: noinline uwtable
define void @_Z2f3v() !dbg !13 {
entry:
  call void @_Z2f1v(), !dbg !14
  ret void, !dbg !16
}

!llvm.dbg.cu = !{!0, !3}
!llvm.ident = !{!5, !5}
!llvm.module.flags = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 5.0.0 (trunk 301051) (llvm/trunk 301062)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "a.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "clang version 5.0.0 (trunk 301051) (llvm/trunk 301062)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!4 = !DIFile(filename: "b.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!5 = !{!"clang version 5.0.0 (trunk 301051) (llvm/trunk 301062)"}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocation(line: 1, scope: !7)
!11 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!12 = !DILocation(line: 1, scope: !11)
!13 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !4, file: !4, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !3, variables: !2)
!14 = !DILocation(line: 1, scope: !11, inlinedAt: !15)
!15 = distinct !DILocation(line: 1, scope: !13)
!16 = !DILocation(line: 1, scope: !13)
