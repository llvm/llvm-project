; RUN: llc < %s | FileCheck %s
; CHECK:             .section .debug$S,"dr"{{$}}
; CHECK-NEXT:        .p2align 2
; CHECK-NEXT:        .long 4
; CHECK-NEXT:        .long	241
; CHECK-NEXT:        .long	[[SUBSEC_END:.*]]-[[SUBSEC_START:.*]] # Subsection size
; CHECK-NEXT:        [[SUBSEC_START]]:
; CHECK-NEXT:        .short	[[OBJNAME_END:.*]]-[[OBJNAME_START:.*]] # Record length
; CHECK:             [[OBJNAME_END]]:
; CHECK-NEXT:        .short	[[COMPILE3_END:.*]]-[[COMPILE3_START:.*]] # Record length
; CHECK:             [[COMPILE3_END]]:
; CHECK-NEXT:        [[SUBSEC_END]]:
; CHECK-NEXT:        .p2align 2
; CHECK-NEXT:        .cv_filechecksums
; CHECK-NEXT:        .cv_stringtable

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

define void @baz() {
entry:
  %x.i.i = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %x.i.i, metadata !6, metadata !12), !dbg !13
  store i32 5, i32* %x.i.i, align 4, !dbg !13
  ret void
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 (trunk 276756) (llvm/trunk 276952)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "-", directory: "/")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 4.0.0 (trunk 276756) (llvm/trunk 276952)"}
!6 = !DILocalVariable(name: "x", scope: !7, file: !8, line: 1, type: !11)
!7 = distinct !DISubprogram(name: "foo", scope: !8, file: !8, line: 1, type: !9, isLocal: true, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DIFile(filename: "<stdin>", directory: "/")
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !DIExpression()
!13 = !DILocation(line: 1, column: 56, scope: !7, inlinedAt: !14)
!14 = distinct !DILocation(line: 2, column: 52, scope: !15)
!15 = distinct !DISubprogram(name: "bar", scope: !8, file: !8, line: 2, type: !9, isLocal: true, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, retainedNodes: !2)
