target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @_Z8thinlto1v() unnamed_addr {
  %3 = alloca i64, align 4
    #dbg_declare(ptr %3, !14, !DIExpression(), !15)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "B.cpp", directory: ".")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!10 = distinct !DISubprogram(name: "thinlto1", linkageName: "_Z8thinlto1v", scope: !11, file: !11, line: 8, type: !12, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DIFile(filename: "b.cpp", directory: ".")
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !DILocalVariable(name: "a", arg: 1, scope: !10, file: !11, line: 18, type: !16)
!15 = !DILocation(line: 18, column: 19, scope: !10)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S<&func1>", file: !11, line: 2, size: 8, flags: DIFlagTypePassByValue, elements: !17, templateParams: !18, identifier: "_ZTS1SIXadL_Z5func1vEEE")
!17 = !{}
!18 = !{!19}
!19 = !DITemplateValueParameter(name: "Func", type: !20, value: ptr undef)
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
