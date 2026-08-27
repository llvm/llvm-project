; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+amx-int8 -mattr=+avx512f -verify-machineinstrs -o /dev/null

define void @test_debug_instr(ptr %arg) #0 !dbg !4 {
  #dbg_declare(ptr %arg, !12, !DIExpression(), !15)
  %1 = call x86_amx @llvm.x86.cast.vector.to.tile.v1024i8(<1024 x i8> zeroinitializer)
  %2 = call x86_amx @llvm.x86.tdpbssd.internal(i16 0, i16 0, i16 0, x86_amx %1, x86_amx %1, x86_amx %1)
  ret void
}

declare x86_amx @llvm.x86.cast.vector.to.tile.v1024i8(<1024 x i8>) #1
declare x86_amx @llvm.x86.tdpbssd.internal(i16, i16, i16, x86_amx, x86_amx, x86_amx) #2

attributes #0 = { "target-features"="+amx-int8" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C11, file: !2, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!2 = !DIFile(filename: "test.c", directory: "/tmp")
!3 = !{}
!4 = distinct !DISubprogram(name: "test_debug_instr", linkageName: "test_debug_instr", scope: !2, file: !2, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1, retainedNodes: !3)
!9 = distinct !DISubroutineType(types: !3)
!12 = distinct !DILocalVariable(name: "a", arg: 1, scope: !4, file: !2, line: 1, type: !13)
!13 = !DICompositeType(tag: DW_TAG_structure_type, name: "__tile1024i", scope: !4, file: !2, size: 16384, align: 8192, flags: DIFlagPublic, elements: !3, identifier: "__tile1024i")
!15 = !DILocation(line: 1, column: 1, scope: !4)
