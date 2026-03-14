%struct.Class = type { i8 }

define weak_odr i32 @_ZN5ClassIiE3fooEv(ptr %this) align 2 !dbg !4 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr
  ret i32 0, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.6.0 (trunk 224193) (llvm/trunk 224197)", isOptimized: false, emissionKind: LineTablesOnly, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "t2.cpp", directory: "/Users/dexonsmith/data/llvm/staging/test/Linker/repro/d2")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 2, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 2, file: !5, scope: !6, type: !7, retainedNodes: !2)
!5 = !DIFile(filename: "../t.h", directory: "/Users/dexonsmith/data/llvm/staging/test/Linker/repro/d2")
!6 = !DIFile(filename: "../t.h", directory: "/Users/dexonsmith/data/llvm/staging/test/Linker/repro/d2")
!7 = !DISubroutineType(types: !2)
!8 = !{i32 2, !"Dwarf Version", i32 2}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"PIC Level", i32 2}
!11 = !{!"clang version 3.6.0 (trunk 224193) (llvm/trunk 224197)"}
!12 = !DILocation(line: 2, column: 15, scope: !4)
