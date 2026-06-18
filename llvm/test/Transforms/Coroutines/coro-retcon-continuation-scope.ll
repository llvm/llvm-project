; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

@"coroutineTwc" = global <{ i32, i32, i64 }> <{ i32 trunc (i64 sub (i64 ptrtoint (ptr @"coroutine" to i64), i64 ptrtoint (ptr @"coroutineTwc" to i64)) to i32), i32 0, i64 51979 }>, align 8

define swiftcc { ptr, ptr } @"coroutine"(ptr noalias %0, ptr %1, ptr swiftself %2) #0 !dbg !39 {
entry:
  %3 = call token @llvm.coro.id.retcon.once(i32 -1, i32 16, ptr %0, ptr @"$s4test1SVSiIetMIlYl_TC", ptr @_swift_coro_alloc, ptr @_swift_coro_dealloc), !dbg !48
  %4 = call ptr @llvm.coro.begin(token %3, ptr null), !dbg !48
  call swiftcc void @"$s4test6FINISHyyF"(), !dbg !50
  %.member_int = getelementptr inbounds nuw <{ <{ i64 }> }>, ptr %2, i32 0, i32 0, !dbg !51
; Test that the scope of the continuation is NOT on the same line as the split
; point.
; In this case the split point is on line 11, and the scope of the continuation
; should be line 12.
  %5 = call ptr (...) @llvm.coro.suspend.retcon.p0(ptr %.member_int), !dbg !52 ; [debug line = 11:7]
; CHECK-LABEL: define {{.*}} @coroutine.resume
; CHECK-SAME: !dbg ![[CONTINUATION_SP:[0-9]+]]
; CHECK: ![[CONTINUATION_SP]] = {{.*}} scopeLine: 12
  br i1 false, label %7, label %6, !dbg !52       ; [debug line = 11:7]

6:
  call swiftcc void @"$s4test6FINISHyyF"(), !dbg !53 ; [debug line = 12:7]
  br label %coro.end, !dbg !54

7:
  br label %coro.end, !dbg !54

coro.end:
  call void @llvm.coro.end(ptr %4, i1 false, token none), !dbg !54
  unreachable, !dbg !54
}

declare swiftcc void @"$s4test6FINISHyyF"()
declare swiftcc ptr @_swift_coro_alloc(i64) #3
declare swiftcc void @_swift_coro_dealloc(ptr) #3
declare swiftcc ptr @_swift_coro_alloc_frame(ptr, ptr, i64, i64) #3
declare swiftcc void @_swift_coro_dealloc_frame(ptr, ptr, ptr) #3
declare swiftcc void @"$s4test1SVSiIetMIlYl_TC"(ptr noalias, ptr)

attributes #0 = { noinline presplitcoroutine }
attributes #3 = { noinline nounwind }

!llvm.module.flags = !{!7, !8, !9, !10, !11}
!llvm.dbg.cu = !{!17}

!7 = !{i32 7, !"Dwarf Version", i32 5}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 8, !"PIC Level", i32 2}
!11 = !{i32 7, !"uwtable", i32 1}
!17 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !18, producer: "Swift", isOptimized: false, emissionKind: FullDebug)
!18 = !DIFile(filename: "test.ll", directory: "/")
!39 = distinct !DISubprogram(name: "COMPUTED_PROPERTY.yielding_mutate", linkageName: "coroutine", scope: !40, file: !18, line: 9, type: !41, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !17, declaration: !44, retainedNodes: !45)
!40 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !18, size: 64, runtimeLang: DW_LANG_Swift, identifier: "$s4test1SVD")
!41 = !DISubroutineType(types: !42)
!42 = !{!43, !40}
!43 = !DICompositeType(tag: DW_TAG_structure_type, name: "$sytD", flags: DIFlagFwdDecl, runtimeLang: DW_LANG_Swift, identifier: "$sytD")
!44 = !DISubprogram(name: "COMPUTED_PROPERTY.yielding_mutate", linkageName: "coroutine", scope: !40, file: !18, line: 9, type: !41, scopeLine: 9, spFlags: 0)
!45 = !{!46}
!46 = !DILocalVariable(name: "self", arg: 1, scope: !39, file: !18, line: 9, type: !40, flags: DIFlagArtificial)
!47 = !DILocation(line: 9, column: 14, scope: !39)
!48 = !DILocation(line: 0, scope: !39)
!50 = !DILocation(line: 10, column: 7, scope: !39)
!51 = !DILocation(line: 11, column: 13, scope: !39)
!52 = !DILocation(line: 11, column: 7, scope: !39)
!53 = !DILocation(line: 12, column: 7, scope: !39)
!54 = !DILocation(line: 0, scope: !55)
!55 = !DILexicalBlockFile(scope: !39, file: !56, discriminator: 0)
!56 = !DIFile(filename: "<compiler-generated>", directory: "/")
