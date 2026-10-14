; RUN: opt -passes=warn-uninitialized-late -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -passes=strip -S %s | opt -passes=warn-uninitialized-late \
; RUN:   -disable-output - 2>&1 | FileCheck %s --check-prefix=NO-DEBUG

; NO-DEBUG: warning: <unknown>:0:0: field is uninitialized when used here

target triple = "x86_64-unknown-linux-gnu"

%pair = type { i32, i32 }

declare noalias ptr @malloc(i64) nounwind allockind("alloc,uninitialized") allocsize(0)
declare noalias ptr @calloc(i64, i64) nounwind allockind("alloc,zeroed") allocsize(0,1)
declare void @opaque(ptr)
declare void @consume(i32 noundef)

define void @stack_uninitialized() !dbg !5 {
entry:
  %object = alloca %pair, align 4
  %value = load i32, ptr %object, align 4, !dbg !20
  call void @consume(i32 %value)
  ret void
}
; CHECK: warning: late.cpp:10:7: field is uninitialized when used here

define void @heap_uninitialized() !dbg !6 {
entry:
  %object = call ptr @malloc(i64 8)
  %value = load i32, ptr %object, align 4, !dbg !21
  call void @consume(i32 %value)
  ret void
}
; CHECK: warning: late.cpp:20:7: field is uninitialized when used here

define void @heap_sibling_initialized() !dbg !7 {
entry:
  %object = call ptr @malloc(i64 8)
  %sibling = getelementptr i8, ptr %object, i64 4
  store i32 1, ptr %sibling, align 4
  %value = load i32, ptr %object, align 4, !dbg !22
  call void @consume(i32 %value)
  ret void
}
; CHECK: warning: late.cpp:30:7: field is uninitialized when used here

define void @heap_pointer_spill() !dbg !8 {
entry:
  %slot = alloca ptr, align 8
  %allocation = call ptr @malloc(i64 24)
  %object = getelementptr i8, ptr %allocation, i64 16
  store ptr %object, ptr %slot, align 8
  %restored = load ptr, ptr %slot, align 8
  %value = load i32, ptr %restored, align 4, !dbg !23
  call void @consume(i32 %value)
  ret void
}
; CHECK: warning: late.cpp:40:7: field is uninitialized when used here

define void @heap_initialized() !dbg !9 {
entry:
  %object = call ptr @malloc(i64 8)
  store i32 1, ptr %object, align 4
  %value = load i32, ptr %object, align 4, !dbg !24
  ret void
}
; CHECK-NOT: late.cpp:50:7

define void @heap_conditionally_initialized(i1 %condition) !dbg !10 {
entry:
  %object = call ptr @malloc(i64 8)
  br i1 %condition, label %initialize, label %merge
initialize:
  store i32 1, ptr %object, align 4
  br label %merge
merge:
  %value = load i32, ptr %object, align 4, !dbg !25
  call void @consume(i32 %value)
  ret void
}
; CHECK: warning: late.cpp:60:7: field may be uninitialized when used here

define void @heap_unknown_call() !dbg !11 {
entry:
  %object = call ptr @malloc(i64 8)
  call void @opaque(ptr %object)
  %value = load i32, ptr %object, align 4, !dbg !26
  ret void
}
; CHECK-NOT: late.cpp:70:7

define void @zeroed_allocation() !dbg !12 {
entry:
  %object = call ptr @calloc(i64 1, i64 8)
  %value = load i32, ptr %object, align 4, !dbg !27
  ret void
}
; CHECK-NOT: late.cpp:80:7

define void @not_inlined() !dbg !13 {
entry:
  %object = call ptr @malloc(i64 8)
  %value = load i32, ptr %object, align 4, !dbg !28
  ret void
}
; CHECK-NOT: late.cpp:90:7

define void @unreachable_block() !dbg !15 {
entry:
  %object = call ptr @malloc(i64 8)
  ret void

dead:
  %value = load i32, ptr %object, align 4, !dbg !40
  call void @consume(i32 %value)
  ret void
}
; CHECK-NOT: late.cpp:100:7

define void @bitfield_read_modify_write() !dbg !16 {
entry:
  %object = call ptr @malloc(i64 8)
  %old = load i32, ptr %object, align 4, !dbg !41
  %preserved = and i32 %old, -16
  %new = or i32 %preserved, 7
  store i32 %new, ptr %object, align 4
  ret void
}
; CHECK-NOT: late.cpp:110:7

define void @masked_load() !dbg !17 {
entry:
  %object = call ptr @malloc(i64 8)
  %value = load i32, ptr %object, align 4, !dbg !42
  %selected = select i1 false, i32 %value, i32 0
  call void @consume(i32 %selected)
  ret void
}
; CHECK-NOT: late.cpp:120:7

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "late.cpp", directory: "")
!2 = !DISubroutineType(types: !4)
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{}
!5 = distinct !DISubprogram(name: "stack_uninitialized", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!6 = distinct !DISubprogram(name: "heap_uninitialized", scope: !1, file: !1, line: 2, type: !2, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0)
!7 = distinct !DISubprogram(name: "heap_sibling_initialized", scope: !1, file: !1, line: 3, type: !2, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !0)
!8 = distinct !DISubprogram(name: "heap_pointer_spill", scope: !1, file: !1, line: 4, type: !2, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !0)
!9 = distinct !DISubprogram(name: "heap_initialized", scope: !1, file: !1, line: 5, type: !2, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !0)
!10 = distinct !DISubprogram(name: "heap_conditionally_initialized", scope: !1, file: !1, line: 6, type: !2, scopeLine: 6, spFlags: DISPFlagDefinition, unit: !0)
!11 = distinct !DISubprogram(name: "heap_unknown_call", scope: !1, file: !1, line: 7, type: !2, scopeLine: 7, spFlags: DISPFlagDefinition, unit: !0)
!12 = distinct !DISubprogram(name: "zeroed_allocation", scope: !1, file: !1, line: 8, type: !2, scopeLine: 8, spFlags: DISPFlagDefinition, unit: !0)
!13 = distinct !DISubprogram(name: "not_inlined", scope: !1, file: !1, line: 9, type: !2, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !0)
!14 = distinct !DISubprogram(name: "use", scope: !1, file: !1, line: 100, type: !2, scopeLine: 100, spFlags: DISPFlagDefinition, unit: !0)
!15 = distinct !DISubprogram(name: "unreachable_block", scope: !1, file: !1, line: 10, type: !2, scopeLine: 10, spFlags: DISPFlagDefinition, unit: !0)
!16 = distinct !DISubprogram(name: "bitfield_read_modify_write", scope: !1, file: !1, line: 11, type: !2, scopeLine: 11, spFlags: DISPFlagDefinition, unit: !0)
!17 = distinct !DISubprogram(name: "masked_load", scope: !1, file: !1, line: 12, type: !2, scopeLine: 12, spFlags: DISPFlagDefinition, unit: !0)
!20 = !DILocation(line: 10, column: 7, scope: !14, inlinedAt: !30)
!21 = !DILocation(line: 20, column: 7, scope: !14, inlinedAt: !31)
!22 = !DILocation(line: 30, column: 7, scope: !14, inlinedAt: !32)
!23 = !DILocation(line: 40, column: 7, scope: !14, inlinedAt: !33)
!24 = !DILocation(line: 50, column: 7, scope: !14, inlinedAt: !34)
!25 = !DILocation(line: 60, column: 7, scope: !14, inlinedAt: !35)
!26 = !DILocation(line: 70, column: 7, scope: !14, inlinedAt: !36)
!27 = !DILocation(line: 80, column: 7, scope: !14, inlinedAt: !37)
!28 = !DILocation(line: 90, column: 7, scope: !13)
!30 = !DILocation(line: 1, column: 1, scope: !5)
!31 = !DILocation(line: 2, column: 1, scope: !6)
!32 = !DILocation(line: 3, column: 1, scope: !7)
!33 = !DILocation(line: 4, column: 1, scope: !8)
!34 = !DILocation(line: 5, column: 1, scope: !9)
!35 = !DILocation(line: 6, column: 1, scope: !10)
!36 = !DILocation(line: 7, column: 1, scope: !11)
!37 = !DILocation(line: 8, column: 1, scope: !12)
!40 = !DILocation(line: 100, column: 7, scope: !14, inlinedAt: !43)
!41 = !DILocation(line: 110, column: 7, scope: !14, inlinedAt: !44)
!42 = !DILocation(line: 120, column: 7, scope: !14, inlinedAt: !45)
!43 = !DILocation(line: 10, column: 1, scope: !15)
!44 = !DILocation(line: 11, column: 1, scope: !16)
!45 = !DILocation(line: 12, column: 1, scope: !17)
