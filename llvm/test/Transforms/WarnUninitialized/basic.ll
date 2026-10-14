; RUN: opt -passes=warn-uninitialized-early -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -passes=strip -S %s | opt -passes=warn-uninitialized-early \
; RUN:   -disable-output - 2>&1 | FileCheck %s --check-prefix=NO-DEBUG

; NO-DEBUG: warning: <unknown>:0:0: field is uninitialized when used here

%pair = type { i32, i32 }

@escaped = global ptr null

declare void @opaque(ptr)
declare void @readonly(ptr) memory(read)
declare void @consume(i32 noundef)
declare void @forward(i32)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)

define void @no_write() !dbg !5 {
entry:
  %p = alloca %pair, align 4
  %field = getelementptr inbounds %pair, ptr %p, i64 0, i32 0
  %value = load i32, ptr %field, align 4, !dbg !20
  call void @consume(i32 %value)
  ret void
}

; CHECK: warning: warn-uninitialized.cpp:10:7: field is uninitialized when used here

define void @same_field_write() !dbg !6 {
entry:
  %p = alloca %pair, align 4
  %field = getelementptr inbounds %pair, ptr %p, i64 0, i32 0
  store i32 1, ptr %field, align 4
  %value = load i32, ptr %field, align 4, !dbg !21
  ret void
}

; CHECK-NOT: warn-uninitialized.cpp:20:7

define void @sibling_field_write() !dbg !7 {
entry:
  %p = alloca %pair, align 4
  %sibling = getelementptr inbounds %pair, ptr %p, i64 0, i32 1
  store i32 1, ptr %sibling, align 4
  %field = getelementptr inbounds %pair, ptr %p, i64 0, i32 0
  %value = load i32, ptr %field, align 4, !dbg !22
  call void @consume(i32 %value)
  ret void
}

; CHECK: warning: warn-uninitialized.cpp:30:7: field is uninitialized when used here

define void @unknown_call() !dbg !8 {
entry:
  %p = alloca %pair, align 4
  call void @opaque(ptr %p)
  %field = getelementptr inbounds %pair, ptr %p, i64 0, i32 0
  %value = load i32, ptr %field, align 4, !dbg !23
  ret void
}

; CHECK-NOT: warn-uninitialized.cpp:40:7

define void @readonly_call() !dbg !9 {
entry:
  %p = alloca %pair, align 4
  call void @readonly(ptr %p)
  %field = getelementptr inbounds %pair, ptr %p, i64 0, i32 0
  %value = load i32, ptr %field, align 4, !dbg !24
  call void @consume(i32 %value)
  ret void
}

; CHECK: warning: warn-uninitialized.cpp:50:7: field is uninitialized when used here

define void @conditional_write(i1 %condition) !dbg !10 {
entry:
  %p = alloca %pair, align 4
  br i1 %condition, label %init, label %merge

init:
  %init.field = getelementptr inbounds %pair, ptr %p, i64 0, i32 0
  store i32 1, ptr %init.field, align 4
  br label %merge

merge:
  %field = getelementptr inbounds %pair, ptr %p, i64 0, i32 0
  %value = load i32, ptr %field, align 4, !dbg !25
  call void @consume(i32 %value)
  ret void
}

; CHECK: warning: warn-uninitialized.cpp:60:7: field may be uninitialized when used here

define void @conditional_sibling_write(i1 %condition) !dbg !11 {
entry:
  %p = alloca %pair, align 4
  br i1 %condition, label %init, label %merge

init:
  %sibling = getelementptr inbounds %pair, ptr %p, i64 0, i32 1
  store i32 1, ptr %sibling, align 4
  br label %merge

merge:
  %field = getelementptr inbounds %pair, ptr %p, i64 0, i32 0
  %value = load i32, ptr %field, align 4, !dbg !26
  call void @consume(i32 %value)
  ret void
}

; CHECK: warning: warn-uninitialized.cpp:70:7: field is uninitialized when used here

define internal void @empty(ptr %this) {
entry:
  %slot = alloca ptr, align 8
  store ptr %this, ptr %slot, align 8
  %reload = load ptr, ptr %slot, align 8
  ret void
}

define internal void @write_sibling(ptr %this) {
entry:
  %slot = alloca ptr, align 8
  store ptr %this, ptr %slot, align 8
  %reload = load ptr, ptr %slot, align 8
  %sibling = getelementptr inbounds %pair, ptr %reload, i64 0, i32 1
  store i32 1, ptr %sibling, align 4
  ret void
}

define internal void @write_field(ptr %this) {
entry:
  %slot = alloca ptr, align 8
  store ptr %this, ptr %slot, align 8
  %reload = load ptr, ptr %slot, align 8
  %field = getelementptr inbounds %pair, ptr %reload, i64 0, i32 0
  store i32 1, ptr %field, align 4
  ret void
}

define void @empty_callee() !dbg !12 {
entry:
  %p = alloca %pair, align 4
  call void @empty(ptr %p)
  %field = getelementptr inbounds %pair, ptr %p, i64 0, i32 0
  %value = load i32, ptr %field, align 4, !dbg !27
  call void @consume(i32 %value)
  ret void
}

; CHECK: warning: warn-uninitialized.cpp:80:7: field is uninitialized when used here

define void @sibling_callee() !dbg !13 {
entry:
  %p = alloca %pair, align 4
  call void @write_sibling(ptr %p)
  %field = getelementptr inbounds %pair, ptr %p, i64 0, i32 0
  %value = load i32, ptr %field, align 4, !dbg !28
  call void @consume(i32 %value)
  ret void
}

; CHECK: warning: warn-uninitialized.cpp:90:7: field is uninitialized when used here

define void @field_callee() !dbg !14 {
entry:
  %p = alloca %pair, align 4
  call void @write_field(ptr %p)
  %field = getelementptr inbounds %pair, ptr %p, i64 0, i32 0
  %value = load i32, ptr %field, align 4, !dbg !29
  ret void
}

; CHECK-NOT: warn-uninitialized.cpp:100:7


define void @copy_uninitialized() !dbg !15 {
entry:
  %source = alloca %pair, align 4
  %dest = alloca %pair, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr %dest, ptr %source, i64 8, i1 false)
  %field = getelementptr inbounds %pair, ptr %dest, i64 0, i32 0
  %value = load i32, ptr %field, align 4, !dbg !30
  call void @consume(i32 %value)
  ret void
}

; CHECK: warning: warn-uninitialized.cpp:110:7: field is uninitialized when used here

define void @copy_initialized() !dbg !16 {
entry:
  %source = alloca %pair, align 4
  %dest = alloca %pair, align 4
  %source.field = getelementptr inbounds %pair, ptr %source, i64 0, i32 0
  store i32 1, ptr %source.field, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr %dest, ptr %source, i64 8, i1 false)
  %field = getelementptr inbounds %pair, ptr %dest, i64 0, i32 0
  %value = load i32, ptr %field, align 4, !dbg !31
  ret void
}

; CHECK-NOT: warn-uninitialized.cpp:120:7

define void @copy_sibling_initialized() !dbg !17 {
entry:
  %source = alloca %pair, align 4
  %dest = alloca %pair, align 4
  %source.sibling = getelementptr inbounds %pair, ptr %source, i64 0, i32 1
  store i32 1, ptr %source.sibling, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr %dest, ptr %source, i64 8, i1 false)
  %field = getelementptr inbounds %pair, ptr %dest, i64 0, i32 0
  %value = load i32, ptr %field, align 4, !dbg !32
  call void @consume(i32 %value)
  ret void
}

; CHECK: warning: warn-uninitialized.cpp:130:7: field is uninitialized when used here

define void @escaped_address() !dbg !18 {
entry:
  %p = alloca %pair, align 4
  store ptr %p, ptr @escaped, align 8
  %field = getelementptr inbounds %pair, ptr %p, i64 0, i32 0
  %value = load i32, ptr %field, align 4, !dbg !33
  ret void
}

; CHECK-NOT: warn-uninitialized.cpp:140:7

define internal ptr @return_pointer(ptr %pointer) {
entry:
  ret ptr %pointer
}

define void @returned_address() !dbg !19 {
entry:
  %p = alloca %pair, align 4
  %escaped = call ptr @return_pointer(ptr %p)
  %field = getelementptr inbounds %pair, ptr %p, i64 0, i32 0
  %value = load i32, ptr %field, align 4, !dbg !34
  ret void
}

; CHECK-NOT: warn-uninitialized.cpp:150:7

define void @unreachable_block() !dbg !35 {
entry:
  %p = alloca %pair, align 4
  ret void

dead:
  %field = getelementptr inbounds %pair, ptr %p, i64 0, i32 0
  %value = load i32, ptr %field, align 4, !dbg !36
  ret void
}

; CHECK-NOT: warn-uninitialized.cpp:160:7

define void @masked_read_modify_write() !dbg !37 {
entry:
  %p = alloca %pair, align 4
  %field = getelementptr inbounds %pair, ptr %p, i64 0, i32 0
  %old = load i32, ptr %field, align 4, !dbg !38
  %preserved = and i32 %old, -8
  %new = or i32 %preserved, 3
  store i32 %new, ptr %field, align 4
  ret void
}

; CHECK-NOT: warn-uninitialized.cpp:170:7

define void @masked_read_modify_write_with_use() !dbg !39 {
entry:
  %p = alloca %pair, align 4
  %field = getelementptr inbounds %pair, ptr %p, i64 0, i32 0
  %old = load i32, ptr %field, align 4, !dbg !40
  %preserved = and i32 %old, -8
  %new = or i32 %preserved, 3
  store i32 %new, ptr %field, align 4
  call void @consume(i32 %old)
  ret void
}

; CHECK: warning: warn-uninitialized.cpp:180:7: field is uninitialized when used here

define void @forwarded_uninitialized() !dbg !41 {
entry:
  %p = alloca %pair, align 4
  %field = getelementptr inbounds %pair, ptr %p, i64 0, i32 0
  %value = load i32, ptr %field, align 4, !dbg !42
  call void @forward(i32 %value)
  ret void
}

; CHECK-NOT: warn-uninitialized.cpp:190:7

define void @stored_uninitialized() !dbg !43 {
entry:
  %p = alloca %pair, align 4
  %dest = alloca i32, align 4
  %field = getelementptr inbounds %pair, ptr %p, i64 0, i32 0
  %value = load i32, ptr %field, align 4, !dbg !44
  store i32 %value, ptr %dest, align 4
  ret void
}

; CHECK-NOT: warn-uninitialized.cpp:200:7

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "warn-uninitialized.cpp", directory: "")
!2 = !DISubroutineType(types: !4)
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{}
!5 = distinct !DISubprogram(name: "no_write", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!6 = distinct !DISubprogram(name: "same_field_write", scope: !1, file: !1, line: 2, type: !2, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0)
!7 = distinct !DISubprogram(name: "sibling_field_write", scope: !1, file: !1, line: 3, type: !2, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !0)
!8 = distinct !DISubprogram(name: "unknown_call", scope: !1, file: !1, line: 4, type: !2, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !0)
!9 = distinct !DISubprogram(name: "readonly_call", scope: !1, file: !1, line: 5, type: !2, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !0)
!10 = distinct !DISubprogram(name: "conditional_write", scope: !1, file: !1, line: 6, type: !2, scopeLine: 6, spFlags: DISPFlagDefinition, unit: !0)
!11 = distinct !DISubprogram(name: "conditional_sibling_write", scope: !1, file: !1, line: 7, type: !2, scopeLine: 7, spFlags: DISPFlagDefinition, unit: !0)
!12 = distinct !DISubprogram(name: "empty_callee", scope: !1, file: !1, line: 8, type: !2, scopeLine: 8, spFlags: DISPFlagDefinition, unit: !0)
!13 = distinct !DISubprogram(name: "sibling_callee", scope: !1, file: !1, line: 9, type: !2, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !0)
!14 = distinct !DISubprogram(name: "field_callee", scope: !1, file: !1, line: 10, type: !2, scopeLine: 10, spFlags: DISPFlagDefinition, unit: !0)
!15 = distinct !DISubprogram(name: "copy_uninitialized", scope: !1, file: !1, line: 11, type: !2, scopeLine: 11, spFlags: DISPFlagDefinition, unit: !0)
!16 = distinct !DISubprogram(name: "copy_initialized", scope: !1, file: !1, line: 12, type: !2, scopeLine: 12, spFlags: DISPFlagDefinition, unit: !0)
!17 = distinct !DISubprogram(name: "copy_sibling_initialized", scope: !1, file: !1, line: 13, type: !2, scopeLine: 13, spFlags: DISPFlagDefinition, unit: !0)
!18 = distinct !DISubprogram(name: "escaped_address", scope: !1, file: !1, line: 14, type: !2, scopeLine: 14, spFlags: DISPFlagDefinition, unit: !0)
!19 = distinct !DISubprogram(name: "returned_address", scope: !1, file: !1, line: 15, type: !2, scopeLine: 15, spFlags: DISPFlagDefinition, unit: !0)
!20 = !DILocation(line: 10, column: 7, scope: !5)
!21 = !DILocation(line: 20, column: 7, scope: !6)
!22 = !DILocation(line: 30, column: 7, scope: !7)
!23 = !DILocation(line: 40, column: 7, scope: !8)
!24 = !DILocation(line: 50, column: 7, scope: !9)
!25 = !DILocation(line: 60, column: 7, scope: !10)
!26 = !DILocation(line: 70, column: 7, scope: !11)
!27 = !DILocation(line: 80, column: 7, scope: !12)
!28 = !DILocation(line: 90, column: 7, scope: !13)
!29 = !DILocation(line: 100, column: 7, scope: !14)
!30 = !DILocation(line: 110, column: 7, scope: !15)
!31 = !DILocation(line: 120, column: 7, scope: !16)
!32 = !DILocation(line: 130, column: 7, scope: !17)
!33 = !DILocation(line: 140, column: 7, scope: !18)
!34 = !DILocation(line: 150, column: 7, scope: !19)
!35 = distinct !DISubprogram(name: "unreachable_block", scope: !1, file: !1, line: 16, type: !2, scopeLine: 16, spFlags: DISPFlagDefinition, unit: !0)
!36 = !DILocation(line: 160, column: 7, scope: !35)
!37 = distinct !DISubprogram(name: "masked_read_modify_write", scope: !1, file: !1, line: 17, type: !2, scopeLine: 17, spFlags: DISPFlagDefinition, unit: !0)
!38 = !DILocation(line: 170, column: 7, scope: !37)
!39 = distinct !DISubprogram(name: "masked_read_modify_write_with_use", scope: !1, file: !1, line: 18, type: !2, scopeLine: 18, spFlags: DISPFlagDefinition, unit: !0)
!40 = !DILocation(line: 180, column: 7, scope: !39)
!41 = distinct !DISubprogram(name: "forwarded_uninitialized", scope: !1, file: !1, line: 19, type: !2, scopeLine: 19, spFlags: DISPFlagDefinition, unit: !0)
!42 = !DILocation(line: 190, column: 7, scope: !41)
!43 = distinct !DISubprogram(name: "stored_uninitialized", scope: !1, file: !1, line: 20, type: !2, scopeLine: 20, spFlags: DISPFlagDefinition, unit: !0)
!44 = !DILocation(line: 200, column: 7, scope: !43)
