define void @f() !dbg !3 {
entry:
  ; load: check remarks for both unnamed and named values.
  ; CHECK: remark: test.c:3:11: in function 'f', 'load' instruction ('%0') accesses memory in flat address space
  %0 = load i32, ptr null, align 4, !dbg !6
  ; CHECK: remark: test.c:3:11: in function 'f', 'load' instruction ('%load') accesses memory in flat address space
  %load = load i32, ptr null, align 4, !dbg !6
  ; CHECK: remark: test.c:3:11: in function 'f', 'load' instruction ('%load0') accesses memory in flat address space
  %load0 = load i32, ptr addrspace(0) null, align 4, !dbg !6
  %load1 = load i32, ptr addrspace(1) null, align 4, !dbg !6
  %load2 = load i32, ptr addrspace(2) null, align 4, !dbg !6

  ; store
  ; CHECK: remark: test.c:4:6: in function 'f', 'store' instruction accesses memory in flat address space
  store i32 0, ptr null, align 4, !dbg !7
  ; CHECK: remark: test.c:4:6: in function 'f', 'store' instruction accesses memory in flat address space
  store i32 0, ptr addrspace(0) null, align 4, !dbg !7
  store i32 0, ptr addrspace(1) null, align 4, !dbg !7
  store i32 0, ptr addrspace(8) null, align 4, !dbg !7

  ; atomicrmw
  ; CHECK: remark: test.c:5:1: in function 'f', 'atomicrmw' instruction ('%[[#]]') accesses memory in flat address space
  atomicrmw xchg ptr null, i32 10 seq_cst, !dbg !8
  ; CHECK: remark: test.c:5:1: in function 'f', 'atomicrmw' instruction ('%[[#]]') accesses memory in flat address space
  atomicrmw add ptr addrspace(0) null, i32 10 seq_cst, !dbg !8
  atomicrmw xchg ptr addrspace(1) null, i32 10 seq_cst, !dbg !8
  atomicrmw add ptr addrspace(37) null, i32 10 seq_cst, !dbg !8

  ; cmpxchg
  ; CHECK: remark: test.c:6:2: in function 'f', 'cmpxchg' instruction ('%[[#]]') accesses memory in flat address space
  cmpxchg ptr null, i32 0, i32 1 acq_rel monotonic, !dbg !9
  ; CHECK: remark: test.c:6:2: in function 'f', 'cmpxchg' instruction ('%[[#]]') accesses memory in flat address space
  cmpxchg ptr addrspace(0) null, i32 0, i32 1 acq_rel monotonic, !dbg !9
  cmpxchg ptr addrspace(1) null, i32 0, i32 1 acq_rel monotonic, !dbg !9
  cmpxchg ptr addrspace(934) null, i32 0, i32 1 acq_rel monotonic, !dbg !9

  ; llvm.memcpy
  ; CHECK: remark: test.c:7:3: in function 'f', 'llvm.memcpy.p0.p1.i64' call accesses memory in flat address space
  call void @llvm.memcpy.p0.p1.i64(ptr align 4 null, ptr addrspace(1) align 4 null, i64 10, i1 false), !dbg !10
  ; CHECK: remark: test.c:7:3: in function 'f', 'llvm.memcpy.p0.p1.i64' call accesses memory in flat address space
  call void @llvm.memcpy.p0.p1.i64(ptr addrspace(0) align 4 null, ptr addrspace(1) align 4 null, i64 10, i1 false), !dbg !10
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 4 null, ptr addrspace(1) align 4 null, i64 10, i1 false), !dbg !10
  call void @llvm.memcpy.p3.p1.i64(ptr addrspace(3) align 4 null, ptr addrspace(1) align 4 null, i64 10, i1 false), !dbg !10
  ; CHECK: remark: test.c:7:3: in function 'f', 'llvm.memcpy.p1.p0.i64' call accesses memory in flat address space
  call void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) align 4 null, ptr align 4 null, i64 10, i1 false), !dbg !10
  ; CHECK: remark: test.c:7:3: in function 'f', 'llvm.memcpy.p1.p0.i64' call accesses memory in flat address space
  call void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) align 4 null, ptr addrspace(0) align 4 null, i64 10, i1 false), !dbg !10
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 4 null, ptr addrspace(1) align 4 null, i64 10, i1 false), !dbg !10
  call void @llvm.memcpy.p1.p4.i64(ptr addrspace(1) align 4 null, ptr addrspace(4) align 4 null, i64 10, i1 false), !dbg !10
  ; CHECK: remark: test.c:7:3: in function 'f', 'llvm.memcpy.p0.p0.i64' call accesses memory in flat address space
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 null, ptr align 4 null, i64 10, i1 false), !dbg !10
  ; CHECK: remark: test.c:7:3: in function 'f', 'llvm.memcpy.p0.p0.i64' call accesses memory in flat address space
  call void @llvm.memcpy.p0.p0.i64(ptr addrspace(0) align 4 null, ptr addrspace(0) align 4 null, i64 10, i1 false), !dbg !10

  ; llvm.memcpy.inline
  ; CHECK: remark: test.c:7:3: in function 'f', 'llvm.memcpy.inline.p0.p0.i64' call accesses memory in flat address space
  call void @llvm.memcpy.inline.p0.p0.i64(ptr addrspace(0) align 4 null, ptr addrspace(0) align 4 null, i64 10, i1 false), !dbg !10
  ; CHECK: remark: test.c:7:3: in function 'f', 'llvm.memcpy.inline.p0.p1.i64' call accesses memory in flat address space
  call void @llvm.memcpy.inline.p0.p1.i64(ptr addrspace(0) align 4 null, ptr addrspace(1) align 4 null, i64 10, i1 false), !dbg !10
  ; CHECK: remark: test.c:7:3: in function 'f', 'llvm.memcpy.inline.p1.p0.i64' call accesses memory in flat address space
  call void @llvm.memcpy.inline.p1.p0.i64(ptr addrspace(1) align 4 null, ptr addrspace(0) align 4 null, i64 10, i1 false), !dbg !10
  call void @llvm.memcpy.inline.p1.p1.i64(ptr addrspace(1) align 4 null, ptr addrspace(1) align 4 null, i64 10, i1 false), !dbg !10

  ; llvm.memcpy.element.unordered.atomic
  ; CHECK: remark: test.c:7:3: in function 'f', 'llvm.memcpy.element.unordered.atomic.p0.p0.i64' call accesses memory in flat address space
  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr addrspace(0) align 4 null, ptr addrspace(0) align 4 null, i64 10, i32 4), !dbg !10
  ; CHECK: remark: test.c:7:3: in function 'f', 'llvm.memcpy.element.unordered.atomic.p0.p1.i64' call accesses memory in flat address space
  call void @llvm.memcpy.element.unordered.atomic.p0.p1.i64(ptr addrspace(0) align 4 null, ptr addrspace(1) align 4 null, i64 10, i32 4), !dbg !10
  ; CHECK: remark: test.c:7:3: in function 'f', 'llvm.memcpy.element.unordered.atomic.p1.p0.i64' call accesses memory in flat address space
  call void @llvm.memcpy.element.unordered.atomic.p1.p0.i64(ptr addrspace(1) align 4 null, ptr addrspace(0) align 4 null, i64 10, i32 4), !dbg !10
  call void @llvm.memcpy.element.unordered.atomic.p1.p1.i64(ptr addrspace(1) align 4 null, ptr addrspace(1) align 4 null, i64 10, i32 4), !dbg !10

  ; llvm.memmove
  ; CHECK: remark: test.c:8:4: in function 'f', 'llvm.memmove.p0.p1.i64' call accesses memory in flat address space
  call void @llvm.memmove.p0.p1.i64(ptr align 4 null, ptr addrspace(1) align 4 null, i64 10, i1 false), !dbg !11
  ; CHECK: remark: test.c:8:4: in function 'f', 'llvm.memmove.p0.p1.i64' call accesses memory in flat address space
  call void @llvm.memmove.p0.p1.i64(ptr addrspace(0) align 4 null, ptr addrspace(1) align 4 null, i64 10, i1 false), !dbg !11
  call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) align 4 null, ptr addrspace(1) align 4 null, i64 10, i1 false), !dbg !11
  call void @llvm.memmove.p3.p1.i64(ptr addrspace(3) align 4 null, ptr addrspace(1) align 4 null, i64 10, i1 false), !dbg !11
  ; CHECK: remark: test.c:8:4: in function 'f', 'llvm.memmove.p1.p0.i64' call accesses memory in flat address space
  call void @llvm.memmove.p1.p0.i64(ptr addrspace(1) align 4 null, ptr align 4 null, i64 10, i1 false), !dbg !11
  ; CHECK: remark: test.c:8:4: in function 'f', 'llvm.memmove.p1.p0.i64' call accesses memory in flat address space
  call void @llvm.memmove.p1.p0.i64(ptr addrspace(1) align 4 null, ptr addrspace(0) align 4 null, i64 10, i1 false), !dbg !11
  call void @llvm.memmove.p1.p1.i64(ptr addrspace(1) align 4 null, ptr addrspace(1) align 4 null, i64 10, i1 false), !dbg !11
  call void @llvm.memmove.p1.p4.i64(ptr addrspace(1) align 4 null, ptr addrspace(4) align 4 null, i64 10, i1 false), !dbg !11
  ; CHECK: remark: test.c:8:4: in function 'f', 'llvm.memmove.p0.p0.i64' call accesses memory in flat address space
  call void @llvm.memmove.p0.p0.i64(ptr align 4 null, ptr align 4 null, i64 10, i1 false), !dbg !11
  ; CHECK: remark: test.c:8:4: in function 'f', 'llvm.memmove.p0.p0.i64' call accesses memory in flat address space
  call void @llvm.memmove.p0.p0.i64(ptr addrspace(0) align 4 null, ptr addrspace(0) align 4 null, i64 10, i1 false), !dbg !11

  ; llvm.memmove.element.unordered.atomic
  ; CHECK: remark: test.c:8:4: in function 'f', 'llvm.memmove.element.unordered.atomic.p0.p0.i64' call accesses memory in flat address space
  call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr addrspace(0) align 4 null, ptr addrspace(0) align 4 null, i64 10, i32 4), !dbg !11
  ; CHECK: remark: test.c:8:4: in function 'f', 'llvm.memmove.element.unordered.atomic.p0.p1.i64' call accesses memory in flat address space
  call void @llvm.memmove.element.unordered.atomic.p0.p1.i64(ptr addrspace(0) align 4 null, ptr addrspace(1) align 4 null, i64 10, i32 4), !dbg !11
  ; CHECK: remark: test.c:8:4: in function 'f', 'llvm.memmove.element.unordered.atomic.p1.p0.i64' call accesses memory in flat address space
  call void @llvm.memmove.element.unordered.atomic.p1.p0.i64(ptr addrspace(1) align 4 null, ptr addrspace(0) align 4 null, i64 10, i32 4), !dbg !11
  call void @llvm.memmove.element.unordered.atomic.p1.p1.i64(ptr addrspace(1) align 4 null, ptr addrspace(1) align 4 null, i64 10, i32 4), !dbg !11

  ; llvm.memset
  ; CHECK: remark: test.c:9:5: in function 'f', 'llvm.memset.p0.i64' call accesses memory in flat address space
  call void @llvm.memset.p0.i64(ptr align 4 null, i8 0, i64 10, i1 false), !dbg !12
  ; CHECK: remark: test.c:9:5: in function 'f', 'llvm.memset.p0.i64' call accesses memory in flat address space
  call void @llvm.memset.p0.i64(ptr addrspace(0) align 4 null, i8 0, i64 10, i1 false), !dbg !12
  call void @llvm.memset.p1.i64(ptr addrspace(1) align 4 null, i8 0, i64 10, i1 false), !dbg !12
  call void @llvm.memset.p3.i64(ptr addrspace(3) align 4 null, i8 0, i64 10, i1 false), !dbg !12

  ; llvm.memset.inline
  ; CHECK: remark: test.c:9:5: in function 'f', 'llvm.memset.inline.p0.i64' call accesses memory in flat address space
  call void @llvm.memset.inline.p0.i64(ptr align 4 null, i8 0, i64 10, i1 false), !dbg !12
  ; CHECK: remark: test.c:9:5: in function 'f', 'llvm.memset.inline.p0.i64' call accesses memory in flat address space
  call void @llvm.memset.inline.p0.i64(ptr addrspace(0) align 4 null, i8 0, i64 10, i1 false), !dbg !12
  call void @llvm.memset.inline.p1.i64(ptr addrspace(1) align 4 null, i8 0, i64 10, i1 false), !dbg !12
  call void @llvm.memset.inline.p3.i64(ptr addrspace(3) align 4 null, i8 0, i64 10, i1 false), !dbg !12

  ; llvm.memset.element.unordered.atomic
  ; CHECK: remark: test.c:9:5: in function 'f', 'llvm.memset.element.unordered.atomic.p0.i64' call accesses memory in flat address space
  call void @llvm.memset.element.unordered.atomic.p0.i64(ptr align 4 null, i8 0, i64 10, i32 4), !dbg !12
  ; CHECK: remark: test.c:9:5: in function 'f', 'llvm.memset.element.unordered.atomic.p0.i64' call accesses memory in flat address space
  call void @llvm.memset.element.unordered.atomic.p0.i64(ptr addrspace(0) align 4 null, i8 0, i64 10, i32 4), !dbg !12
  call void @llvm.memset.element.unordered.atomic.p1.i64(ptr addrspace(1) align 4 null, i8 0, i64 10, i32 4), !dbg !12
  call void @llvm.memset.element.unordered.atomic.p3.i64(ptr addrspace(3) align 4 null, i8 0, i64 10, i32 4), !dbg !12

  ret void
}
; CHECK: remark: test.c:2:0: in function 'f', FlatAddrspaceAccesses = 36

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 19.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !4, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !5)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !DILocation(line: 3, column: 11, scope: !3)
!7 = !DILocation(line: 4, column: 6, scope: !3)
!8 = !DILocation(line: 5, column: 1, scope: !3)
!9 = !DILocation(line: 6, column: 2, scope: !3)
!10 = !DILocation(line: 7, column: 3, scope: !3)
!11 = !DILocation(line: 8, column: 4, scope: !3)
!12 = !DILocation(line: 9, column: 5, scope: !3)
