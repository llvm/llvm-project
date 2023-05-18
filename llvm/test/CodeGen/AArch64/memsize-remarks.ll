; RUN: llc %s -pass-remarks-analysis=gisel-irtranslator-memsize -pass-remarks-output=%t.opt.yaml -pass-remarks-filter=gisel-irtranslator-memsize -global-isel -o /dev/null 2>&1 | FileCheck %s --check-prefix=GISEL --implicit-check-not=GISEL
; RUN: cat %t.opt.yaml | FileCheck -check-prefix=YAML %s

source_filename = "memsize.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios7.0.0"

declare ptr @__memmove_chk(ptr, ptr, i64, i64) #1
declare ptr @__memcpy_chk(ptr, ptr, i64, i64) #1
declare ptr @__memset_chk(ptr, i32, i64, i64) #1
declare i64 @llvm.objectsize.i64.p0(ptr, i1 immarg, i1 immarg, i1 immarg) #2
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) argmemonly nounwind willreturn writeonly
declare void @llvm.memmove.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1 immarg) argmemonly nounwind willreturn
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) argmemonly nounwind willreturn
declare void @bzero(ptr nocapture, i64) nofree nounwind
declare void @bcopy(ptr nocapture, ptr nocapture, i64) nofree nounwind
declare ptr @memset(ptr, i32, i64)

define void @memcpy_dynamic(ptr %d, ptr %s, i64 %l) #0 !dbg !14 {
entry:
  %0 = call i64 @llvm.objectsize.i64.p0(ptr %d, i1 false, i1 true, i1 false), !dbg !16
; GISEL: remark: memsize.c:4:3: Call to memcpy.{{$}}
  %call = call ptr @__memcpy_chk(ptr %d, ptr %s, i64 %l, i64 %0) #4, !dbg !17
  ret void, !dbg !18
}

define void @memcpy_single(ptr %d, ptr %s, i64 %l) #0 !dbg !23 {
entry:
  %0 = call i64 @llvm.objectsize.i64.p0(ptr %d, i1 false, i1 true, i1 false), !dbg !24
; GISEL: remark: memsize.c:10:3: Call to memcpy. Memory operation size: 1 bytes.
  %call = call ptr @__memcpy_chk(ptr %d, ptr %s, i64 1, i64 %0) #4, !dbg !25
  ret void, !dbg !26
}

define void @memcpy_intrinsic(ptr %d, ptr %s, i64 %l) #0 {
entry:
  %0 = call i64 @llvm.objectsize.i64.p0(ptr %d, i1 false, i1 true, i1 false)
; GISEL: remark: <unknown>:0:0: Call to memcpy. Memory operation size: 1 bytes.
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 1, i1 false)
  ret void
}

define void @memcpy_static(ptr %d, ptr %s, i64 %l) #0 !dbg !27 {
entry:
  %0 = call i64 @llvm.objectsize.i64.p0(ptr %d, i1 false, i1 true, i1 false), !dbg !28
; GISEL: remark: memsize.c:13:3: Call to memcpy. Memory operation size: 100 bytes.
  %call = call ptr @__memcpy_chk(ptr %d, ptr %s, i64 100, i64 %0) #4, !dbg !29
  ret void, !dbg !30
}

define void @memcpy_huge(ptr %d, ptr %s, i64 %l) #0 !dbg !31 {
entry:
  %0 = call i64 @llvm.objectsize.i64.p0(ptr %d, i1 false, i1 true, i1 false), !dbg !32
; GISEL: remark: memsize.c:16:3: Call to memcpy. Memory operation size: 100000 bytes.
  %call = call ptr @__memcpy_chk(ptr %d, ptr %s, i64 100000, i64 %0) #4, !dbg !33
  ret void, !dbg !34
}

define void @memmove_dynamic(ptr %d, ptr %s, i64 %l) #0 {
entry:
  %0 = call i64 @llvm.objectsize.i64.p0(ptr %d, i1 false, i1 true, i1 false)
; GISEL: remark: <unknown>:0:0: Call to memmove.{{$}}
  %call = call ptr @__memmove_chk(ptr %d, ptr %s, i64 %l, i64 %0) #4
  ret void
}

define void @memmove_single(ptr %d, ptr %s, i64 %l) #0 {
entry:
  %0 = call i64 @llvm.objectsize.i64.p0(ptr %d, i1 false, i1 true, i1 false)
; GISEL: remark: <unknown>:0:0: Call to memmove. Memory operation size: 1 bytes.
  %call = call ptr @__memmove_chk(ptr %d, ptr %s, i64 1, i64 %0) #4
  ret void
}

define void @memmove_static(ptr %d, ptr %s, i64 %l) #0 {
entry:
  %0 = call i64 @llvm.objectsize.i64.p0(ptr %d, i1 false, i1 true, i1 false)
; GISEL: remark: <unknown>:0:0: Call to memmove. Memory operation size: 100 bytes.
  %call = call ptr @__memmove_chk(ptr %d, ptr %s, i64 100, i64 %0) #4
  ret void
}

define void @memmove_huge(ptr %d, ptr %s, i64 %l) #0 {
entry:
  %0 = call i64 @llvm.objectsize.i64.p0(ptr %d, i1 false, i1 true, i1 false)
; GISEL: remark: <unknown>:0:0: Call to memmove. Memory operation size: 100000 bytes.
  %call = call ptr @__memmove_chk(ptr %d, ptr %s, i64 100000, i64 %0) #4
  ret void
}

define void @memset_dynamic(ptr %d, i64 %l) #0 !dbg !38 {
entry:
  %0 = call i64 @llvm.objectsize.i64.p0(ptr %d, i1 false, i1 true, i1 false), !dbg !39
; GISEL: remark: memsize.c:22:3: Call to memset.{{$}}
  %call = call ptr @__memset_chk(ptr %d, i32 0, i64 %l, i64 %0) #4, !dbg !40
  ret void, !dbg !41
}

define void @memset_single(ptr %d, i64 %l) #0 !dbg !46 {
entry:
  %0 = call i64 @llvm.objectsize.i64.p0(ptr %d, i1 false, i1 true, i1 false), !dbg !47
; GISEL: remark: memsize.c:28:3: Call to memset. Memory operation size: 1 bytes.
  %call = call ptr @__memset_chk(ptr %d, i32 0, i64 1, i64 %0) #4, !dbg !48
  ret void, !dbg !49
}

define void @memset_static(ptr %d, i64 %l) #0 !dbg !50 {
entry:
  %0 = call i64 @llvm.objectsize.i64.p0(ptr %d, i1 false, i1 true, i1 false), !dbg !51
; GISEL: remark: memsize.c:31:3: Call to memset. Memory operation size: 100 bytes.
  %call = call ptr @__memset_chk(ptr %d, i32 0, i64 100, i64 %0) #4, !dbg !52
  ret void, !dbg !53
}

define void @memset_huge(ptr %d, i64 %l) #0 !dbg !54 {
entry:
  %0 = call i64 @llvm.objectsize.i64.p0(ptr %d, i1 false, i1 true, i1 false), !dbg !55
; GISEL: remark: memsize.c:34:3: Call to memset. Memory operation size: 100000 bytes.
  %call = call ptr @__memset_chk(ptr %d, i32 0, i64 100000, i64 %0) #4, !dbg !56
  ret void, !dbg !57
}

define void @memset_empty(ptr %d, i64 %l) #0 !dbg !42 {
entry:
  %0 = call i64 @llvm.objectsize.i64.p0(ptr %d, i1 false, i1 true, i1 false), !dbg !43
; GISEL: remark: memsize.c:25:3: Call to memset. Memory operation size: 0 bytes.
  %call = call ptr @__memset_chk(ptr %d, i32 0, i64 0, i64 %0) #4, !dbg !44
  ret void, !dbg !45
}

; YAML-LABEL: Function:        memcpy_empty
define void @memcpy_empty(ptr %d, ptr %s, i64 %l) #0 !dbg !19 {
entry:
  %0 = call i64 @llvm.objectsize.i64.p0(ptr %d, i1 false, i1 true, i1 false), !dbg !20
; GISEL: remark: memsize.c:7:3: Call to memcpy. Memory operation size: 0 bytes.
  %call = call ptr @__memcpy_chk(ptr %d, ptr %s, i64 0, i64 %0) #4, !dbg !21
  ret void, !dbg !22
}

; Emit remarks for memcpy, memmove, memset, bzero, bcopy with known constant
; sizes to an object of known size.
define void @known_call_with_dereferenceable_bytes(ptr dereferenceable(42) %dst, ptr dereferenceable(314) %src) {
; GISEL: Call to memset. Memory operation size: 1 bytes.
; GISEL-NOT:  Read Variables:
; GISEL-NEXT:  Written Variables: <unknown> (42 bytes).
; YAML:       --- !Analysis
; YAML:       gisel-irtranslator-memsize
; YAML:       Name:            MemoryOpIntrinsicCall
; YAML-LABEL: Function:        known_call_with_dereferenceable_bytes
; YAML-NEXT:  Args:
; YAML-NEXT:    - String:          'Call to '
; YAML-NEXT:    - Callee:          memset
; YAML-NEXT:    - String:          .
; YAML-NEXT:    - String:          ' Memory operation size: '
; YAML-NEXT:    - StoreSize:       '1'
; YAML-NEXT:    - String:          ' bytes.'
; YAML-NEXT:    - String:          "\n Written Variables: "
; YAML-NEXT:    - WVarName:        '<unknown>'
; YAML-NEXT:    - String:          ' ('
; YAML-NEXT:    - WVarSize:        '42'
; YAML-NEXT:    - String:          ' bytes)'
; YAML-NEXT:    - String:          .
; YAML-NEXT:    - String:          ' Inlined: '
; YAML-NEXT:    - StoreInlined:    'false'
; YAML-NEXT:    - String:          .
; YAML-NEXT:    - String:          ' Volatile: '
; YAML-NEXT:    - StoreVolatile:   'false'
; YAML-NEXT:    - String:          .
; YAML-NEXT:    - String:          ' Atomic: '
; YAML-NEXT:    - StoreAtomic:     'false'
; YAML-NEXT:    - String:          .
; YAML-NEXT:  ...
  call void @llvm.memset.p0.i64(ptr %dst, i8 0, i64 1, i1 false)

; GISEL: Call to memcpy. Memory operation size: 1 bytes.
; GISEL-NEXT:  Read Variables: <unknown> (314 bytes).
; GISEL-NEXT:  Written Variables: <unknown> (42 bytes).
; YAML:       --- !Analysis
; YAML:       gisel-irtranslator-memsize
; YAML:       Name:            MemoryOpIntrinsicCall
; YAML-LABEL: Function:        known_call_with_dereferenceable_bytes
; YAML-NEXT:  Args:
; YAML-NEXT:    - String:          'Call to '
; YAML-NEXT:    - Callee:          memcpy
; YAML-NEXT:    - String:          .
; YAML-NEXT:    - String:          ' Memory operation size: '
; YAML-NEXT:    - StoreSize:       '1'
; YAML-NEXT:    - String:          ' bytes.'
; YAML-NEXT:    - String:          "\n Read Variables: "
; YAML-NEXT:    - RVarName:        '<unknown>'
; YAML-NEXT:    - String:          ' ('
; YAML-NEXT:    - RVarSize:        '314'
; YAML-NEXT:    - String:          ' bytes)'
; YAML-NEXT:    - String:          .
; YAML-NEXT:    - String:          "\n Written Variables: "
; YAML-NEXT:    - WVarName:        '<unknown>'
; YAML-NEXT:    - String:          ' ('
; YAML-NEXT:    - WVarSize:        '42'
; YAML-NEXT:    - String:          ' bytes)'
; YAML-NEXT:    - String:          .
; YAML-NEXT:    - String:          ' Inlined: '
; YAML-NEXT:    - StoreInlined:    'false'
; YAML-NEXT:    - String:          .
; YAML-NEXT:    - String:          ' Volatile: '
; YAML-NEXT:    - StoreVolatile:   'false'
; YAML-NEXT:    - String:          .
; YAML-NEXT:    - String:          ' Atomic: '
; YAML-NEXT:    - StoreAtomic:     'false'
; YAML-NEXT:    - String:          .
; YAML-NEXT:  ...
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 1, i1 false)

; GISEL: Call to memmove. Memory operation size: 1 bytes.
; GISEL-NEXT:  Read Variables: <unknown> (314 bytes).
; GISEL-NEXT:  Written Variables: <unknown> (42 bytes).
; YAML:       --- !Analysis
; YAML:       gisel-irtranslator-memsize
; YAML:       Name:            MemoryOpIntrinsicCall
; YAML-LABEL: Function:        known_call_with_dereferenceable_bytes
; YAML-NEXT:  Args:
; YAML-NEXT:    - String:          'Call to '
; YAML-NEXT:    - Callee:          memmove
; YAML-NEXT:    - String:          .
; YAML-NEXT:    - String:          ' Memory operation size: '
; YAML-NEXT:    - StoreSize:       '1'
; YAML-NEXT:    - String:          ' bytes.'
; YAML-NEXT:    - String:          "\n Read Variables: "
; YAML-NEXT:    - RVarName:        '<unknown>'
; YAML-NEXT:    - String:          ' ('
; YAML-NEXT:    - RVarSize:        '314'
; YAML-NEXT:    - String:          ' bytes)'
; YAML-NEXT:    - String:          .
; YAML-NEXT:    - String:          "\n Written Variables: "
; YAML-NEXT:    - WVarName:        '<unknown>'
; YAML-NEXT:    - String:          ' ('
; YAML-NEXT:    - WVarSize:        '42'
; YAML-NEXT:    - String:          ' bytes)'
; YAML-NEXT:    - String:          .
; YAML-NEXT:    - String:          ' Inlined: '
; YAML-NEXT:    - StoreInlined:    'false'
; YAML-NEXT:    - String:          .
; YAML-NEXT:    - String:          ' Volatile: '
; YAML-NEXT:    - StoreVolatile:   'false'
; YAML-NEXT:    - String:          .
; YAML-NEXT:    - String:          ' Atomic: '
; YAML-NEXT:    - StoreAtomic:     'false'
; YAML-NEXT:    - String:          .
; YAML-NEXT:  ...
  call void @llvm.memmove.p0.p0.i64(ptr %dst, ptr %src, i64 1, i1 false)

; GISEL: Call to bzero. Memory operation size: 1 bytes.
; GISEL-NOT:  Read Variables:
; GISEL-NEXT:  Written Variables: <unknown> (42 bytes).
; YAML:       --- !Analysis
; YAML:       gisel-irtranslator-memsize
; YAML:       Name:            MemoryOpCall
; YAML-LABEL: Function:        known_call_with_dereferenceable_bytes
; YAML-NEXT:  Args:
; YAML-NEXT:    - String:          'Call to '
; YAML-NEXT:    - Callee:          bzero
; YAML-NEXT:    - String:          .
; YAML-NEXT:    - String:          ' Memory operation size: '
; YAML-NEXT:    - StoreSize:       '1'
; YAML-NEXT:    - String:          ' bytes.'
; YAML-NEXT:    - String:          "\n Written Variables: "
; YAML-NEXT:    - WVarName:        '<unknown>'
; YAML-NEXT:    - String:          ' ('
; YAML-NEXT:    - WVarSize:        '42'
; YAML-NEXT:    - String:          ' bytes)'
; YAML-NEXT:    - String:          .
; YAML-NEXT:  ...
  call void @bzero(ptr %dst, i64 1)

; GISEL: Call to bcopy. Memory operation size: 1 bytes.
; GISEL-NEXT:  Read Variables: <unknown> (314 bytes).
; GISEL-NEXT:  Written Variables: <unknown> (42 bytes).
; YAML:       --- !Analysis
; YAML:       gisel-irtranslator-memsize
; YAML:       Name:            MemoryOpCall
; YAML-LABEL: Function:        known_call_with_dereferenceable_bytes
; YAML-NEXT:  Args:
; YAML-NEXT:    - String:          'Call to '
; YAML-NEXT:    - Callee:          bcopy
; YAML-NEXT:    - String:          .
; YAML-NEXT:    - String:          ' Memory operation size: '
; YAML-NEXT:    - StoreSize:       '1'
; YAML-NEXT:    - String:          ' bytes.'
; YAML-NEXT:    - String:          "\n Read Variables: "
; YAML-NEXT:    - RVarName:        '<unknown>'
; YAML-NEXT:    - String:          ' ('
; YAML-NEXT:    - RVarSize:        '314'
; YAML-NEXT:    - String:          ' bytes)'
; YAML-NEXT:    - String:          .
; YAML-NEXT:    - String:          "\n Written Variables: "
; YAML-NEXT:    - WVarName:        '<unknown>'
; YAML-NEXT:    - String:          ' ('
; YAML-NEXT:    - WVarSize:        '42'
; YAML-NEXT:    - String:          ' bytes)'
; YAML-NEXT:    - String:          .
; YAML-NEXT:  ...
  call void @bcopy(ptr %dst, ptr %src, i64 1)
  ret void
}

@dropbear = external unnamed_addr constant [3 x i8], align 1
@koala = external unnamed_addr constant [7 x i8], align 1

define void @slicePun() {
bb:
; GISEL: remark: <unknown>:0:0: Call to memcpy. Memory operation size: 24 bytes.{{$}}
; GISEL-NEXT: Read Variables: koala (56 bytes).
; GISEL-NEXT: Written Variables: dropbear (24 bytes).
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 @dropbear,
                                            ptr @koala, i64 24, i1 false)
  ret void
}

attributes #0 = { noinline nounwind ssp uwtable "frame-pointer"="non-leaf" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-a7" "target-features"="+aes,+crypto,+fp-armv8,+neon,+sha2,+zcm,+zcz" }
attributes #1 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-a7" "target-features"="+aes,+crypto,+fp-armv8,+neon,+sha2,+zcm,+zcz" }
attributes #2 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #3 = { argmemonly nofree nosync nounwind willreturn }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9}
!llvm.dbg.cu = !{!10}
!llvm.ident = !{!13}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 0]}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 8, !"branch-target-enforcement", i32 0}
!4 = !{i32 8, !"sign-return-address", i32 0}
!5 = !{i32 8, !"sign-return-address-all", i32 0}
!6 = !{i32 8, !"sign-return-address-with-bkey", i32 0}
!7 = !{i32 7, !"PIC Level", i32 2}
!8 = !{i32 7, !"uwtable", i32 1}
!9 = !{i32 7, !"frame-pointer", i32 1}
!10 = distinct !DICompileUnit(language: DW_LANG_C99, file: !11, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !12, splitDebugInlining: false, nameTableKind: None, sysroot: "/")
!11 = !DIFile(filename: "memsize.c", directory: "")
!12 = !{}
!13 = !{!"clang"}
!14 = distinct !DISubprogram(name: "memcpy_dynamic", scope: !11, file: !11, line: 3, type: !15, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !10, retainedNodes: !12)
!15 = !DISubroutineType(types: !12)
!16 = !DILocation(line: 4, column: 36, scope: !14)
!17 = !DILocation(line: 4, column: 3, scope: !14)
!18 = !DILocation(line: 5, column: 1, scope: !14)
!19 = distinct !DISubprogram(name: "memcpy_empty", scope: !11, file: !11, line: 6, type: !15, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !10, retainedNodes: !12)
!20 = !DILocation(line: 7, column: 36, scope: !19)
!21 = !DILocation(line: 7, column: 3, scope: !19)
!22 = !DILocation(line: 8, column: 1, scope: !19)
!23 = distinct !DISubprogram(name: "memcpy_single", scope: !11, file: !11, line: 9, type: !15, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !10, retainedNodes: !12)
!24 = !DILocation(line: 10, column: 36, scope: !23)
!25 = !DILocation(line: 10, column: 3, scope: !23)
!26 = !DILocation(line: 11, column: 1, scope: !23)
!27 = distinct !DISubprogram(name: "memcpy_static", scope: !11, file: !11, line: 12, type: !15, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !10, retainedNodes: !12)
!28 = !DILocation(line: 13, column: 38, scope: !27)
!29 = !DILocation(line: 13, column: 3, scope: !27)
!30 = !DILocation(line: 14, column: 1, scope: !27)
!31 = distinct !DISubprogram(name: "memcpy_huge", scope: !11, file: !11, line: 15, type: !15, scopeLine: 15, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !10, retainedNodes: !12)
!32 = !DILocation(line: 16, column: 41, scope: !31)
!33 = !DILocation(line: 16, column: 3, scope: !31)
!34 = !DILocation(line: 17, column: 1, scope: !31)
!35 = distinct !DISubprogram(name: "memcpy_inline", scope: !11, file: !11, line: 18, type: !15, scopeLine: 18, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !10, retainedNodes: !12)
!36 = !DILocation(line: 19, column: 3, scope: !35)
!37 = !DILocation(line: 20, column: 1, scope: !35)
!38 = distinct !DISubprogram(name: "memset_dynamic", scope: !11, file: !11, line: 21, type: !15, scopeLine: 21, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !10, retainedNodes: !12)
!39 = !DILocation(line: 22, column: 36, scope: !38)
!40 = !DILocation(line: 22, column: 3, scope: !38)
!41 = !DILocation(line: 23, column: 1, scope: !38)
!42 = distinct !DISubprogram(name: "memset_empty", scope: !11, file: !11, line: 24, type: !15, scopeLine: 24, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !10, retainedNodes: !12)
!43 = !DILocation(line: 25, column: 36, scope: !42)
!44 = !DILocation(line: 25, column: 3, scope: !42)
!45 = !DILocation(line: 26, column: 1, scope: !42)
!46 = distinct !DISubprogram(name: "memset_single", scope: !11, file: !11, line: 27, type: !15, scopeLine: 27, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !10, retainedNodes: !12)
!47 = !DILocation(line: 28, column: 36, scope: !46)
!48 = !DILocation(line: 28, column: 3, scope: !46)
!49 = !DILocation(line: 29, column: 1, scope: !46)
!50 = distinct !DISubprogram(name: "memset_static", scope: !11, file: !11, line: 30, type: !15, scopeLine: 30, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !10, retainedNodes: !12)
!51 = !DILocation(line: 31, column: 38, scope: !50)
!52 = !DILocation(line: 31, column: 3, scope: !50)
!53 = !DILocation(line: 32, column: 1, scope: !50)
!54 = distinct !DISubprogram(name: "memset_huge", scope: !11, file: !11, line: 33, type: !15, scopeLine: 33, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !10, retainedNodes: !12)
!55 = !DILocation(line: 34, column: 41, scope: !54)
!56 = !DILocation(line: 34, column: 3, scope: !54)
!57 = !DILocation(line: 35, column: 1, scope: !54)
!58 = distinct !DISubprogram(name: "auto_init", scope: !11, file: !11, line: 37, type: !15, scopeLine: 37, spFlags: DISPFlagDefinition, unit: !10, retainedNodes: !12)
