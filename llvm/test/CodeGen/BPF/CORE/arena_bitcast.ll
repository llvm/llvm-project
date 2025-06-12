; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -mcpu=v3 -filetype=asm -o - %t1 | FileCheck %s
; Source code:
;   struct lock_t {
;     int counter;
;   } __attribute__((preserve_access_index));
;
;   #define __arena __attribute__((address_space(1)))
;   int test(struct lock_t __arena *lock, unsigned val)
;   {
;     return __sync_val_compare_and_swap((&lock->counter), val, 1);
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes arena_bitcast.c

target triple = "bpf"

%struct.lock_t = type { i32 }

; Function Attrs: nounwind
define dso_local i32 @test(ptr addrspace(1) noundef %lock, i32 noundef %val) #0 !dbg !7 {
entry:
  %lock.addr = alloca ptr addrspace(1), align 8
  %val.addr = alloca i32, align 4
  store ptr addrspace(1) %lock, ptr %lock.addr, align 8, !tbaa !19
    #dbg_declare(ptr %lock.addr, !17, !DIExpression(), !24)
  store i32 %val, ptr %val.addr, align 4, !tbaa !25
    #dbg_declare(ptr %val.addr, !18, !DIExpression(), !27)
  %0 = load ptr addrspace(1), ptr %lock.addr, align 8, !dbg !28, !tbaa !19
  %1 = call ptr addrspace(1) @llvm.preserve.struct.access.index.p1.p1(ptr addrspace(1) elementtype(%struct.lock_t) %0, i32 0, i32 0), !dbg !29, !llvm.preserve.access.index !12
  %2 = load i32, ptr %val.addr, align 4, !dbg !30, !tbaa !25
  %3 = cmpxchg ptr addrspace(1) %1, i32 %2, i32 1 seq_cst seq_cst, align 4, !dbg !31
  %4 = extractvalue { i32, i1 } %3, 0, !dbg !31
  ret i32 %4, !dbg !32
}
; CHECK:  r1 = addr_space_cast(r1, 0, 1)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr addrspace(1) @llvm.preserve.struct.access.index.p1.p1(ptr addrspace(1), i32 immarg, i32 immarg) #1

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 21.0.0git (https://github.com/llvm/llvm-project.git 7d4d8509cbec7eecd8aaf2510015b54bc5c173e1)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "arena_bitcast.c", directory: "/root/home/yhs/tests/arena/simple", checksumkind: CSK_MD5, checksum: "51cb51c1fc09d3033dbd9aee9044dc9b")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!"clang version 21.0.0git (https://github.com/llvm/llvm-project.git 7d4d8509cbec7eecd8aaf2510015b54bc5c173e1)"}
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 6, type: !8, scopeLine: 7, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11, !15}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "lock_t", file: !1, line: 1, size: 32, elements: !13)
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "counter", scope: !12, file: !1, line: 2, baseType: !10, size: 32)
!15 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!16 = !{!17, !18}
!17 = !DILocalVariable(name: "lock", arg: 1, scope: !7, file: !1, line: 6, type: !11)
!18 = !DILocalVariable(name: "val", arg: 2, scope: !7, file: !1, line: 6, type: !15)
!19 = !{!20, !20, i64 0}
!20 = !{!"p1 _ZTS6lock_t", !21, i64 0}
!21 = !{!"any pointer", !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C/C++ TBAA"}
!24 = !DILocation(line: 6, column: 33, scope: !7)
!25 = !{!26, !26, i64 0}
!26 = !{!"int", !22, i64 0}
!27 = !DILocation(line: 6, column: 48, scope: !7)
!28 = !DILocation(line: 8, column: 40, scope: !7)
!29 = !DILocation(line: 8, column: 46, scope: !7)
!30 = !DILocation(line: 8, column: 56, scope: !7)
!31 = !DILocation(line: 8, column: 10, scope: !7)
!32 = !DILocation(line: 8, column: 3, scope: !7)
