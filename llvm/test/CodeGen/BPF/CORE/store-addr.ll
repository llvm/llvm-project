; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck %s
; RUN: opt -passes='default<O2>' %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck %s
; Source code:
;   struct t {
;     int a;
;   } __attribute__((preserve_access_index));
;   int foo(ptr);
;   int test(struct t *arg) {
;       long param[1];
;       param[0] = (long)&arg->a;
;       return foo(param);
;   }
; Compiler flag to generate IR:
;   clang -target bpf -S -O2 -g -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpf"

%struct.t = type { i32 }

; Function Attrs: nounwind
define dso_local i32 @test(ptr %arg) local_unnamed_addr #0 !dbg !14 {
entry:
  %param = alloca [1 x i64], align 8
  call void @llvm.dbg.value(metadata ptr %arg, metadata !22, metadata !DIExpression()), !dbg !27
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %param) #5, !dbg !28
  call void @llvm.dbg.declare(metadata ptr %param, metadata !23, metadata !DIExpression()), !dbg !29
  %0 = tail call ptr @llvm.preserve.struct.access.index.p0.p0.ts(ptr elementtype(%struct.t) %arg, i32 0, i32 0), !dbg !30, !llvm.preserve.access.index !18
  %1 = ptrtoint ptr %0 to i64, !dbg !31
  store i64 %1, ptr %param, align 8, !dbg !33, !tbaa !34
  %call = call i32 @foo(ptr nonnull %param) #5, !dbg !38
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %param) #5, !dbg !39
  ret i32 %call, !dbg !40
}

; CHECK:  r[[OFFSET:[0-9]+]] = 0
; CHECK:  r1 += r[[OFFSET]]
; CHECK:  *(u64 *)(r10 - 8) = r1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #2

; Function Attrs: nounwind readnone
declare ptr @llvm.preserve.struct.access.index.p0.p0.ts(ptr, i32, i32) #3

declare !dbg !5 dso_local i32 @foo(ptr) local_unnamed_addr #4

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #2

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind }
attributes #3 = { nounwind readnone }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git 4f995959a05ae94cc4f9cc80035f7e4b3ecd2d88)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/core")
!2 = !{}
!3 = !{!4, !5}
!4 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!5 = !DISubprogram(name: "foo", scope: !1, file: !1, line: 4, type: !6, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!10 = !{i32 7, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 4f995959a05ae94cc4f9cc80035f7e4b3ecd2d88)"}
!14 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 5, type: !15, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !21)
!15 = !DISubroutineType(types: !16)
!16 = !{!8, !17}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: !1, line: 1, size: 32, elements: !19)
!19 = !{!20}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !18, file: !1, line: 2, baseType: !8, size: 32)
!21 = !{!22, !23}
!22 = !DILocalVariable(name: "arg", arg: 1, scope: !14, file: !1, line: 5, type: !17)
!23 = !DILocalVariable(name: "param", scope: !14, file: !1, line: 6, type: !24)
!24 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, size: 64, elements: !25)
!25 = !{!26}
!26 = !DISubrange(count: 1)
!27 = !DILocation(line: 0, scope: !14)
!28 = !DILocation(line: 6, column: 5, scope: !14)
!29 = !DILocation(line: 6, column: 10, scope: !14)
!30 = !DILocation(line: 7, column: 28, scope: !14)
!31 = !DILocation(line: 7, column: 16, scope: !14)
!32 = !DILocation(line: 7, column: 5, scope: !14)
!33 = !DILocation(line: 7, column: 14, scope: !14)
!34 = !{!35, !35, i64 0}
!35 = !{!"long", !36, i64 0}
!36 = !{!"omnipotent char", !37, i64 0}
!37 = !{!"Simple C/C++ TBAA"}
!38 = !DILocation(line: 8, column: 12, scope: !14)
!39 = !DILocation(line: 9, column: 1, scope: !14)
!40 = !DILocation(line: 8, column: 5, scope: !14)
