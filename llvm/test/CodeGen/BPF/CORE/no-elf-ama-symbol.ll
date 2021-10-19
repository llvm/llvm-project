; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=obj -o - %t1 | llvm-readelf -s - | FileCheck -check-prefixes=CHECK %s
; RUN: llc -filetype=obj -addrsig -o - %t1 | llvm-readelf -s - | FileCheck -check-prefixes=CHECK %s
;
; Source Code:
;   struct tt { int a; } __attribute__((preserve_access_index));
;   int test(struct tt *arg) {
;     return arg->a;
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes t.c

target triple = "bpf"

%struct.tt = type { i32 }

; Function Attrs: nounwind readonly
define dso_local i32 @test(%struct.tt* readonly %arg) local_unnamed_addr #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata %struct.tt* %arg, metadata !16, metadata !DIExpression()), !dbg !17
  %0 = tail call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.tts(%struct.tt* elementtype(%struct.tt) %arg, i32 0, i32 0), !dbg !18, !llvm.preserve.access.index !12
  %1 = load i32, i32* %0, align 4, !dbg !18, !tbaa !19
  ret i32 %1, !dbg !24
}

; CHECK-NOT: llvm.tt:0:0$0:0

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.tts(%struct.tt*, i32, i32) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readnone speculatable}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git 947f9692440836dcb8d88b74b69dd379d85974ce)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/bug")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git 947f9692440836dcb8d88b74b69dd379d85974ce)"}
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !15)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "tt", file: !1, line: 1, size: 32, elements: !13)
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !12, file: !1, line: 1, baseType: !10, size: 32)
!15 = !{!16}
!16 = !DILocalVariable(name: "arg", arg: 1, scope: !7, file: !1, line: 2, type: !11)
!17 = !DILocation(line: 0, scope: !7)
!18 = !DILocation(line: 3, column: 15, scope: !7)
!19 = !{!20, !21, i64 0}
!20 = !{!"tt", !21, i64 0}
!21 = !{!"int", !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C/C++ TBAA"}
!24 = !DILocation(line: 3, column: 3, scope: !7)
