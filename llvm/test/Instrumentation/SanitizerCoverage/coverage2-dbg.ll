; Test that coverage instrumentation does not lose debug location.

; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=2 -S | FileCheck %s

; C++ source:
; 1: void foo(int *a) {
; 2:     if (a)
; 3:         *a = 0;
; 4: }
; clang++ if.cc -O3 -g -S -emit-llvm
; and add sanitize_address to @_Z3fooPi


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Check that __sanitizer_cov call has !dgb pointing to the beginning
; of appropriate basic blocks.
; CHECK-LABEL:_Z3fooPi
; CHECK: call void @__sanitizer_cov{{.*}}(ptr{{.*}}) #{{.*}}, !dbg [[A:!.*]]
; CHECK: call void @__sanitizer_cov{{.*}}(ptr{{.*}}) #{{.*}}, !dbg [[B:!.*]]
; CHECK: ret void
; CHECK: [[A]] = !DILocation(line: 1, scope: !{{.*}})
; CHECK: [[B]] = !DILocation(line: 3, column: 5, scope: !{{.*}})

define void @_Z3fooPi(ptr %a) #0 !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata ptr %a, i64 0, metadata !11, metadata !DIExpression()), !dbg !15
  %tobool = icmp eq ptr %a, null, !dbg !16
  br i1 %tobool, label %if.end, label %if.then, !dbg !16

if.then:                                          ; preds = %entry
  store i32 0, ptr %a, align 4, !dbg !18, !tbaa !19
  br label %if.end, !dbg !18

if.end:                                           ; preds = %entry, %if.then
  ret void, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" sanitize_address}
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.6.0 (217079)", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "if.cc", directory: "FOO")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooPi", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, retainedNodes: !10)
!5 = !DIFile(filename: "if.cc", directory: "FOO")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !9)
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{!11}
!11 = !DILocalVariable(name: "a", line: 1, arg: 1, scope: !4, file: !5, type: !8)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{!"clang version 3.6.0 (217079)"}
!15 = !DILocation(line: 1, column: 15, scope: !4)
!16 = !DILocation(line: 2, column: 7, scope: !17)
!17 = distinct !DILexicalBlock(line: 2, column: 7, file: !1, scope: !4)
!18 = !DILocation(line: 3, column: 5, scope: !17)
!19 = !{!20, !20, i64 0}
!20 = !{!"int", !21, i64 0}
!21 = !{!"omnipotent char", !22, i64 0}
!22 = !{!"Simple C/C++ TBAA"}
!23 = !DILocation(line: 4, column: 1, scope: !4)
