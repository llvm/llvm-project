; RUN: llc -generate-arange-section < %s | FileCheck %s

; CHECK: .short  2 # DWARF Arange version number
; CHECK: # Segment Size
; CHECK-NOT: debug_loc
; CHECK: .quad global
; CHECK-NOT: debug_loc
; CHECK: # ARange terminator

; --- Source code ---
; Generated with "clang -g -O1 -S -emit-llvm"

; int global = 2;
; int foo(int bar) { return bar; }
; int foo2(int bar2) { return bar2; }

; int main() {
;   return foo(2) + foo2(1) + global;
; }


; ModuleID = 'tmp/debug_ranges/a.cc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@global = global i32 2, align 4

; Function Attrs: nounwind readnone uwtable
define i32 @_Z3fooi(i32 %bar) #0 !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %bar, i64 0, metadata !10, metadata !DIExpression()), !dbg !20
  ret i32 %bar, !dbg !20
}

; Function Attrs: nounwind readnone uwtable
define i32 @_Z4foo2i(i32 %bar2) #0 !dbg !11 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %bar2, i64 0, metadata !13, metadata !DIExpression()), !dbg !21
  ret i32 %bar2, !dbg !21
}

; Function Attrs: nounwind readonly uwtable
define i32 @main() #1 !dbg !14 {
entry:
  %call = tail call i32 @_Z3fooi(i32 2), !dbg !22
  %call1 = tail call i32 @_Z4foo2i(i32 1), !dbg !22
  %add = add nsw i32 %call1, %call, !dbg !22
  %0 = load i32, i32* @global, align 4, !dbg !22, !tbaa !23
  %add2 = add nsw i32 %add, %0, !dbg !22
  ret i32 %add2, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readonly uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!19, !26}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.4 (191881)", isOptimized: true, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !17, imports: !2)
!1 = !DIFile(filename: "tmp/debug_ranges/a.cc", directory: "/")
!2 = !{}
!3 = !{!4, !11, !14}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 2, file: !1, scope: !5, type: !6, variables: !9)
!5 = !DIFile(filename: "tmp/debug_ranges/a.cc", directory: "/")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DILocalVariable(name: "bar", line: 2, arg: 1, scope: !4, file: !5, type: !8)
!11 = distinct !DISubprogram(name: "foo2", linkageName: "_Z4foo2i", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 3, file: !1, scope: !5, type: !6, variables: !12)
!12 = !{!13}
!13 = !DILocalVariable(name: "bar2", line: 3, arg: 1, scope: !11, file: !5, type: !8)
!14 = distinct !DISubprogram(name: "main", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 5, file: !1, scope: !5, type: !15, variables: !2)
!15 = !DISubroutineType(types: !16)
!16 = !{!8}
!17 = !{!18}
!18 = !DIGlobalVariable(name: "global", line: 1, isLocal: false, isDefinition: true, scope: null, file: !5, type: !8, variable: i32* @global)
!19 = !{i32 2, !"Dwarf Version", i32 4}
!20 = !DILocation(line: 2, scope: !4)
!21 = !DILocation(line: 3, scope: !11)
!22 = !DILocation(line: 6, scope: !14)
!23 = !{!"int", !24}
!24 = !{!"omnipotent char", !25}
!25 = !{!"Simple C/C++ TBAA"}
!26 = !{i32 1, !"Debug Info Version", i32 3}
