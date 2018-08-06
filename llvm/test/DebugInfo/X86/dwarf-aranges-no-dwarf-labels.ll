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
source_filename = "test/DebugInfo/X86/dwarf-aranges-no-dwarf-labels.ll"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@global = global i32 2, align 4, !dbg !0

; Function Attrs: nounwind readnone uwtable
define i32 @_Z3fooi(i32 %bar) #0 !dbg !9 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %bar, metadata !13, metadata !14), !dbg !15
  ret i32 %bar, !dbg !15
}

; Function Attrs: nounwind readnone uwtable
define i32 @_Z4foo2i(i32 %bar2) #0 !dbg !16 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %bar2, metadata !18, metadata !14), !dbg !19
  ret i32 %bar2, !dbg !19
}

; Function Attrs: nounwind readonly uwtable
define i32 @main() #1 !dbg !20 {
entry:
  %call = tail call i32 @_Z3fooi(i32 2), !dbg !23
  %call1 = tail call i32 @_Z4foo2i(i32 1), !dbg !23
  %add = add nsw i32 %call1, %call, !dbg !23
  %0 = load i32, i32* @global, align 4, !dbg !23, !tbaa !24
  %add2 = add nsw i32 %add, %0, !dbg !23
  ret i32 %add2, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readonly uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!7, !8}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "global", scope: null, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "tmp/debug_ranges/a.cc", directory: "/")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.4 (191881)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!0}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 1, !"Debug Info Version", i32 3}
!9 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !2, file: !2, line: 2, type: !10, isLocal: false, isDefinition: true, scopeLine: 2, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !4, retainedNodes: !12)
!10 = !DISubroutineType(types: !11)
!11 = !{!3, !3}
!12 = !{!13}
!13 = !DILocalVariable(name: "bar", arg: 1, scope: !9, file: !2, line: 2, type: !3)
!14 = !DIExpression()
!15 = !DILocation(line: 2, scope: !9)
!16 = distinct !DISubprogram(name: "foo2", linkageName: "_Z4foo2i", scope: !2, file: !2, line: 3, type: !10, isLocal: false, isDefinition: true, scopeLine: 3, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !4, retainedNodes: !17)
!17 = !{!18}
!18 = !DILocalVariable(name: "bar2", arg: 1, scope: !16, file: !2, line: 3, type: !3)
!19 = !DILocation(line: 3, scope: !16)
!20 = distinct !DISubprogram(name: "main", scope: !2, file: !2, line: 5, type: !21, isLocal: false, isDefinition: true, scopeLine: 5, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !4, retainedNodes: !5)
!21 = !DISubroutineType(types: !22)
!22 = !{!3}
!23 = !DILocation(line: 6, scope: !20)
!24 = !{!25, !25, i64 0}
!25 = !{!"int", !26}
!26 = !{!"omnipotent char", !27}
!27 = !{!"Simple C/C++ TBAA"}

