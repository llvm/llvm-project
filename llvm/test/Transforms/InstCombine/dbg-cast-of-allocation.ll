; RUN: opt -opaque-pointers=0 -S --passes=instcombine %s | FileCheck %s

; https://github.com/llvm/llvm-project/issues/56807
%si16 = type { i8, i8 }

declare void @foo(i16* %pixels)

declare void @llvm.dbg.declare(metadata, metadata, metadata)

; CHECK-LABEL: @toplevel(
; CHECK:  entry:
; CHECK-NEXT:    %pixels = alloca i16
; CHECK-NEXT:    call void @llvm.dbg.declare(metadata i16* %pixels, metadata ![[MD:[0-9]+]], metadata !DIExpression()), !dbg ![[DBG:[0-9]+]]
; CHECK-NEXT:    call void @foo(i16* nonnull %pixels)
; CHECK-NEXT:    ret void
define dso_local void @toplevel() {
entry:
  %pixels = alloca %si16
  call void @llvm.dbg.declare(metadata %si16* %pixels, metadata !11, metadata !DIExpression()), !dbg !12
  %arraydecay = bitcast %si16* %pixels to i16*
  call void @foo(i16* nonnull %arraydecay)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}


!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 11.1.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/path/to/test_cpp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!7 = distinct !DISubprogram(name: "toplevel", linkageName: "_Z8toplevelv", scope: !1, file: !1, line: 9, type: !8, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
; CHECK: ![[MD]] = !DILocalVariable(name: "pixels"
!11 = !DILocalVariable(name: "pixels", arg: 1, scope: !7, file: !1, line: 9, type: !10)
; CHECK: ![[DBG]] = !DILocation(line: 9, column: 16,
!12 = !DILocation(line: 9, column: 16, scope: !7)
