; RUN: opt < %s --passes='print<source-expr>' -disable-output  2>&1 | FileCheck %s

; ModuleID = '../cpp/st.cpp'
source_filename = "../cpp/st.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.foo = type { i32, [5 x i32] }

$_ZN3fooC2Ev = comdat any

@constinit = private constant [5 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5], align 4

; CHECK-LABEL: Load Store Expression _Z4funcv
define dso_local void @_Z4funcv() #0 !dbg !19 {
entry:
; CHECK: %obj = obj
  %obj = alloca %struct.foo, align 4
; CHECK: %s = s
  %s = alloca i32, align 4
  call void @llvm.dbg.declare(metadata ptr %obj, metadata !23, metadata !DIExpression()), !dbg !24
  call void @_ZN3fooC2Ev(ptr noundef nonnull align 4 dereferenceable(24) %obj) #4, !dbg !24
; CHECK: %a = unknown
  %a = getelementptr inbounds %struct.foo, ptr %obj, i32 0, i32 0, !dbg !25
  store i32 20, ptr %a, align 4, !dbg !26
  call void @llvm.dbg.declare(metadata ptr %s, metadata !27, metadata !DIExpression()), !dbg !28
; CHECK: %arr = unknown
  %arr = getelementptr inbounds %struct.foo, ptr %obj, i32 0, i32 1, !dbg !29
; CHECK: %arrayidx = unknown
  %arrayidx = getelementptr inbounds [5 x i32], ptr %arr, i64 0, i64 3, !dbg !30
; CHECK: %0 = unknown
  %0 = load i32, ptr %arrayidx, align 4, !dbg !30
  store i32 %0, ptr %s, align 4, !dbg !28
  ret void, !dbg !31
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN3fooC2Ev(ptr noundef nonnull align 4 dereferenceable(24) %this) unnamed_addr #2 comdat align 2 !dbg !32 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %this.addr, metadata !37, metadata !DIExpression()), !dbg !39
  %this1 = load ptr, ptr %this.addr, align 8
  %arr = getelementptr inbounds %struct.foo, ptr %this1, i32 0, i32 1, !dbg !40
  %arrayinit.begin = getelementptr inbounds [5 x i32], ptr %arr, i64 0, i64 0, !dbg !41
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %arr, ptr align 4 @constinit, i64 20, i1 false), !dbg !41
  ret void, !dbg !42
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #3

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12, !13, !14, !15, !16, !17}
!llvm.ident = !{!18}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 17.0.0 (https://github.com/phyBrackets/llvm-project-1.git 0d3edc0be92f6c8f60d49772b74e456f285483e6)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "../cpp/st.cpp", directory: "/home/shivam/llvm-project-1", checksumkind: CSK_MD5, checksum: "19ed7b3e714703c8f231838bfe106739")
!2 = !{!3}
!3 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !1, line: 1, size: 192, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !4, identifier: "_ZTS3foo")
!4 = !{!5, !7}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !3, file: !1, line: 2, baseType: !6, size: 32)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "arr", scope: !3, file: !1, line: 3, baseType: !8, size: 160, offset: 32)
!8 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, size: 160, elements: !9)
!9 = !{!10}
!10 = !DISubrange(count: 5)
!11 = !{i32 7, !"Dwarf Version", i32 5}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{i32 8, !"PIC Level", i32 2}
!15 = !{i32 7, !"PIE Level", i32 2}
!16 = !{i32 7, !"uwtable", i32 2}
!17 = !{i32 7, !"frame-pointer", i32 2}
!18 = !{!"clang version 17.0.0 (https://github.com/phyBrackets/llvm-project-1.git 0d3edc0be92f6c8f60d49772b74e456f285483e6)"}
!19 = distinct !DISubprogram(name: "func", linkageName: "_Z4funcv", scope: !1, file: !1, line: 6, type: !20, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !22)
!20 = !DISubroutineType(types: !21)
!21 = !{null}
!22 = !{}
!23 = !DILocalVariable(name: "obj", scope: !19, file: !1, line: 7, type: !3)
!24 = !DILocation(line: 7, column: 9, scope: !19)
!25 = !DILocation(line: 8, column: 9, scope: !19)
!26 = !DILocation(line: 8, column: 11, scope: !19)
!27 = !DILocalVariable(name: "s", scope: !19, file: !1, line: 9, type: !6)
!28 = !DILocation(line: 9, column: 9, scope: !19)
!29 = !DILocation(line: 9, column: 17, scope: !19)
!30 = !DILocation(line: 9, column: 13, scope: !19)
!31 = !DILocation(line: 10, column: 1, scope: !19)
!32 = distinct !DISubprogram(name: "foo", linkageName: "_ZN3fooC2Ev", scope: !3, file: !1, line: 1, type: !33, scopeLine: 1, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, declaration: !36, retainedNodes: !22)
!33 = !DISubroutineType(types: !34)
!34 = !{null, !35}
!35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!36 = !DISubprogram(name: "foo", scope: !3, type: !33, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: 0)
!37 = !DILocalVariable(name: "this", arg: 1, scope: !32, type: !38, flags: DIFlagArtificial | DIFlagObjectPointer)
!38 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64)
!39 = !DILocation(line: 0, scope: !32)
!40 = !DILocation(line: 3, column: 9, scope: !32)
!41 = !DILocation(line: 3, column: 16, scope: !32)
!42 = !DILocation(line: 1, column: 8, scope: !32)
