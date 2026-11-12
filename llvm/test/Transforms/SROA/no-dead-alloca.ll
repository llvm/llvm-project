; RUN: opt -S -p sroa < %s | FileCheck %s

; Test that SROA removes allocas with only droppable uses left

; CHECK-LABEL: @no_dead_alloca
; CHECK-NOT: %items = alloca
; CHECK-NOT: #dbg_declare(ptr %items,

define i64 @no_dead_alloca(i1 %cond, ptr %other) !dbg !5 {
entry:
  %items = alloca { ptr, ptr }, align 8
    #dbg_declare(ptr %items, !8, !DIExpression(), !9)
  call void @llvm.memset.p0.i64(ptr %items, i8 0, i64 16, i1 false)
  br i1 %cond, label %local, label %join

local:
  %n = getelementptr i8, ptr %items, i64 8
  call void @llvm.assume(i1 true) [ "separate_storage"(ptr %items, ptr %other) ]
  br label %join

join:
  %p = phi ptr [ %n, %local ], [ %other, %entry ]
  %value = load i64, ptr %p, align 8
  store i64 0, ptr %p, align 8
  ret i64 %value, !dbg !9
}

declare void @llvm.assume(i1)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1 immarg)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "reduced", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cc", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !2)
!5 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Vec", size: 128, elements: !2)
!8 = !DILocalVariable(name: "items", scope: !5, file: !1, line: 1, type: !6)
!9 = !DILocation(line: 1, column: 1, scope: !5)
