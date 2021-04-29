; RUN: opt -O2 -pass-remarks-analysis=openmp-opt -enable-new-pm < %s 2>&1 | FileCheck %s --check-prefix=MODULE
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"

@.str = private unnamed_addr constant [13 x i8] c"Alloc Shared\00", align 1

; MODULE: remark: openmp_opt_module.c:5:7: Found thread data sharing on the GPU. Expect degraded performance due to data globalization.

define void @foo() {
entry:
  %x = call i8* @__kmpc_data_sharing_push_stack(i64 4, i16 0), !dbg !7
  %x_on_stack = bitcast i8* %x to i32*
  %0 = bitcast i32* %x_on_stack to i8*
  call void @use(i8* %0)
  call void @__kmpc_data_sharing_pop_stack(i8* %x)
  ret void
}

define void @use(i8* %0) {
entry:
  %.addr = alloca i8*, align 8
  store i8* %0, i8** %.addr, align 8
  ret void
}

define internal i8* @__kmpc_data_sharing_push_stack(i64 %DataSize, i16 %shared) {
entry:
  %call = call i8* @_Z10SafeMallocmPKc(i64 %DataSize, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i64 0, i64 0)) #11
  ret i8* %call
}

; Function Attrs: convergent nounwind mustprogress
declare i8* @_Z10SafeMallocmPKc(i64 %size, i8* nocapture readnone %msg)

declare void @__kmpc_data_sharing_pop_stack(i8*)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "openmp_opt_module.c", directory: "/tmp/openmp_opt_module.c")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!6 = !DISubroutineType(types: !2)
!7 = !DILocation(line: 5, column: 7, scope: !5)
