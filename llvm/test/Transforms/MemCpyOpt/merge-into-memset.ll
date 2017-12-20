; RUN: opt < %s -memcpyopt -S | FileCheck %s
; Update cached non-local dependence information when merging stores into memset.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Don't delete the memcpy in %if.then, even though it depends on an instruction
; which will be deleted.

; CHECK-LABEL: @foo
define void @foo(i1 %c, i8* %d, i8* %e, i8* %f) {
entry:
  %tmp = alloca [50 x i8], align 8
  %tmp4 = bitcast [50 x i8]* %tmp to i8*
  %tmp1 = getelementptr inbounds i8, i8* %tmp4, i64 1
  call void @llvm.memset.p0i8.i64(i8* nonnull %d, i8 0, i64 10, i32 1, i1 false), !dbg !5
  store i8 0, i8* %tmp4, align 8, !dbg !5
; CHECK: call void @llvm.memset.p0i8.i64(i8* nonnull %d, i8 0, i64 10, i32 1, i1 false), !dbg !5
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull %tmp1, i8* nonnull %d, i64 10, i32 1, i1 false)
  br i1 %c, label %if.then, label %exit

if.then:
; CHECK: if.then:
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %f, i8* nonnull %tmp4, i64 30, i32 8, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %f, i8* nonnull %tmp4, i64 30, i32 8, i1 false)
  br label %exit

exit:
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i32, i1)
declare void @llvm.memset.p0i8.i64(i8*, i8, i64, i32, i1)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.rs", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DILocation(line: 8, column: 5, scope: !6)
!6 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 5, type: !7, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
