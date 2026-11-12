; RUN: llc -O2 < %s > %t1.o
; RUN: llc -O2 < %s > %t2.o
; RUN: llc -O2 < %s > %t3.o
; RUN: llc -O2 < %s > %t4.o
; RUN: llc -O2 < %s > %t5.o
; RUN: cmp %t1.o %t2.o
; RUN: cmp %t1.o %t3.o
; RUN: cmp %t1.o %t4.o
; RUN: cmp %t1.o %t5.o

; Regression test for nondeterminism introduced in https://reviews.llvm.org/D57694

define void @test(i32 %x) !dbg !4 {
entry:
	call void @llvm.dbg.value(metadata ptr @f1, metadata !6, metadata !DIExpression()), !dbg !8
	call void @llvm.dbg.value(metadata ptr @f2, metadata !7, metadata !DIExpression()), !dbg !8
	%cmp = icmp eq i32 %x, 0, !dbg !8
	br i1 %cmp, label %cleanup, label %if.end

	if.end:
	%tobool = icmp eq i32 0, 0
	%a = select i1 %tobool, ptr addrspace(0) @f1, ptr null, !dbg !8
	%b = select i1 %tobool, ptr addrspace(0) @f2, ptr null, !dbg !8
	call addrspace(0) void %a(i32 %x)
	call addrspace(0) void %b(i32 %x)
	unreachable

	cleanup:
	ret void
}

declare void @f1(i32) addrspace(0)
declare void @f2(i32) addrspace(0)

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cc", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test", scope: !1, file: !1, type: !5, unit: !0)
!5 = !DISubroutineType(types: !2)
!6 = !DILocalVariable(name: "a", scope: !4, file: !1, type: !9)
!7 = !DILocalVariable(name: "b", scope: !4, file: !1, type: !9)
!8 = !DILocation(scope: !4)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
