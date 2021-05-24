; This tests that emitting CodeView arrays doesn't assert when an ODR violation
; makes our array dimension size calculations inaccurate. (PR32383)

; Here was the scenario:
; $ cat a.cpp
; typedef union YYSTYPE { int x; } YYSTYPE;
; YYSTYPE a;
; $ cat b.cpp
; typedef union YYSTYPE { char x; } YYSTYPE;
; void fn1() { YYSTYPE a[1]; }
; $ clang-cl -c -Zi -flto a.cpp b.cpp
; $ llvm-link a.obj b.obj -S -o t.ll  # This is the test case IR.
; $ llc t.ll  # Used to assert

; RUN: llc < %s | FileCheck %s

; FIXME: sizeof(a) in the user program is 1, but we claim it is 4 because
; sometimes the frontend lies to us. See array-types-advanced.ll for an example.
;
; CHECK:	# Array (0x1004)
; CHECK:	.short	0xe                     # Record length
; CHECK:	.short	0x1503                  # Record kind: LF_ARRAY
; CHECK:	.long	0x1003                  # ElementType: YYSTYPE
; CHECK:	.long	0x23                    # IndexType: unsigned __int64
; CHECK:	.short	0x4                     # SizeOf
; CHECK:	.byte	0                       # Name
; CHECK:	.byte	241

; CHECK:	# Union (0x1006)
; CHECK:	.short	0x22                    # Record length
; CHECK:	.short	0x1506                  # Record kind: LF_UNION
; CHECK:	.short	0x1                     # MemberCount
; CHECK:	.short	0x600                   # Properties ( HasUniqueName (0x200) | Sealed (0x400) )
; CHECK:	.long	0x1005                  # FieldList: <field list>
; CHECK:	.short	0x4                     # SizeOf
; CHECK:	.asciz	"YYSTYPE"               # Name
; CHECK:	.asciz	".?ATYYSTYPE@@"         # LinkageName

; ModuleID = 'llvm-link'
source_filename = "llvm-link"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.10.24728"

%union.YYSTYPE = type { i32 }
%union.YYSTYPE.0 = type { i8 }

@"\01?a@@3TYYSTYPE@@A" = global %union.YYSTYPE zeroinitializer, align 4, !dbg !0

; Function Attrs: noinline nounwind sspstrong uwtable
define void @"\01?fn1@@YAXXZ"() #0 !dbg !21 {
entry:
  %a = alloca [1 x %union.YYSTYPE.0], align 1
  call void @llvm.dbg.declare(metadata [1 x %union.YYSTYPE.0]* %a, metadata !24, metadata !29), !dbg !30
  ret void, !dbg !30
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind sspstrong uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!2, !11}
!llvm.ident = !{!13, !13}
!llvm.module.flags = !{!18, !19, !20}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", linkageName: "\01?a@@3TYYSTYPE@@A", scope: !2, file: !3, line: 2, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 5.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "a.cpp", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "c0005139aa3df153c30d8c6953390a4b")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "YYSTYPE", file: !3, line: 1, baseType: !7)
!7 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "YYSTYPE", file: !3, line: 1, size: 32, elements: !8, identifier: ".?ATYYSTYPE@@")
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !7, file: !3, line: 1, baseType: !10, size: 32)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !12, producer: "clang version 5.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4)
!12 = !DIFile(filename: "b.cpp", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "9cfd390d8827beab36769147bb037abc")
!13 = !{!"clang version 5.0.0 "}
!18 = !{i32 2, !"CodeView", i32 1}
!19 = !{i32 2, !"Debug Info Version", i32 3}
!20 = !{i32 1, !"PIC Level", i32 2}
!21 = distinct !DISubprogram(name: "fn1", linkageName: "\01?fn1@@YAXXZ", scope: !12, file: !12, line: 2, type: !22, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !11, retainedNodes: !4)
!22 = !DISubroutineType(types: !23)
!23 = !{null}
!24 = !DILocalVariable(name: "a", scope: !21, file: !12, line: 2, type: !25)
!25 = !DICompositeType(tag: DW_TAG_array_type, baseType: !26, size: 8, elements: !27)
!26 = !DIDerivedType(tag: DW_TAG_typedef, name: "YYSTYPE", file: !12, line: 1, baseType: !7)
!27 = !{!28}
!28 = !DISubrange(count: 1)
!29 = !DIExpression()
!30 = !DILocation(line: 2, scope: !21)
