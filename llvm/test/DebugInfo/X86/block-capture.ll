; RUN: llc %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck %s

; Checks that we emit debug info for the block variable declare.
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_location [DW_FORM_block1] (DW_OP_fbreg +8, DW_OP_deref, DW_OP_plus_uconst 0x20)
; CHECK-NEXT: DW_AT_name {{.*}} "block"

; Extracted from the clang output for:
; void foo() {
;  void (^block)() = ^{ block(); };
; }

; ModuleID = 'foo.m'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

%struct.__block_descriptor = type { i64, i64 }
%struct.__block_literal_generic = type { ptr, i32, i32, ptr, ptr }

@_NSConcreteStackBlock = external global ptr
@.str = private unnamed_addr constant [6 x i8] c"v8@?0\00", align 1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: ssp uwtable
define internal void @__foo_block_invoke(ptr %.block_descriptor) #2 !dbg !8 {
entry:
  %.block_descriptor.addr = alloca ptr, align 8
  %block.addr = alloca ptr, align 8
  store ptr %.block_descriptor, ptr %.block_descriptor.addr, align 8
  %0 = load ptr, ptr %.block_descriptor.addr
  call void @llvm.dbg.value(metadata ptr %0, metadata !47, metadata !43), !dbg !66
  call void @llvm.dbg.declare(metadata ptr %.block_descriptor, metadata !47, metadata !43), !dbg !66
  store ptr %.block_descriptor, ptr %block.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %block.addr, metadata !68, metadata !69), !dbg !70
  %block.capture.addr = getelementptr inbounds <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %.block_descriptor, i32 0, i32 5, !dbg !71
  %1 = load ptr, ptr %block.capture.addr, align 8, !dbg !71
  %2 = getelementptr inbounds %struct.__block_literal_generic, ptr %1, i32 0, i32 3, !dbg !71
  %3 = load ptr, ptr %2, !dbg !71
  call void (ptr, ...) %3(ptr %1), !dbg !71
  ret void, !dbg !73
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1


attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { ssp uwtable }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!16, !17, !18, !19, !20, !21, !22}
!llvm.ident = !{!23}

!0 = distinct !DICompileUnit(language: DW_LANG_ObjC, producer: "clang version 3.6.0 (trunk 223471)", isOptimized: false, runtimeVersion: 2, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "foo.m", directory: "")
!2 = !{}
!5 = !DIFile(filename: "foo.m", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = distinct !DISubprogram(name: "__foo_block_invoke", line: 2, isLocal: true, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 2, file: !1, scope: !5, type: !9, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: null)
!13 = !DISubroutineType(types: !14)
!14 = !{null, !11, !11}
!16 = !{i32 1, !"Objective-C Version", i32 2}
!17 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!18 = !{i32 1, !"Objective-C Image Info Section", !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!19 = !{i32 4, !"Objective-C Garbage Collection", i32 0}
!20 = !{i32 2, !"Dwarf Version", i32 2}
!21 = !{i32 2, !"Debug Info Version", i32 3}
!22 = !{i32 1, !"PIC Level", i32 2}
!23 = !{!"clang version 3.6.0 (trunk 223471)"}
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, baseType: !26)
!26 = !DICompositeType(tag: DW_TAG_structure_type, name: "__block_literal_generic", line: 2, size: 256, flags: DIFlagAppleBlock, file: !1, scope: !5, elements: !27)
!27 = !{!28, !29, !31, !32, !36}
!28 = !DIDerivedType(tag: DW_TAG_member, name: "__isa", size: 64, align: 64, file: !1, scope: !5, baseType: !11)
!29 = !DIDerivedType(tag: DW_TAG_member, name: "__flags", size: 32, align: 32, offset: 64, file: !1, scope: !5, baseType: !30)
!30 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!31 = !DIDerivedType(tag: DW_TAG_member, name: "__reserved", size: 32, align: 32, offset: 96, file: !1, scope: !5, baseType: !30)
!32 = !DIDerivedType(tag: DW_TAG_member, name: "__FuncPtr", size: 64, align: 64, offset: 128, file: !1, scope: !5, baseType: !33)
!33 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !34)
!34 = !DISubroutineType(types: !35)
!35 = !{null, null}
!36 = !DIDerivedType(tag: DW_TAG_member, name: "__descriptor", line: 2, size: 64, align: 64, offset: 192, file: !1, scope: !5, baseType: !37)
!37 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, baseType: !38)
!38 = !DICompositeType(tag: DW_TAG_structure_type, name: "__block_descriptor", line: 2, size: 128, flags: DIFlagAppleBlock, file: !1, scope: !5, elements: !39)
!39 = !{!40, !42}
!40 = !DIDerivedType(tag: DW_TAG_member, name: "reserved", size: 64, align: 64, file: !1, scope: !5, baseType: !41)
!41 = !DIBasicType(tag: DW_TAG_base_type, name: "long unsigned int", size: 64, align: 64, encoding: DW_ATE_unsigned)
!42 = !DIDerivedType(tag: DW_TAG_member, name: "Size", size: 64, align: 64, offset: 64, file: !1, scope: !5, baseType: !41)
!43 = !DIExpression()
!47 = !DILocalVariable(name: ".block_descriptor", line: 2, arg: 1, flags: DIFlagArtificial, scope: !8, file: !5, type: !48)
!48 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, baseType: !49)
!49 = !DICompositeType(tag: DW_TAG_structure_type, name: "__block_literal_1", line: 2, size: 320, align: 64, file: !1, scope: !5, elements: !50)
!50 = !{!51, !52, !53, !54, !56, !65}
!51 = !DIDerivedType(tag: DW_TAG_member, name: "__isa", line: 2, size: 64, align: 64, flags: DIFlagPublic, file: !1, scope: !5, baseType: !11)
!52 = !DIDerivedType(tag: DW_TAG_member, name: "__flags", line: 2, size: 32, align: 32, offset: 64, flags: DIFlagPublic, file: !1, scope: !5, baseType: !30)
!53 = !DIDerivedType(tag: DW_TAG_member, name: "__reserved", line: 2, size: 32, align: 32, offset: 96, flags: DIFlagPublic, file: !1, scope: !5, baseType: !30)
!54 = !DIDerivedType(tag: DW_TAG_member, name: "__FuncPtr", line: 2, size: 64, align: 64, offset: 128, flags: DIFlagPublic, file: !1, scope: !5, baseType: !55)
!55 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !6)
!56 = !DIDerivedType(tag: DW_TAG_member, name: "__descriptor", line: 2, size: 64, align: 64, offset: 192, flags: DIFlagPublic, file: !1, scope: !5, baseType: !57)
!57 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !58)
!58 = !DICompositeType(tag: DW_TAG_structure_type, name: "__block_descriptor_withcopydispose", line: 2, size: 256, align: 64, file: !1, elements: !59)
!59 = !{!60, !61, !62, !64}
!60 = !DIDerivedType(tag: DW_TAG_member, name: "reserved", line: 2, size: 64, align: 64, file: !1, scope: !58, baseType: !41)
!61 = !DIDerivedType(tag: DW_TAG_member, name: "Size", line: 2, size: 64, align: 64, offset: 64, file: !1, scope: !58, baseType: !41)
!62 = !DIDerivedType(tag: DW_TAG_member, name: "CopyFuncPtr", line: 2, size: 64, align: 64, offset: 128, file: !1, scope: !58, baseType: !63)
!63 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !11)
!64 = !DIDerivedType(tag: DW_TAG_member, name: "DestroyFuncPtr", line: 2, size: 64, align: 64, offset: 192, file: !1, scope: !58, baseType: !63)
!65 = !DIDerivedType(tag: DW_TAG_member, name: "block", line: 2, size: 64, align: 64, offset: 256, flags: DIFlagPublic, file: !1, scope: !5, baseType: !25)
!66 = !DILocation(line: 2, column: 20, scope: !8)
!67 = !DILocation(line: 2, column: 21, scope: !8)
!68 = !DILocalVariable(name: "block", line: 2, scope: !8, file: !5, type: !25)
!69 = !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 32)
!70 = !DILocation(line: 2, column: 9, scope: !8)
!71 = !DILocation(line: 2, column: 23, scope: !72)
!72 = distinct !DILexicalBlock(line: 2, column: 21, file: !1, scope: !8)
!73 = !DILocation(line: 2, column: 32, scope: !8)
