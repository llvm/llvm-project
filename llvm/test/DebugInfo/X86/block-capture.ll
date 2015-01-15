; RUN: llc %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; Checks that we emit debug info for the block variable declare.
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_variable
;                                              fbreg +8, deref, +32
; CHECK-NEXT: DW_AT_location [DW_FORM_block1] (<0x05> 91 08 06 23 20 )
; CHECK-NEXT: DW_AT_name {{.*}} "block"

; Extracted from the clang output for:
; void foo() {
;  void (^block)() = ^{ block(); };
; }

; ModuleID = 'foo.m'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

%struct.__block_descriptor = type { i64, i64 }
%struct.__block_literal_generic = type { i8*, i32, i32, i8*, %struct.__block_descriptor* }

@_NSConcreteStackBlock = external global i8*
@.str = private unnamed_addr constant [6 x i8] c"v8@?0\00", align 1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: ssp uwtable
define internal void @__foo_block_invoke(i8* %.block_descriptor) #2 {
entry:
  %.block_descriptor.addr = alloca i8*, align 8
  %block.addr = alloca <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, void (...)* }>*, align 8
  store i8* %.block_descriptor, i8** %.block_descriptor.addr, align 8
  %0 = load i8** %.block_descriptor.addr
  call void @llvm.dbg.value(metadata i8* %0, i64 0, metadata !47, metadata !43), !dbg !66
  call void @llvm.dbg.declare(metadata i8* %.block_descriptor, metadata !47, metadata !43), !dbg !66
  %block = bitcast i8* %.block_descriptor to <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, void (...)* }>*, !dbg !67
  store <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, void (...)* }>* %block, <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, void (...)* }>** %block.addr, align 8
  call void @llvm.dbg.declare(metadata <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, void (...)* }>** %block.addr, metadata !68, metadata !69), !dbg !70
  %block.capture.addr = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, void (...)* }>* %block, i32 0, i32 5, !dbg !71
  %1 = load void (...)** %block.capture.addr, align 8, !dbg !71
  %block.literal = bitcast void (...)* %1 to %struct.__block_literal_generic*, !dbg !71
  %2 = getelementptr inbounds %struct.__block_literal_generic* %block.literal, i32 0, i32 3, !dbg !71
  %3 = bitcast %struct.__block_literal_generic* %block.literal to i8*, !dbg !71
  %4 = load i8** %2, !dbg !71
  %5 = bitcast i8* %4 to void (i8*, ...)*, !dbg !71
  call void (i8*, ...)* %5(i8* %3), !dbg !71
  ret void, !dbg !73
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1


attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { ssp uwtable }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!16, !17, !18, !19, !20, !21, !22}
!llvm.ident = !{!23}

!0 = !{!"0x11\0016\00clang version 3.6.0 (trunk 223471)\000\00\002\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/foo.m] [DW_LANG_ObjC]
!1 = !{!"foo.m", !""}
!2 = !{}
!3 = !{!8}
!5 = !{!"0x29", !1}    ; [ DW_TAG_file_type ] [/foo.m]
!6 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null}
!8 = !{!"0x2e\00__foo_block_invoke\00__foo_block_invoke\00\002\001\001\000\000\00256\000\002", !1, !5, !9, null, void (i8*)* @__foo_block_invoke, null, null, !2} ; [ DW_TAG_subprogram ] [line 2] [local] [def] [__foo_block_invoke]
!9 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !10, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = !{null, !11}
!11 = !{!"0xf\00\000\0064\0064\000\000", null, null, null} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!13 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !14, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = !{null, !11, !11}
!16 = !{i32 1, !"Objective-C Version", i32 2}
!17 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!18 = !{i32 1, !"Objective-C Image Info Section", !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!19 = !{i32 4, !"Objective-C Garbage Collection", i32 0}
!20 = !{i32 2, !"Dwarf Version", i32 2}
!21 = !{i32 2, !"Debug Info Version", i32 2}
!22 = !{i32 1, !"PIC Level", i32 2}
!23 = !{!"clang version 3.6.0 (trunk 223471)"}
!25 = !{!"0xf\00\000\0064\000\000\000", null, null, !26} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from __block_literal_generic]
!26 = !{!"0x13\00__block_literal_generic\002\00256\000\000\008\000", !1, !5, null, !27, null, null, null} ; [ DW_TAG_structure_type ] [__block_literal_generic] [line 2, size 256, align 0, offset 0] [def] [from ]
!27 = !{!28, !29, !31, !32, !36}
!28 = !{!"0xd\00__isa\000\0064\0064\000\000", !1, !5, !11} ; [ DW_TAG_member ] [__isa] [line 0, size 64, align 64, offset 0] [from ]
!29 = !{!"0xd\00__flags\000\0032\0032\0064\000", !1, !5, !30} ; [ DW_TAG_member ] [__flags] [line 0, size 32, align 32, offset 64] [from int]
!30 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!31 = !{!"0xd\00__reserved\000\0032\0032\0096\000", !1, !5, !30} ; [ DW_TAG_member ] [__reserved] [line 0, size 32, align 32, offset 96] [from int]
!32 = !{!"0xd\00__FuncPtr\000\0064\0064\00128\000", !1, !5, !33} ; [ DW_TAG_member ] [__FuncPtr] [line 0, size 64, align 64, offset 128] [from ]
!33 = !{!"0xf\00\000\0064\0064\000\000", null, null, !34} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!34 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !35, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!35 = !{null, null}
!36 = !{!"0xd\00__descriptor\002\0064\0064\00192\000", !1, !5, !37} ; [ DW_TAG_member ] [__descriptor] [line 2, size 64, align 64, offset 192] [from ]
!37 = !{!"0xf\00\000\0064\000\000\000", null, null, !38} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from __block_descriptor]
!38 = !{!"0x13\00__block_descriptor\002\00128\000\000\008\000", !1, !5, null, !39, null, null, null} ; [ DW_TAG_structure_type ] [__block_descriptor] [line 2, size 128, align 0, offset 0] [def] [from ]
!39 = !{!40, !42}
!40 = !{!"0xd\00reserved\000\0064\0064\000\000", !1, !5, !41} ; [ DW_TAG_member ] [reserved] [line 0, size 64, align 64, offset 0] [from long unsigned int]
!41 = !{!"0x24\00long unsigned int\000\0064\0064\000\000\007", null, null} ; [ DW_TAG_base_type ] [long unsigned int] [line 0, size 64, align 64, offset 0, enc DW_ATE_unsigned]
!42 = !{!"0xd\00Size\000\0064\0064\0064\000", !1, !5, !41} ; [ DW_TAG_member ] [Size] [line 0, size 64, align 64, offset 64] [from long unsigned int]
!43 = !{!"0x102"}               ; [ DW_TAG_expression ]
!47 = !{!"0x101\00.block_descriptor\0016777218\0064", !8, !5, !48} ; [ DW_TAG_arg_variable ] [.block_descriptor] [line 2]
!48 = !{!"0xf\00\000\0064\000\000\000", null, null, !49} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from __block_literal_1]
!49 = !{!"0x13\00__block_literal_1\002\00320\0064\000\000\000", !1, !5, null, !50, null, null, null} ; [ DW_TAG_structure_type ] [__block_literal_1] [line 2, size 320, align 64, offset 0] [def] [from ]
!50 = !{!51, !52, !53, !54, !56, !65}
!51 = !{!"0xd\00__isa\002\0064\0064\000\003", !1, !5, !11} ; [ DW_TAG_member ] [__isa] [line 2, size 64, align 64, offset 0] [public] [from ]
!52 = !{!"0xd\00__flags\002\0032\0032\0064\003", !1, !5, !30} ; [ DW_TAG_member ] [__flags] [line 2, size 32, align 32, offset 64] [public] [from int]
!53 = !{!"0xd\00__reserved\002\0032\0032\0096\003", !1, !5, !30} ; [ DW_TAG_member ] [__reserved] [line 2, size 32, align 32, offset 96] [public] [from int]
!54 = !{!"0xd\00__FuncPtr\002\0064\0064\00128\003", !1, !5, !55} ; [ DW_TAG_member ] [__FuncPtr] [line 2, size 64, align 64, offset 128] [public] [from ]
!55 = !{!"0xf\00\000\0064\0064\000\000", null, null, !6} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!56 = !{!"0xd\00__descriptor\002\0064\0064\00192\003", !1, !5, !57} ; [ DW_TAG_member ] [__descriptor] [line 2, size 64, align 64, offset 192] [public] [from ]
!57 = !{!"0xf\00\000\0064\0064\000\000", null, null, !58} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from __block_descriptor_withcopydispose]
!58 = !{!"0x13\00__block_descriptor_withcopydispose\002\00256\0064\000\000\000", !1, null, null, !59, null, null, null} ; [ DW_TAG_structure_type ] [__block_descriptor_withcopydispose] [line 2, size 256, align 64, offset 0] [def] [from ]
!59 = !{!60, !61, !62, !64}
!60 = !{!"0xd\00reserved\002\0064\0064\000\000", !1, !58, !41} ; [ DW_TAG_member ] [reserved] [line 2, size 64, align 64, offset 0] [from long unsigned int]
!61 = !{!"0xd\00Size\002\0064\0064\0064\000", !1, !58, !41} ; [ DW_TAG_member ] [Size] [line 2, size 64, align 64, offset 64] [from long unsigned int]
!62 = !{!"0xd\00CopyFuncPtr\002\0064\0064\00128\000", !1, !58, !63} ; [ DW_TAG_member ] [CopyFuncPtr] [line 2, size 64, align 64, offset 128] [from ]
!63 = !{!"0xf\00\000\0064\0064\000\000", null, null, !11} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!64 = !{!"0xd\00DestroyFuncPtr\002\0064\0064\00192\000", !1, !58, !63} ; [ DW_TAG_member ] [DestroyFuncPtr] [line 2, size 64, align 64, offset 192] [from ]
!65 = !{!"0xd\00block\002\0064\0064\00256\003", !1, !5, !25} ; [ DW_TAG_member ] [block] [line 2, size 64, align 64, offset 256] [public] [from ]
!66 = !MDLocation(line: 2, column: 20, scope: !8)
!67 = !MDLocation(line: 2, column: 21, scope: !8)
!68 = !{!"0x100\00block\002\000", !8, !5, !25} ; [ DW_TAG_auto_variable ] [block] [line 2]
!69 = !{!"0x102\006\0034\0032"} ; [ DW_TAG_expression ] [DW_OP_deref]
!70 = !MDLocation(line: 2, column: 9, scope: !8)
!71 = !MDLocation(line: 2, column: 23, scope: !72)
!72 = !{!"0xb\002\0021\000", !1, !8} ; [ DW_TAG_lexical_block ] [/foo.m]
!73 = !MDLocation(line: 2, column: 32, scope: !8)
