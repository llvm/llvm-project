; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; Checks that we emit debug info for the block variable declare.
; CHECK: DW_TAG_subprogram [3]
; CHECK: DW_TAG_variable [5]
; CHECK: DW_AT_name [DW_FORM_strp]     ( .debug_str[{{.*}}] = "block")
; CHECK: DW_AT_location [DW_FORM_sec_offset]        ({{.*}})

%struct.__block_descriptor = type { i64, i64 }
%struct.__block_literal_generic = type { i8*, i32, i32, i8*, %struct.__block_descriptor* }

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define hidden void @__foo_block_invoke_0(i8* %.block_descriptor) uwtable ssp {
entry:
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  call void @llvm.dbg.value(metadata !{i8* %.block_descriptor}, i64 0, metadata !39), !dbg !51
  %block = bitcast i8* %.block_descriptor to <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, void ()* }>*, !dbg !52
  call void @llvm.dbg.declare(metadata !{<{ i8*, i32, i32, i8*, %struct.__block_descriptor*, void ()* }>* %block}, metadata !53), !dbg !54
  %block.capture.addr = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, void ()* }>* %block, i32 0, i32 5, !dbg !55
  %0 = load void ()** %block.capture.addr, align 8, !dbg !55
  %block.literal = bitcast void ()* %0 to %struct.__block_literal_generic*, !dbg !55
  %1 = getelementptr inbounds %struct.__block_literal_generic* %block.literal, i32 0, i32 3, !dbg !55
  %2 = bitcast %struct.__block_literal_generic* %block.literal to i8*, !dbg !55
  %3 = load i8** %1, !dbg !55
  %4 = bitcast i8* %3 to void (i8*)*, !dbg !55
  invoke void %4(i8* %2)
          to label %invoke.cont unwind label %lpad, !dbg !55

invoke.cont:                                      ; preds = %entry
  br label %eh.cont, !dbg !58

eh.cont:                                          ; preds = %catch, %invoke.cont
  ret void, !dbg !61

lpad:                                             ; preds = %entry
  %5 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__objc_personality_v0 to i8*)
          catch i8* null, !dbg !55
  %6 = extractvalue { i8*, i32 } %5, 0, !dbg !55
  store i8* %6, i8** %exn.slot, !dbg !55
  %7 = extractvalue { i8*, i32 } %5, 1, !dbg !55
  store i32 %7, i32* %ehselector.slot, !dbg !55
  br label %catch, !dbg !55

catch:                                            ; preds = %lpad
  %exn = load i8** %exn.slot, !dbg !62
  %exn.adjusted = call i8* @objc_begin_catch(i8* %exn) nounwind, !dbg !62
  call void @objc_end_catch(), !dbg !58
  br label %eh.cont, !dbg !58
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

declare i8* @objc_begin_catch(i8*)

declare void @objc_end_catch()

declare i32 @__objc_personality_v0(...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!35, !36, !37, !38, !64}

!0 = metadata !{i32 786449, metadata !63, i32 16, metadata !"clang version 3.1 (trunk 151227)", i1 false, metadata !"", i32 2, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5, metadata !28, metadata !31, metadata !34}
!5 = metadata !{i32 786478, metadata !6, metadata !6, metadata !"foo", metadata !"foo", metadata !"", i32 5, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !26, i32 5} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 786473, metadata !63} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null, metadata !9}
!9 = metadata !{i32 786454, metadata !63, null, metadata !"dispatch_block_t", i32 1, i64 0, i64 0, i64 0, i32 0, metadata !10} ; [ DW_TAG_typedef ]
!10 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 0, i64 0, i32 0, metadata !11} ; [ DW_TAG_pointer_type ]
!11 = metadata !{i32 786451, metadata !63, metadata !6, metadata !"__block_literal_generic", i32 5, i64 256, i64 0, i32 0, i32 8, null, metadata !12, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [__block_literal_generic] [line 5, size 256, align 0, offset 0] [def] [from ]
!12 = metadata !{metadata !13, metadata !15, metadata !17, metadata !18, metadata !19}
!13 = metadata !{i32 786445, metadata !63, metadata !6, metadata !"__isa", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !14} ; [ DW_TAG_member ]
!14 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, null} ; [ DW_TAG_pointer_type ]
!15 = metadata !{i32 786445, metadata !63, metadata !6, metadata !"__flags", i32 0, i64 32, i64 32, i64 64, i32 0, metadata !16} ; [ DW_TAG_member ]
!16 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!17 = metadata !{i32 786445, metadata !63, metadata !6, metadata !"__reserved", i32 0, i64 32, i64 32, i64 96, i32 0, metadata !16} ; [ DW_TAG_member ]
!18 = metadata !{i32 786445, metadata !63, metadata !6, metadata !"__FuncPtr", i32 0, i64 64, i64 64, i64 128, i32 0, metadata !14} ; [ DW_TAG_member ]
!19 = metadata !{i32 786445, metadata !63, metadata !6, metadata !"__descriptor", i32 5, i64 64, i64 64, i64 192, i32 0, metadata !20} ; [ DW_TAG_member ]
!20 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 0, i64 0, i32 0, metadata !21} ; [ DW_TAG_pointer_type ]
!21 = metadata !{i32 786451, metadata !63, metadata !6, metadata !"__block_descriptor", i32 5, i64 128, i64 0, i32 0, i32 8, null, metadata !22, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [__block_descriptor] [line 5, size 128, align 0, offset 0] [def] [from ]
!22 = metadata !{metadata !23, metadata !25}
!23 = metadata !{i32 786445, metadata !63, metadata !6, metadata !"reserved", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !24} ; [ DW_TAG_member ]
!24 = metadata !{i32 786468, null, null, metadata !"long unsigned int", i32 0, i64 64, i64 64, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!25 = metadata !{i32 786445, metadata !63, metadata !6, metadata !"Size", i32 0, i64 64, i64 64, i64 64, i32 0, metadata !24} ; [ DW_TAG_member ]
!26 = metadata !{metadata !27}
!27 = metadata !{i32 786468}                      ; [ DW_TAG_base_type ]
!28 = metadata !{i32 786478, metadata !6, metadata !6, metadata !"__foo_block_invoke_0", metadata !"__foo_block_invoke_0", metadata !"", i32 7, metadata !29, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i8*)* @__foo_block_invoke_0, null, null, metadata !26, i32 7} ; [ DW_TAG_subprogram ]
!29 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !30, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!30 = metadata !{null, metadata !14}
!31 = metadata !{i32 786478, metadata !6, metadata !6, metadata !"__copy_helper_block_", metadata !"__copy_helper_block_", metadata !"", i32 10, metadata !32, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !26, i32 10} ; [ DW_TAG_subprogram ]
!32 = metadata !{i32 786453, i32 0, null, i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !33, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!33 = metadata !{null, metadata !14, metadata !14}
!34 = metadata !{i32 786478, metadata !6, metadata !6, metadata !"__destroy_helper_block_", metadata !"__destroy_helper_block_", metadata !"", i32 10, metadata !29, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !26, i32 10} ; [ DW_TAG_subprogram ]
!35 = metadata !{i32 1, metadata !"Objective-C Version", i32 2}
!36 = metadata !{i32 1, metadata !"Objective-C Image Info Version", i32 0}
!37 = metadata !{i32 1, metadata !"Objective-C Image Info Section", metadata !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!38 = metadata !{i32 4, metadata !"Objective-C Garbage Collection", i32 0}
!39 = metadata !{i32 786689, metadata !28, metadata !".block_descriptor", metadata !6, i32 16777223, metadata !40, i32 64, i32 0} ; [ DW_TAG_arg_variable ]
!40 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 0, i64 0, i32 0, metadata !41} ; [ DW_TAG_pointer_type ]
!41 = metadata !{i32 786451, metadata !63, metadata !6, metadata !"__block_literal_1", i32 7, i64 320, i64 64, i32 0, i32 0, null, metadata !42, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [__block_literal_1] [line 7, size 320, align 64, offset 0] [def] [from ]
!42 = metadata !{metadata !43, metadata !44, metadata !45, metadata !46, metadata !47, metadata !50}
!43 = metadata !{i32 786445, metadata !63, metadata !6, metadata !"__isa", i32 7, i64 64, i64 64, i64 0, i32 0, metadata !14} ; [ DW_TAG_member ]
!44 = metadata !{i32 786445, metadata !63, metadata !6, metadata !"__flags", i32 7, i64 32, i64 32, i64 64, i32 0, metadata !16} ; [ DW_TAG_member ]
!45 = metadata !{i32 786445, metadata !63, metadata !6, metadata !"__reserved", i32 7, i64 32, i64 32, i64 96, i32 0, metadata !16} ; [ DW_TAG_member ]
!46 = metadata !{i32 786445, metadata !63, metadata !6, metadata !"__FuncPtr", i32 7, i64 64, i64 64, i64 128, i32 0, metadata !14} ; [ DW_TAG_member ]
!47 = metadata !{i32 786445, metadata !63, metadata !6, metadata !"__descriptor", i32 7, i64 64, i64 64, i64 192, i32 0, metadata !48} ; [ DW_TAG_member ]
!48 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !49} ; [ DW_TAG_pointer_type ]
!49 = metadata !{i32 786451, metadata !63, null, metadata !"__block_descriptor_withcopydispose", i32 7, i32 0, i32 0, i32 0, i32 4, null, null, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [__block_descriptor_withcopydispose] [line 7, size 0, align 0, offset 0] [decl] [from ]
!50 = metadata !{i32 786445, metadata !63, metadata !6, metadata !"block", i32 7, i64 64, i64 64, i64 256, i32 0, metadata !9} ; [ DW_TAG_member ]
!51 = metadata !{i32 7, i32 18, metadata !28, null}
!52 = metadata !{i32 7, i32 19, metadata !28, null}
!53 = metadata !{i32 786688, metadata !28, metadata !"block", metadata !6, i32 5, metadata !9, i32 0, i32 0, i64 1, i64 32} ; [ DW_TAG_auto_variable ]
!54 = metadata !{i32 5, i32 27, metadata !28, null}
!55 = metadata !{i32 8, i32 22, metadata !56, null}
!56 = metadata !{i32 786443, metadata !57, i32 7, i32 26, metadata !6, i32 2} ; [ DW_TAG_lexical_block ]
!57 = metadata !{i32 786443, metadata !28, i32 7, i32 19, metadata !6, i32 1} ; [ DW_TAG_lexical_block ]
!58 = metadata !{i32 10, i32 20, metadata !59, null}
!59 = metadata !{i32 786443, metadata !60, i32 9, i32 35, metadata !6, i32 4} ; [ DW_TAG_lexical_block ]
!60 = metadata !{i32 786443, metadata !57, i32 9, i32 35, metadata !6, i32 3} ; [ DW_TAG_lexical_block ]
!61 = metadata !{i32 10, i32 21, metadata !28, null}
!62 = metadata !{i32 9, i32 20, metadata !56, null}
!63 = metadata !{metadata !"foo.m", metadata !"/Users/echristo"}
!64 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
