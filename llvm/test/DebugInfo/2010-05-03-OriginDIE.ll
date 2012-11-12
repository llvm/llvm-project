
;RUN: llc < %s -o /dev/null
;Radar 7937109

%struct.anon = type { i64, i32, i32, i32, [1 x i32] }
%struct.gpm_t = type { i32, i8*, [16 x i8], i32, i64, i64, i64, i64, i64, i64, i32, i16, i16, [8 x %struct.gpmr_t] }
%struct.gpmr_t = type { [48 x i8], [48 x i8], [16 x i8], i64, i64, i64, i64, i16 }
%struct.gpt_t = type { [8 x i8], i32, i32, i32, i32, i64, i64, i64, i64, [16 x i8], %struct.anon }

@llvm.used = appending global [1 x i8*] [i8* bitcast (void (%struct.gpm_t*, %struct.gpt_t*)* @gpt2gpm to i8*)], section "llvm.metadata" ; <[1 x i8*]*> [#uses=0]

define fastcc void @gpt2gpm(%struct.gpm_t* %gpm, %struct.gpt_t* %gpt) nounwind optsize ssp {
entry:
  %data_addr.i18 = alloca i64, align 8            ; <i64*> [#uses=1]
  %data_addr.i17 = alloca i64, align 8            ; <i64*> [#uses=2]
  %data_addr.i16 = alloca i64, align 8            ; <i64*> [#uses=0]
  %data_addr.i15 = alloca i32, align 4            ; <i32*> [#uses=0]
  %data_addr.i = alloca i64, align 8              ; <i64*> [#uses=0]
  %0 = getelementptr inbounds %struct.gpm_t* %gpm, i32 0, i32 2, i32 0 ; <i8*> [#uses=1]
  %1 = getelementptr inbounds %struct.gpt_t* %gpt, i32 0, i32 9, i32 0 ; <i8*> [#uses=1]
  call void @uuid_LtoB(i8* %0, i8* %1) nounwind, !dbg !0
  %a9 = load volatile i64* %data_addr.i18, align 8 ; <i64> [#uses=1]
  %a10 = call i64 @llvm.bswap.i64(i64 %a9) nounwind ; <i64> [#uses=1]
  %a11 = getelementptr inbounds %struct.gpt_t* %gpt, i32 0, i32 8, !dbg !7 ; <i64*> [#uses=1]
  %a12 = load i64* %a11, align 4, !dbg !7         ; <i64> [#uses=1]
  call void @llvm.dbg.declare(metadata !{i64* %data_addr.i17}, metadata !8) nounwind, !dbg !14
  store i64 %a12, i64* %data_addr.i17, align 8
  call void @llvm.dbg.value(metadata !6, i64 0, metadata !15) nounwind
  call void @llvm.dbg.value(metadata !18, i64 0, metadata !19) nounwind
  call void @llvm.dbg.declare(metadata !6, metadata !23) nounwind
  call void @llvm.dbg.value(metadata !{i64* %data_addr.i17}, i64 0, metadata !34) nounwind
  %a13 = load volatile i64* %data_addr.i17, align 8 ; <i64> [#uses=1]
  %a14 = call i64 @llvm.bswap.i64(i64 %a13) nounwind ; <i64> [#uses=2]
  %a15 = add i64 %a10, %a14, !dbg !7              ; <i64> [#uses=1]
  %a16 = sub i64 %a15, %a14                       ; <i64> [#uses=1]
  %a17 = getelementptr inbounds %struct.gpm_t* %gpm, i32 0, i32 5, !dbg !7 ; <i64*> [#uses=1]
  store i64 %a16, i64* %a17, align 4, !dbg !7
  ret void, !dbg !7
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

declare i32 @llvm.bswap.i32(i32) nounwind readnone

declare i64 @llvm.bswap.i64(i64) nounwind readnone

declare void @uuid_LtoB(i8*, i8*)

!0 = metadata !{i32 808, i32 0, metadata !1, null}
!1 = metadata !{i32 524299, metadata !2, i32 807, i32 0} ; [ DW_TAG_lexical_block ]
!2 = metadata !{i32 524334, i32 0, metadata !3, metadata !"gpt2gpm", metadata !"gpt2gpm", metadata !"gpt2gpm", metadata !3, i32 807, metadata !5, i1 true, i1 true, i32 0, i32 0, null, i1 false} ; [ DW_TAG_subprogram ]
!3 = metadata !{i32 524329, metadata !"G.c", metadata !"/tmp", metadata !4} ; [ DW_TAG_file_type ]
!4 = metadata !{i32 524305, i32 0, i32 1, metadata !"G.c", metadata !"/tmp", metadata !"llvm-gcc", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!5 = metadata !{i32 524309, metadata !3, metadata !"", metadata !3, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !6, i32 0, null} ; [ DW_TAG_subroutine_type ]
!6 = metadata !{null}
!7 = metadata !{i32 810, i32 0, metadata !1, null}
!8 = metadata !{i32 524545, metadata !9, metadata !"data", metadata !10, i32 201, metadata !11} ; [ DW_TAG_arg_variable ]
!9 = metadata !{i32 524334, i32 0, metadata !3, metadata !"_OSSwapInt64", metadata !"_OSSwapInt64", metadata !"_OSSwapInt64", metadata !10, i32 202, metadata !5, i1 true, i1 true, i32 0, i32 0, null, i1 false} ; [ DW_TAG_subprogram ]
!10 = metadata !{i32 524329, metadata !"OSByteOrder.h", metadata !"/usr/include/libkern/ppc", metadata !4} ; [ DW_TAG_file_type ]
!11 = metadata !{i32 524310, metadata !3, metadata !"uint64_t", metadata !12, i32 59, i64 0, i64 0, i64 0, i32 0, metadata !13} ; [ DW_TAG_typedef ]
!12 = metadata !{i32 524329, metadata !"stdint.h", metadata !"/usr/4.2.1/include", metadata !4} ; [ DW_TAG_file_type ]
!13 = metadata !{i32 524324, metadata !3, metadata !"long long unsigned int", metadata !3, i32 0, i64 64, i64 64, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!14 = metadata !{i32 202, i32 0, metadata !9, metadata !7}
!15 = metadata !{i32 524545, metadata !16, metadata !"base", metadata !10, i32 92, metadata !17} ; [ DW_TAG_arg_variable ]
!16 = metadata !{i32 524334, i32 0, metadata !3, metadata !"OSReadSwapInt64", metadata !"OSReadSwapInt64", metadata !"OSReadSwapInt64", metadata !10, i32 95, metadata !5, i1 true, i1 true, i32 0, i32 0, null, i1 false} ; [ DW_TAG_subprogram ]
!17 = metadata !{i32 524303, metadata !3, metadata !"", metadata !3, i32 0, i64 32, i64 32, i64 0, i32 0, null} ; [ DW_TAG_pointer_type ]
!18 = metadata !{i32 0}
!19 = metadata !{i32 524545, metadata !16, metadata !"byteOffset", metadata !10, i32 94, metadata !20} ; [ DW_TAG_arg_variable ]
!20 = metadata !{i32 524310, metadata !3, metadata !"uintptr_t", metadata !21, i32 114, i64 0, i64 0, i64 0, i32 0, metadata !22} ; [ DW_TAG_typedef ]
!21 = metadata !{i32 524329, metadata !"types.h", metadata !"/usr/include/ppc", metadata !4} ; [ DW_TAG_file_type ]
!22 = metadata !{i32 524324, metadata !3, metadata !"long unsigned int", metadata !3, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!23 = metadata !{i32 524544, metadata !24, metadata !"u", metadata !10, i32 100, metadata !25} ; [ DW_TAG_auto_variable ]
!24 = metadata !{i32 524299, metadata !16, i32 95, i32 0} ; [ DW_TAG_lexical_block ]
!25 = metadata !{i32 524311, metadata !16, metadata !"", metadata !10, i32 97, i64 64, i64 64, i64 0, i32 0, null, metadata !26, i32 0, null} ; [ DW_TAG_union_type ]
!26 = metadata !{metadata !27, metadata !28}
!27 = metadata !{i32 524301, metadata !25, metadata !"u64", metadata !10, i32 98, i64 64, i64 64, i64 0, i32 0, metadata !11} ; [ DW_TAG_member ]
!28 = metadata !{i32 524301, metadata !25, metadata !"u32", metadata !10, i32 99, i64 64, i64 32, i64 0, i32 0, metadata !29} ; [ DW_TAG_member ]
!29 = metadata !{i32 524289, metadata !3, metadata !"", metadata !3, i32 0, i64 64, i64 32, i64 0, i32 0, metadata !30, metadata !32, i32 0, null} ; [ DW_TAG_array_type ]
!30 = metadata !{i32 524310, metadata !3, metadata !"uint32_t", metadata !12, i32 55, i64 0, i64 0, i64 0, i32 0, metadata !31} ; [ DW_TAG_typedef ]
!31 = metadata !{i32 524324, metadata !3, metadata !"unsigned int", metadata !3, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!32 = metadata !{metadata !33}
!33 = metadata !{i32 524321, i64 0, i64 1}        ; [ DW_TAG_subrange_type ]
!34 = metadata !{i32 524544, metadata !24, metadata !"addr", metadata !10, i32 96, metadata !35} ; [ DW_TAG_auto_variable ]
!35 = metadata !{i32 524303, metadata !3, metadata !"", metadata !3, i32 0, i64 32, i64 32, i64 0, i32 0, metadata !11} ; [ DW_TAG_pointer_type ]
