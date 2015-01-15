; RUN: llc -O0 -mtriple=x86_64-linux-gnu %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; Verify that we've emitted template arguments for the union
; CHECK: DW_TAG_union_type
; CHECK-NEXT: "Value<float>"
; CHECK: DW_TAG_template_type_parameter
; CHECK: "T"

%"union.PR15637::Value" = type { i32 }

@_ZN7PR156371fE = global %"union.PR15637::Value" zeroinitializer, align 4

define void @_ZN7PR156371gEf(float %value) #0 {
entry:
  %value.addr = alloca float, align 4
  %tempValue = alloca %"union.PR15637::Value", align 4
  store float %value, float* %value.addr, align 4
  call void @llvm.dbg.declare(metadata float* %value.addr, metadata !23, metadata !{!"0x102"}), !dbg !24
  call void @llvm.dbg.declare(metadata %"union.PR15637::Value"* %tempValue, metadata !25, metadata !{!"0x102"}), !dbg !26
  ret void, !dbg !27
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!28}

!0 = !{!"0x11\004\00clang version 3.3 (trunk 178499) (llvm/trunk 178472)\000\00\000\00\000", !1, !2, !2, !3, !9,  !2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/foo.cc] [DW_LANG_C_plus_plus]
!1 = !{!"foo.cc", !"/usr/local/google/home/echristo/tmp"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00g\00g\00_ZN7PR156371gEf\003\000\001\000\006\00256\000\003", !1, !5, !6, null, void (float)* @_ZN7PR156371gEf, null, null, !2} ; [ DW_TAG_subprogram ] [line 3] [def] [g]
!5 = !{!"0x39\00PR15637\001", !1, null} ; [ DW_TAG_namespace ] [PR15637] [line 1]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null, !8}
!8 = !{!"0x24\00float\000\0032\0032\000\000\004", null, null} ; [ DW_TAG_base_type ] [float] [line 0, size 32, align 32, offset 0, enc DW_ATE_float]
!9 = !{!10}
!10 = !{!"0x34\00f\00f\00_ZN7PR156371fE\006\000\001", !5, !11, !12, %"union.PR15637::Value"* @_ZN7PR156371fE, null} ; [ DW_TAG_variable ] [f] [line 6] [def]
!11 = !{!"0x29", !1}         ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/foo.cc]
!12 = !{!"0x17\00Value<float>\002\0032\0032\000\000\000", !1, !5, null, !13, null, !21, null} ; [ DW_TAG_union_type ] [Value<float>] [line 2, size 32, align 32, offset 0] [def] [from ]
!13 = !{!14, !16}
!14 = !{!"0xd\00a\002\0032\0032\000\000", !1, !12, !15} ; [ DW_TAG_member ] [a] [line 2, size 32, align 32, offset 0] [from int]
!15 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!16 = !{!"0x2e\00Value\00Value\00\002\000\000\000\006\00320\000\002", !1, !12, !17, null, null, null, i32 0, !20} ; [ DW_TAG_subprogram ] [line 2] [Value]
!17 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !18, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = !{null, !19}
!19 = !{!"0xf\00\000\0064\0064\000\001088", i32 0, null, !12} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from Value<float>]
!20 = !{i32 786468}
!21 = !{!22}
!22 = !{!"0x2f\00T\000\000", null, !8, null} ; [ DW_TAG_template_type_parameter ]
!23 = !{!"0x101\00value\0016777219\000", !4, !11, !8} ; [ DW_TAG_arg_variable ] [value] [line 3]
!24 = !MDLocation(line: 3, scope: !4)
!25 = !{!"0x100\00tempValue\004\000", !4, !11, !12} ; [ DW_TAG_auto_variable ] [tempValue] [line 4]
!26 = !MDLocation(line: 4, scope: !4)
!27 = !MDLocation(line: 5, scope: !4)
!28 = !{i32 1, !"Debug Info Version", i32 2}
