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
  call void @llvm.dbg.declare(metadata !{float* %value.addr}, metadata !23), !dbg !24
  call void @llvm.dbg.declare(metadata !{%"union.PR15637::Value"* %tempValue}, metadata !25), !dbg !26
  ret void, !dbg !27
}

declare void @llvm.dbg.declare(metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!28}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.3 (trunk 178499) (llvm/trunk 178472)", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !9,  metadata !9, metadata !""} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/foo.cc] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"foo.cc", metadata !"/usr/local/google/home/echristo/tmp"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"g", metadata !"g", metadata !"_ZN7PR156371gEf", i32 3, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (float)* @_ZN7PR156371gEf, null, null, metadata !2, i32 3} ; [ DW_TAG_subprogram ] [line 3] [def] [g]
!5 = metadata !{i32 786489, metadata !1, null, metadata !"PR15637", i32 1} ; [ DW_TAG_namespace ] [PR15637] [line 1]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"float", i32 0, i64 32, i64 32, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ] [float] [line 0, size 32, align 32, offset 0, enc DW_ATE_float]
!9 = metadata !{metadata !10}
!10 = metadata !{i32 786484, i32 0, metadata !5, metadata !"f", metadata !"f", metadata !"_ZN7PR156371fE", metadata !11, i32 6, metadata !12, i32 0, i32 1, %"union.PR15637::Value"* @_ZN7PR156371fE, null} ; [ DW_TAG_variable ] [f] [line 6] [def]
!11 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/foo.cc]
!12 = metadata !{i32 786455, metadata !1, metadata !5, metadata !"Value<float>", i32 2, i64 32, i64 32, i64 0, i32 0, null, metadata !13, i32 0, null, metadata !21, null} ; [ DW_TAG_union_type ] [Value<float>] [line 2, size 32, align 32, offset 0] [def] [from ]
!13 = metadata !{metadata !14, metadata !16}
!14 = metadata !{i32 786445, metadata !1, metadata !12, metadata !"a", i32 2, i64 32, i64 32, i64 0, i32 0, metadata !15} ; [ DW_TAG_member ] [a] [line 2, size 32, align 32, offset 0] [from int]
!15 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!16 = metadata !{i32 786478, metadata !1, metadata !12, metadata !"Value", metadata !"Value", metadata !"", i32 2, metadata !17, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !20, i32 2} ; [ DW_TAG_subprogram ] [line 2] [Value]
!17 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !18, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = metadata !{null, metadata !19}
!19 = metadata !{i32 786447, i32 0, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !12} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from Value<float>]
!20 = metadata !{i32 786468}
!21 = metadata !{metadata !22}
!22 = metadata !{i32 786479, null, metadata !"T", metadata !8, null, i32 0, i32 0} ; [ DW_TAG_template_type_parameter ]
!23 = metadata !{i32 786689, metadata !4, metadata !"value", metadata !11, i32 16777219, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [value] [line 3]
!24 = metadata !{i32 3, i32 0, metadata !4, null}
!25 = metadata !{i32 786688, metadata !4, metadata !"tempValue", metadata !11, i32 4, metadata !12, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [tempValue] [line 4]
!26 = metadata !{i32 4, i32 0, metadata !4, null}
!27 = metadata !{i32 5, i32 0, metadata !4, null}
!28 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
