; RUN: llc < %s -mtriple=x86_64-apple-macosx -enable-misched \
; RUN:          -verify-machineinstrs | FileCheck %s
;
; Test LiveInterval update handling of DBG_VALUE.
; rdar://12777252.
;
; CHECK: %entry
; CHECK: DEBUG_VALUE: hg
; CHECK: j

%struct.node.0.27 = type { i16, double, [3 x double], i32, i32 }
%struct.hgstruct.2.29 = type { %struct.bnode.1.28*, [3 x double], double, [3 x double] }
%struct.bnode.1.28 = type { i16, double, [3 x double], i32, i32, [3 x double], [3 x double], [3 x double], double, %struct.bnode.1.28*, %struct.bnode.1.28* }

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define signext i16 @subdivp(%struct.node.0.27* nocapture %p, double %dsq, double %tolsq, %struct.hgstruct.2.29* nocapture byval align 8 %hg) nounwind uwtable readonly ssp {
entry:
  call void @llvm.dbg.declare(metadata !{%struct.hgstruct.2.29* %hg}, metadata !4)
  %type = getelementptr inbounds %struct.node.0.27* %p, i64 0, i32 0
  %0 = load i16* %type, align 2
  %cmp = icmp eq i16 %0, 1
  br i1 %cmp, label %return, label %for.cond.preheader

for.cond.preheader:                               ; preds = %entry
  %arrayidx6.1 = getelementptr inbounds %struct.hgstruct.2.29* %hg, i64 0, i32 1, i64 1
  %cmp22 = fcmp olt double 0.000000e+00, %dsq
  %conv24 = zext i1 %cmp22 to i16
  br label %return

return:                                           ; preds = %for.cond.preheader, %entry
  %retval.0 = phi i16 [ %conv24, %for.cond.preheader ], [ 0, %entry ]
  ret i16 %retval.0
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12}

!0 = metadata !{i32 786449, metadata !11, i32 12, metadata !"clang version 3.3 (trunk 168918) (llvm/trunk 168920)", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !2, metadata !3, null, metadata !""} ; [ DW_TAG_compile_unit ] [MultiSource/Benchmarks/Olden/bh/newbh.c] [DW_LANG_C99]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{null}
!4 = metadata !{i32 786689, null, metadata !"hg", metadata !5, i32 67109589, metadata !6, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [hg] [line 725]
!5 = metadata !{i32 786473, metadata !11} ; [ DW_TAG_file_type ]
!6 = metadata !{i32 786454, metadata !11, null, metadata !"hgstruct", i32 492, i64 0, i64 0, i64 0, i32 0, metadata !7} ; [ DW_TAG_typedef ] [hgstruct] [line 492, size 0, align 0, offset 0] [from ]
!7 = metadata !{i32 786451, metadata !11, null, metadata !"", i32 487, i64 512, i64 64, i32 0, i32 0, null, null, i32 0, null, i32 0, null} ; [ DW_TAG_structure_type ] [line 487, size 512, align 64, offset 0] [def] [from ]
!11 = metadata !{metadata !"MultiSource/Benchmarks/Olden/bh/newbh.c", metadata !"MultiSource/Benchmarks/Olden/bh"}
!12 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
