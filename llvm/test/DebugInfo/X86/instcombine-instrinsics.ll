; RUN: opt < %s -O2 -S | FileCheck %s
; Verify that we emit the same intrinsic at most once.
; CHECK: call void @llvm.dbg.value(metadata !{%struct.i14** %i14}
; CHECK-NOT: call void @llvm.dbg.value(metadata !{%struct.i14** %i14}
; CHECK: ret

;*** IR Dump After Dead Argument Elimination ***
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%struct.i3 = type { i32 }
%struct.i14 = type { i32 }
%struct.i24 = type opaque

define %struct.i3* @barz(i64 %i9) nounwind {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  br label %while.body

while.body:                                       ; preds = %while.cond
  br label %while.cond
}

declare void @llvm.dbg.declare(metadata, metadata)

define void @init() nounwind {
entry:
  %i14 = alloca %struct.i14*, align 8
  call void @llvm.dbg.declare(metadata !{%struct.i14** %i14}, metadata !25)
  store %struct.i14* null, %struct.i14** %i14, align 8
  %call = call i32 @foo(i8* bitcast (void ()* @bar to i8*), %struct.i14** %i14)
  %0 = load %struct.i14** %i14, align 8
  %i16 = getelementptr inbounds %struct.i14* %0, i32 0, i32 0
  %1 = load i32* %i16, align 4
  %or = or i32 %1, 4
  store i32 %or, i32* %i16, align 4
  %call4 = call i32 @foo(i8* bitcast (void ()* @baz to i8*), %struct.i14** %i14)
  ret void
}

declare i32 @foo(i8*, %struct.i14**) nounwind

define internal void @bar() nounwind {
entry:
  %i9 = alloca i64, align 8
  store i64 0, i64* %i9, align 8
  %call = call i32 @put(i64 0, i64* %i9, i64 0, %struct.i24* null)
  ret void
}

define internal void @baz() nounwind {
entry:
  ret void
}

declare i32 @put(i64, i64*, i64, %struct.i24*) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!73}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.3 ", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !48, null, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !"i1", metadata !""}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !21, metadata !33, metadata !47}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"i2", metadata !"i2", metadata !"", i32 31, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, %struct.i3* (i64)* @barz, null, null, metadata !16, i32 32} ; [ DW_TAG_subprogram ] [line 31]  [scope 32]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !13}
!8 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from i3]
!9 = metadata !{i32 786451, metadata !1, null, metadata !"i3", i32 25, i64 32, i64 32, i32 0, i32 0, null, metadata !10, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [i3] [line 25, size 32, align 32, offset 0] [def] [from ]
!10 = metadata !{metadata !11}
!11 = metadata !{i32 786445, metadata !1, metadata !9, metadata !"i4", i32 26, i64 32, i64 32, i64 0, i32 0, metadata !12} ; [ DW_TAG_member ]  [line 26, size 32, align 32, offset 0] [from i5]
!12 = metadata !{i32 786468, null, null, metadata !"i5", i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]  [line 0, size 32, align 32, offset 0, enc DW_ATE_unsigned]
!13 = metadata !{i32 786454, metadata !1, null, metadata !"i6", i32 5, i64 0, i64 0, i64 0, i32 0, metadata !14} ; [ DW_TAG_typedef ]  [line 5, size 0, align 0, offset 0] [from i7]
!14 = metadata !{i32 786454, metadata !1, null, metadata !"i7", i32 2, i64 0, i64 0, i64 0, i32 0, metadata !15} ; [ DW_TAG_typedef ]  [line 2, size 0, align 0, offset 0] [from i8]
!15 = metadata !{i32 786468, null, null, metadata !"i8", i32 0, i64 64, i64 64, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]  [line 0, size 64, align 64, offset 0, enc DW_ATE_unsigned]
!16 = metadata !{}
!21 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"i13", metadata !"i13", metadata !"", i32 42, metadata !22, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void ()* @init, null, null, metadata !24, i32 43} ; [ DW_TAG_subprogram ] [line 42]  [scope 43]
!22 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !34, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!23 = metadata !{null}
!24 = metadata !{metadata !25}
!25 = metadata !{i32 786688, metadata !21, metadata !"i14", metadata !5, i32 45, metadata !27, i32 0, i32 0} ; [ DW_TAG_auto_variable ]  [line 45]
!27 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !28} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from i14]
!28 = metadata !{i32 786451, metadata !1, null, metadata !"i14", i32 16, i64 32, i64 32, i32 0, i32 0, null, metadata !29, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [i14] [line 16, size 32, align 32, offset 0] [def] [from ]
!29 = metadata !{metadata !30}
!30 = metadata !{i32 786445, metadata !1, metadata !28, metadata !"i16", i32 17, i64 32, i64 32, i64 0, i32 0, metadata !31} ; [ DW_TAG_member ]  [line 17, size 32, align 32, offset 0] [from i17]
!31 = metadata !{i32 786454, metadata !1, null, metadata !"i17", i32 7, i64 0, i64 0, i64 0, i32 0, metadata !32} ; [ DW_TAG_typedef ]  [line 7, size 0, align 0, offset 0] [from int]
!32 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]  [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!33 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"i18", metadata !"i18", metadata !"", i32 54, metadata !22, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 true, void ()* @bar, null, null, metadata !34, i32 55} ; [ DW_TAG_subprogram ] [line 54]   [scope 55]
!34 = metadata !{null}
!47 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"i29", metadata !"i29", metadata !"", i32 53, metadata !22, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 true, void ()* @baz, null, null, metadata !2, i32 53} ; [ DW_TAG_subprogram ] [line 53]
!48 = metadata !{metadata !49}
!49 = metadata !{i32 786484, i32 0, metadata !21, metadata !"i30", metadata !"i30", metadata !"", metadata !5, i32 44, metadata !50, i32 1, i32 1, null, null}
!50 = metadata !{i32 786454, metadata !1, null, metadata !"i31", i32 6, i64 0, i64 0, i64 0, i32 0, metadata !32} ; [ DW_TAG_typedef ]  [line 6, size 0, align 0, offset 0] [from int]
!52 = metadata !{i64 0}
!55 = metadata !{%struct.i3* null}
!72 = metadata !{%struct.i24* null}
!73 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
