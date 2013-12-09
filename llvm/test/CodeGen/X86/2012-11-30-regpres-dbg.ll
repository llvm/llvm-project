; RUN: llc < %s -mtriple=x86_64-apple-macosx -enable-misched \
; RUN:          -verify-machineinstrs | FileCheck %s
;
; Test RegisterPressure handling of DBG_VALUE.
;
; CHECK: %entry
; CHECK: DEBUG_VALUE: callback
; CHECK: ret

%struct.btCompoundLeafCallback = type { i32, i32 }

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define void @test() unnamed_addr uwtable ssp align 2 {
entry:
  %callback = alloca %struct.btCompoundLeafCallback, align 8
  br i1 undef, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  unreachable

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata !{%struct.btCompoundLeafCallback* %callback}, metadata !3)
  %m = getelementptr inbounds %struct.btCompoundLeafCallback* %callback, i64 0, i32 1
  store i32 0, i32* undef, align 8
  %cmp12447 = icmp sgt i32 undef, 0
  br i1 %cmp12447, label %for.body.lr.ph, label %invoke.cont44

for.body.lr.ph:                                   ; preds = %if.end
  unreachable

invoke.cont44:                                    ; preds = %if.end
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8}

!0 = metadata !{i32 786449, metadata !6, i32 4, metadata !"clang version 3.3 (trunk 168984) (llvm/trunk 168983)", i1 true, metadata !"", i32 0, metadata !2, metadata !7, null, null, null, metadata !""} ; [ DW_TAG_compile_unit ] [MultiSource/Benchmarks/Bullet/MultiSource/Benchmarks/Bullet/btCompoundCollisionAlgorithm.cpp] [DW_LANG_C_plus_plus]
!2 = metadata !{null}
!3 = metadata !{i32 786688, null, metadata !"callback", null, i32 214, metadata !4, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [callback] [line 214]
!4 = metadata !{i32 786451, metadata !6, null, metadata !"btCompoundLeafCallback", i32 90, i64 512, i64 64, i32 0, i32 0, null, null, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [btCompoundLeafCallback] [line 90, size 512, align 64, offset 0] [def] [from ]
!5 = metadata !{i32 786473, metadata !6} ; [ DW_TAG_file_type ]
!6 = metadata !{metadata !"MultiSource/Benchmarks/Bullet/btCompoundCollisionAlgorithm.cpp", metadata !"MultiSource/Benchmarks/Bullet"}
!7 = metadata !{i32 0}
!8 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
