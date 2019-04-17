; RUN: opt < %s -loop-reduce

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32-S128"

%struct.nsTArray = type { i8 }
%struct.nsTArrayHeader = type { i32 }

define void @_Z6foobarR8nsTArray(%struct.nsTArray* %aValues, i32 %foo, %struct.nsTArrayHeader* %bar) nounwind {
entry:
  br label %for.body

for.body:                                         ; preds = %_ZN8nsTArray9ElementAtEi.exit, %entry
  %i.06 = phi i32 [ %add, %_ZN8nsTArray9ElementAtEi.exit ], [ 0, %entry ]
  %call.i = call %struct.nsTArrayHeader* @_ZN8nsTArray4Hdr2Ev() nounwind
  %add.ptr.i = getelementptr inbounds %struct.nsTArrayHeader, %struct.nsTArrayHeader* %call.i, i32 1
  %tmp = bitcast %struct.nsTArrayHeader* %add.ptr.i to %struct.nsTArray*
  %arrayidx = getelementptr inbounds %struct.nsTArray, %struct.nsTArray* %tmp, i32 %i.06
  %add = add nsw i32 %i.06, 1
  call void @llvm.dbg.value(metadata %struct.nsTArray* %aValues, metadata !0, metadata !DIExpression()) nounwind, !dbg !DILocation(scope: !1)
  br label %_ZN8nsTArray9ElementAtEi.exit

_ZN8nsTArray9ElementAtEi.exit:                    ; preds = %for.body
  %arrayidx.i = getelementptr inbounds %struct.nsTArray, %struct.nsTArray* %tmp, i32 %add
  call void @_ZN11nsTArray15ComputeDistanceERKS_Rd(%struct.nsTArray* %arrayidx, %struct.nsTArray* %arrayidx.i) nounwind
  %cmp = icmp slt i32 %add, %foo
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %_ZN8nsTArray9ElementAtEi.exit
  ret void
}

declare void @_ZN11nsTArray15ComputeDistanceERKS_Rd(%struct.nsTArray*, %struct.nsTArray*)

declare %struct.nsTArrayHeader* @_ZN8nsTArray4Hdr2Ev()

declare void @llvm.dbg.value(metadata, metadata, metadata) nounwind readnone

!0 = !DILocalVariable(scope: !1)
!1 = distinct !DISubprogram()
