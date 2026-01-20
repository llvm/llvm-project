; REQUIRES: asserts
;
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1201 -O1 < %s -o /dev/null
;
; Test that SplitAnalysis doesn't crash when UseSlots contains uses in blocks that
; are not covered by the live interval.

%struct.S1 = type { i8, %struct.S0 }
%struct.S0 = type { i64 }

define i16 @backsmith_pure_2(ptr addrspace(5) %BS_VAR_0, i1 %tobool.not, i64 %add919, <32 x i8> %shuffle57, <32 x i8> %vecinit827, <16 x i16> %0) {
entry:
  %BS_VAR_01 = alloca [8 x <32 x i8>], align 32, addrspace(5)
  %call2 = call i8 @backsmith_snippet_44()
  br i1 %tobool.not, label %if.end, label %if.then, !prof !0

if.then:                                          ; preds = %entry
  br i1 %tobool.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %if.then
  %call976 = call <4 x i64> @backsmith_pure_0(<16 x i16> zeroinitializer)
  br label %if.end

for.body:                                         ; preds = %for.body, %if.then
  store <32 x i8> %shuffle57, ptr addrspace(5) %BS_VAR_0, align 32
  %call394 = call <4 x i16> @backsmith_snippet_679()
  %call393 = call <4 x i16> @backsmith_snippet_679()
  %cmp104 = icmp ne <32 x i8> %vecinit827, zeroinitializer
  %sext = sext <32 x i1> %cmp104 to <32 x i8>
  %vecext347 = extractelement <32 x i8> %sext, i64 29
  %cond893.v = select i1 %tobool.not, i8 0, i8 %vecext347
  %vecinit895 = insertelement <32 x i8> %vecinit827, i8 %cond893.v, i64 29
  store <32 x i8> %vecinit895, ptr addrspace(5) null, align 32
  %exitcond.not = icmp eq i64 0, %add919
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

if.end:                                           ; preds = %for.cond.cleanup, %entry
  %call1066 = call <4 x i64> @backsmith_pure_0(<16 x i16> %0)
  ret i16 0
}

declare i8 @backsmith_snippet_44()

declare <4 x i16> @backsmith_snippet_679()

declare <4 x i64> @backsmith_pure_0(<16 x i16>)

define %struct.S1 @func_103(i8 %0, i1 %tobool2.not30) {
entry:
  %add = shufflevector <4 x i16> splat (i16 1), <4 x i16> zeroinitializer, <4 x i32> zeroinitializer
  br label %for.cond1.preheader

for.cond.loopexit:                                ; preds = %for.end, %for.cond1.preheader
  %tobool.not = icmp eq i8 %0, 0
  br i1 %tobool.not, label %for.end25, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond.loopexit, %entry
  br i1 %tobool2.not30, label %for.cond.loopexit, label %for.body3

for.body3:                                        ; preds = %for.end, %for.cond1.preheader
  br label %for.cond4

for.cond4:                                        ; preds = %for.cond4, %for.body3
  %tobool7.not = icmp eq i64 0, 0
  br i1 %tobool7.not, label %for.end, label %for.cond4

for.end:                                          ; preds = %for.cond4
  %call121 = call i16 @backsmith_pure_2(ptr addrspace(5) null, i1 false, i64 0, <32 x i8> zeroinitializer, <32 x i8> zeroinitializer, <16 x i16> zeroinitializer)
  store <4 x i16> %add, ptr addrspace(5) null, align 8
  tail call void @backsmith_pure_7(<16 x i64> zeroinitializer, i32 -352674805, <2 x i32> splat (i32 -247168742))
  tail call void @backsmith_pure_7(<16 x i64> zeroinitializer, i32 -352674805, <2 x i32> splat (i32 -247168742))
  %tobool2.not = icmp eq ptr addrspace(5) null, null
  br i1 %tobool2.not, label %for.cond.loopexit, label %for.body3

for.end25:                                        ; preds = %for.cond.loopexit
  ret %struct.S1 zeroinitializer
}

declare void @backsmith_pure_7(<16 x i64>, i32, <2 x i32>)

; uselistorder directives
uselistorder ptr @backsmith_snippet_679, { 1, 0 }
uselistorder ptr @backsmith_pure_0, { 1, 0 }
uselistorder ptr @backsmith_pure_7, { 1, 0 }

!0 = !{!"branch_weights", !"expected", i32 2000, i32 1}
