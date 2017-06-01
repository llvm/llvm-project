; RUN: llc -march=hexagon -enable-pipeliner=false < %s | FileCheck %s

; Test that we generate a .cur

; CHECK: v{{[0-9]*}}.cur{{ *}}
; CHECK: v{{[0-9]*}}.cur{{ *}}

define void @conv3x3_i(i8* noalias nocapture readonly %iptr0, i32 %shift, i32 %width) #0 {
entry:
  br i1 undef, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  br label %for.body

for.body:
  %iptr0.pn = phi i8* [ %iptr0, %for.body.lr.ph ], [ %iptr0.addr.0121, %for.body ]
  %j.0115 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %for.body ]
  %sline000.0114 = phi <16 x i32> [ zeroinitializer, %for.body.lr.ph ], [ %1, %for.body ]
  %sline100.0113 = phi <16 x i32> [ zeroinitializer, %for.body.lr.ph ], [ zeroinitializer, %for.body ]
  %iptr0.addr.0121 = getelementptr inbounds i8, i8* %iptr0.pn, i32 64
  %0 = bitcast i8* %iptr0.addr.0121 to <16 x i32>*
  %1 = load <16 x i32>, <16 x i32>* %0, align 64, !tbaa !1
  %2 = load <16 x i32>, <16 x i32>* null, align 64, !tbaa !1
  %3 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %1, <16 x i32> %sline000.0114, i32 4)
  %4 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> zeroinitializer, <16 x i32> %sline100.0113, i32 4)
  %5 = tail call <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32> %2, <16 x i32> zeroinitializer, i32 4)
  %6 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %3, <16 x i32> %sline000.0114)
  %7 = tail call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %5, <16 x i32> zeroinitializer)
  %8 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi(<32 x i32> %6, i32 0, i32 0)
  %9 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %8, <32 x i32> zeroinitializer, i32 undef, i32 0)
  %10 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> %9, <32 x i32> undef, i32 undef, i32 0)
  %11 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %10)
  %12 = tail call <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32> %11, <16 x i32> undef, i32 %shift)
  %13 = tail call <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32> undef, <16 x i32> %12)
  store <16 x i32> %13, <16 x i32>* undef, align 64, !tbaa !1
  %14 = tail call <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32> zeroinitializer, <32 x i32> %7, i32 undef, i32 1)
  %15 = tail call <16 x i32> @llvm.hexagon.V6.hi(<32 x i32> %14)
  %16 = tail call <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32> %15, <16 x i32> undef, i32 %shift)
  %17 = tail call <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32> %16, <16 x i32> undef)
  store <16 x i32> %17, <16 x i32>* undef, align 64, !tbaa !1
  %add = add nsw i32 %j.0115, 64
  %cmp = icmp slt i32 %add, %width
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

declare <16 x i32> @llvm.hexagon.V6.valignbi(<16 x i32>, <16 x i32>, i32) #1
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vrmpybusi(<32 x i32>, i32, i32) #1
declare <32 x i32> @llvm.hexagon.V6.vrmpybusi.acc(<32 x i32>, <32 x i32>, i32, i32) #1
declare <16 x i32> @llvm.hexagon.V6.vasrwh(<16 x i32>, <16 x i32>, i32) #1
declare <16 x i32> @llvm.hexagon.V6.hi(<32 x i32>) #1
declare <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx" }
attributes #1 = { nounwind readnone }

!1 = !{!2, !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
