; RUN: llc -mtriple=i386-apple-macosx -relocation-model=pic < %s > /dev/null
; RUN: llc -mtriple=x86_64-apple-macosx -relocation-model=pic < %s > /dev/null

; rdar://12393897

%TRp = type { i32, ptr, i32, i32 }
%TRH = type { ptr, ptr, ptr, ptr, ptr }

define i32 @t(ptr inreg %rp) nounwind optsize ssp {
entry:
  %handler = getelementptr inbounds %TRp, ptr %rp, i32 0, i32 1
  %0 = load ptr, ptr %handler, align 4
  %sync = getelementptr inbounds %TRH, ptr %0, i32 0, i32 4
  %sync12 = load ptr, ptr %sync, align 4
  %call = tail call i32 %sync12(ptr inreg %rp) nounwind optsize
  ret i32 %call
}

%btConeShape = type { %btConvexInternalShape, float, float, float, [3 x i32] }
%btConvexInternalShape = type { %btConvexShape, %btVector, %btVector, float, float }
%btConvexShape = type { %btCollisionShape }
%btCollisionShape = type { ptr, i32, ptr }
%btVector = type { [4 x float] }

define { <2 x float>, <2 x float> } @t2(ptr %this) unnamed_addr uwtable ssp align 2 {
entry:
  br i1 undef, label %if.then, label %if.end17

if.then:                                          ; preds = %entry
  %vecnorm.sroa.2.8.copyload = load float, ptr undef, align 4
  %cmp4 = fcmp olt float undef, 0x3D10000000000000
  %vecnorm.sroa.2.8.copyload36 = select i1 %cmp4, float -1.000000e+00, float %vecnorm.sroa.2.8.copyload
  %call.i.i.i = tail call float @sqrtf(float 0.000000e+00) nounwind readnone
  %div.i.i = fdiv float 1.000000e+00, %call.i.i.i
  %mul7.i.i.i = fmul float %div.i.i, %vecnorm.sroa.2.8.copyload36
  %0 = load ptr, ptr undef, align 8
  %call12 = tail call float %0(ptr %this)
  %mul7.i.i = fmul float %call12, %mul7.i.i.i
  %retval.sroa.0.4.insert = insertelement <2 x float> zeroinitializer, float undef, i32 1
  %add13.i = fadd float undef, %mul7.i.i
  %retval.sroa.1.8.insert = insertelement <2 x float> undef, float %add13.i, i32 0
  br label %if.end17

if.end17:                                         ; preds = %if.then, %entry
  %retval.sroa.1.8.load3338 = phi <2 x float> [ %retval.sroa.1.8.insert, %if.then ], [ undef, %entry ]
  %retval.sroa.0.0.load3137 = phi <2 x float> [ %retval.sroa.0.4.insert, %if.then ], [ undef, %entry ]
  ret { <2 x float>, <2 x float> } undef
}

declare float @sqrtf(float) nounwind readnone
