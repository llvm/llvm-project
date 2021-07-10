; RUN: opt -passes=simplifycfg -S < %s | FileCheck %s

; Ensure that we do not crash when trying to evaluate alignment of a <1 x ???*>.

%struct.ap = type { i8 }
%"struct.z::ai" = type { i32 }

@x = external dso_local local_unnamed_addr global i32, align 4

define dso_local void @_ZN2ap2aqEv(%struct.ap* %this, i1 %c, i32 %v2) {
; CHECK-LABEL: @_ZN2ap2aqEv(
_ZN1yIN1z2aiE2aaIS1_EE2ahEv.exit:
  br i1 %c, label %if.end, label %land.rhs

land.rhs:                                         ; preds = %_ZN1yIN1z2aiE2aaIS1_EE2ahEv.exit
  %0 = bitcast <1 x %"struct.z::ai"*> zeroinitializer to %"struct.z::ai"*
  %retval.sroa.0.0..sroa_idx.i = getelementptr inbounds %"struct.z::ai", %"struct.z::ai"* %0, i64 0, i32 0
  %retval.sroa.0.0.copyload.i = load i32, i32* %retval.sroa.0.0..sroa_idx.i, align 4
  %tobool5 = icmp eq i32 %retval.sroa.0.0.copyload.i, 0
  %spec.select = select i1 %tobool5, i32 %v2, i32 0
  br label %if.end

if.end:                                           ; preds = %land.rhs, %_ZN1yIN1z2aiE2aaIS1_EE2ahEv.exit
  %b.0 = phi i32 [ %spec.select, %land.rhs ], [ 0, %_ZN1yIN1z2aiE2aaIS1_EE2ahEv.exit ]
  ret void
}
