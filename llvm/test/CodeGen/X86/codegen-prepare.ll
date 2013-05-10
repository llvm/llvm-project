; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck %s

; Check that the CodeGenPrepare Pass
; does not wrongly rewrite the address computed by Instruction %4
; as [12 + Base:%this].

; This test makes sure that:
; - both the store and the first load instructions
;   within basic block labeled 'if.then' are not removed. 
; - the store instruction stores a value at address [60 + %this]
; - the first load instruction loads a value at address [12 + %this]

%class.A = type { %struct.B }
%struct.B = type { %class.C, %class.D, %class.C, %class.D }
%class.C = type { float, float, float }
%class.D = type { [3 x %class.C] }

define linkonce_odr void @foo(%class.A* nocapture %this, i32 %BoolValue) nounwind uwtable {
entry:
  %cmp = icmp eq i32 %BoolValue, 0
  %address1 = getelementptr inbounds %class.A* %this, i64 0, i32 0, i32 3
  %address2 = getelementptr inbounds %class.A* %this, i64 0, i32 0, i32 1
  br i1 %cmp, label %if.else, label %if.then

if.then:                                         ; preds = %entry
  %0 = getelementptr inbounds %class.D* %address2, i64 0, i32 0, i64 0, i32 0
  %1 = load float* %0, align 4 
  %2 = getelementptr inbounds float* %0, i64 3
  %3 = load float* %2, align 4 
  %4 = getelementptr inbounds %class.D* %address1, i64 0, i32 0, i64 0, i32 0
  store float %1, float* %4, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  br label %if.end

if.end:                                           ; preds = %if.then, %if.else, %entry
  ret void
}

; CHECK: foo:
; CHECK: movss 12([[THIS:%[a-zA-Z0-9]+]]), [[REGISTER:%[a-zA-Z0-9]+]]
; CHECK-NEXT: movss [[REGISTER]], 60([[THIS]])

