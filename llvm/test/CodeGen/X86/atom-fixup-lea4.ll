; RUN: llc < %s -mcpu=atom -mtriple=x86_64-linux

%struct.ValueWrapper = type { double }
%struct.ValueWrapper.6 = type { %struct.ValueWrapper.7 }
%struct.ValueWrapper.7 = type { %struct.ValueWrapper.8 }
%struct.ValueWrapper.8 = type { %struct.ValueWrapper }

; Function Attrs: uwtable
define linkonce_odr void @_ZN12ValueWrapperIS_IS_IS_IdEEEEC2Ev(ptr %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr
  call void @_ZN12ValueWrapperIS_IS_IdEEEC2Ev(ptr %this1)
  ret void
}

; Function Attrs: uwtable
declare void @_ZN12ValueWrapperIS_IS_IdEEEC2Ev(ptr) unnamed_addr #0 align 2

attributes #0 = { uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

