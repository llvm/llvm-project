; RUN: llc < %s -mcpu=atom -mtriple=x86_64-linux

%struct.ValueWrapper = type { double }
%struct.ValueWrapper.6 = type { %struct.ValueWrapper.7 }
%struct.ValueWrapper.7 = type { %struct.ValueWrapper.8 }
%struct.ValueWrapper.8 = type { %struct.ValueWrapper }

; Function Attrs: uwtable
define linkonce_odr void @_ZN12ValueWrapperIS_IS_IS_IdEEEEC2Ev(%struct.ValueWrapper.6* %this) unnamed_addr #0 align 2 {
entry:
  %this.addr = alloca %struct.ValueWrapper.6*, align 8
  store %struct.ValueWrapper.6* %this, %struct.ValueWrapper.6** %this.addr, align 8
  %this1 = load %struct.ValueWrapper.6*, %struct.ValueWrapper.6** %this.addr
  %value = getelementptr inbounds %struct.ValueWrapper.6, %struct.ValueWrapper.6* %this1, i32 0, i32 0
  call void @_ZN12ValueWrapperIS_IS_IdEEEC2Ev(%struct.ValueWrapper.7* %value)
  ret void
}

; Function Attrs: uwtable
declare void @_ZN12ValueWrapperIS_IS_IdEEEC2Ev(%struct.ValueWrapper.7*) unnamed_addr #0 align 2

attributes #0 = { uwtable "frame-pointer"="all" "stack-protector-buffer-size"="8" "use-soft-float"="false" }

