; RUN: llc < %s -mtriple=arm-apple-darwin9 | FileCheck %s

; CHECK: ldr r0, [[CPI_PERSONALITY:[A-Za-z0-9_]+]]
; CHECK: ldr r0, [[CPI_LSDA:[A-Za-z0-9_]+]]
; CHECK: [[CPI_LSDA]]:
; CHECK: .long  [[LSDA_LABEL:[A-Za-z0-9_]+]]-
; CHECK: [[LSDA_LABEL]]:
; CHECK: .byte   255                     @ @LPStart Encoding = omit

%struct.A = type { i32* }

define void @"\01-[MyFunction Name:]"() personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*) {
entry:
  %save_filt.1 = alloca i32
  %save_eptr.0 = alloca i8*
  %a = alloca %struct.A
  %eh_exception = alloca i8*
  %eh_selector = alloca i32
  %"alloca point" = bitcast i32 0 to i32
  call void @_ZN1AC1Ev(%struct.A* %a)
  invoke void @_Z3barv()
          to label %invcont unwind label %lpad

invcont:                                          ; preds = %entry
  call void @_ZN1AD1Ev(%struct.A* %a) nounwind
  br label %return

bb:                                               ; preds = %ppad
  %eh_select = load i32, i32* %eh_selector
  store i32 %eh_select, i32* %save_filt.1, align 4
  %eh_value = load i8*, i8** %eh_exception
  store i8* %eh_value, i8** %save_eptr.0, align 4
  call void @_ZN1AD1Ev(%struct.A* %a) nounwind
  %0 = load i8*, i8** %save_eptr.0, align 4
  store i8* %0, i8** %eh_exception, align 4
  %1 = load i32, i32* %save_filt.1, align 4
  store i32 %1, i32* %eh_selector, align 4
  br label %Unwind

return:                                           ; preds = %invcont
  ret void

lpad:                                             ; preds = %entry
  %exn = landingpad {i8*, i32}
           cleanup
  %eh_ptr = extractvalue {i8*, i32} %exn, 0
  store i8* %eh_ptr, i8** %eh_exception
  %eh_select2 = extractvalue {i8*, i32} %exn, 1
  store i32 %eh_select2, i32* %eh_selector
  br label %ppad

ppad:                                             ; preds = %lpad
  br label %bb

Unwind:                                           ; preds = %bb
  %eh_ptr3 = load i8*, i8** %eh_exception
  call void @_Unwind_SjLj_Resume(i8* %eh_ptr3)
  unreachable
}

define linkonce_odr void @_ZN1AC1Ev(%struct.A* %this) {
entry:
  %this_addr = alloca %struct.A*
  %"alloca point" = bitcast i32 0 to i32
  store %struct.A* %this, %struct.A** %this_addr
  %0 = call i8* @_Znwm(i32 4)
  %1 = bitcast i8* %0 to i32*
  %2 = load %struct.A*, %struct.A** %this_addr, align 4
  %3 = getelementptr inbounds %struct.A, %struct.A* %2, i32 0, i32 0
  store i32* %1, i32** %3, align 4
  br label %return

return:                                           ; preds = %entry
  ret void
}

declare i8* @_Znwm(i32)

define linkonce_odr void @_ZN1AD1Ev(%struct.A* %this) nounwind {
entry:
  %this_addr = alloca %struct.A*
  %"alloca point" = bitcast i32 0 to i32
  store %struct.A* %this, %struct.A** %this_addr
  %0 = load %struct.A*, %struct.A** %this_addr, align 4
  %1 = getelementptr inbounds %struct.A, %struct.A* %0, i32 0, i32 0
  %2 = load i32*, i32** %1, align 4
  %3 = bitcast i32* %2 to i8*
  call void @_ZdlPv(i8* %3) nounwind
  br label %bb

bb:                                               ; preds = %entry
  br label %return

return:                                           ; preds = %bb
  ret void
}

declare void @_ZdlPv(i8*) nounwind

declare void @_Z3barv()

declare i32 @llvm.eh.typeid.for(i8*) nounwind

declare i32 @__gxx_personality_sj0(...)

declare void @_Unwind_SjLj_Resume(i8*)
