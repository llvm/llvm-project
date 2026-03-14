; RUN: llc < %s -mtriple=arm-apple-darwin9 | FileCheck %s

; CHECK: ldr r0, [[CPI_PERSONALITY:[A-Za-z0-9_]+]]
; CHECK: ldr r0, [[CPI_LSDA:[A-Za-z0-9_]+]]
; CHECK: [[CPI_LSDA]]:
; CHECK: .long  [[LSDA_LABEL:[A-Za-z0-9_]+]]-
; CHECK: [[LSDA_LABEL]]:
; CHECK: .byte   255                     @ @LPStart Encoding = omit

%struct.A = type { ptr }

define void @"\01-[MyFunction Name:]"() personality ptr @__gxx_personality_sj0 {
entry:
  %save_filt.1 = alloca i32
  %save_eptr.0 = alloca ptr
  %a = alloca %struct.A
  %eh_exception = alloca ptr
  %eh_selector = alloca i32
  %"alloca point" = bitcast i32 0 to i32
  call void @_ZN1AC1Ev(ptr %a)
  invoke void @_Z3barv()
          to label %invcont unwind label %lpad

invcont:                                          ; preds = %entry
  call void @_ZN1AD1Ev(ptr %a) nounwind
  br label %return

bb:                                               ; preds = %ppad
  %eh_select = load i32, ptr %eh_selector
  store i32 %eh_select, ptr %save_filt.1, align 4
  %eh_value = load ptr, ptr %eh_exception
  store ptr %eh_value, ptr %save_eptr.0, align 4
  call void @_ZN1AD1Ev(ptr %a) nounwind
  %0 = load ptr, ptr %save_eptr.0, align 4
  store ptr %0, ptr %eh_exception, align 4
  %1 = load i32, ptr %save_filt.1, align 4
  store i32 %1, ptr %eh_selector, align 4
  br label %Unwind

return:                                           ; preds = %invcont
  ret void

lpad:                                             ; preds = %entry
  %exn = landingpad {ptr, i32}
           cleanup
  %eh_ptr = extractvalue {ptr, i32} %exn, 0
  store ptr %eh_ptr, ptr %eh_exception
  %eh_select2 = extractvalue {ptr, i32} %exn, 1
  store i32 %eh_select2, ptr %eh_selector
  br label %ppad

ppad:                                             ; preds = %lpad
  br label %bb

Unwind:                                           ; preds = %bb
  %eh_ptr3 = load ptr, ptr %eh_exception
  call void @_Unwind_SjLj_Resume(ptr %eh_ptr3)
  unreachable
}

define linkonce_odr void @_ZN1AC1Ev(ptr %this) {
entry:
  %this_addr = alloca ptr
  %"alloca point" = bitcast i32 0 to i32
  store ptr %this, ptr %this_addr
  %0 = call ptr @_Znwm(i32 4)
  %1 = load ptr, ptr %this_addr, align 4
  store ptr %0, ptr %1, align 4
  br label %return

return:                                           ; preds = %entry
  ret void
}

declare ptr @_Znwm(i32)

define linkonce_odr void @_ZN1AD1Ev(ptr %this) nounwind {
entry:
  %this_addr = alloca ptr
  %"alloca point" = bitcast i32 0 to i32
  store ptr %this, ptr %this_addr
  %0 = load ptr, ptr %this_addr, align 4
  %1 = load ptr, ptr %0, align 4
  call void @_ZdlPv(ptr %1) nounwind
  br label %bb

bb:                                               ; preds = %entry
  br label %return

return:                                           ; preds = %bb
  ret void
}

declare void @_ZdlPv(ptr) nounwind

declare void @_Z3barv()

declare i32 @llvm.eh.typeid.for(ptr) nounwind

declare i32 @__gxx_personality_sj0(...)

declare void @_Unwind_SjLj_Resume(ptr)
