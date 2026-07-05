; RUN: llc < %s -mtriple=thumbv7-apple-darwin10
; <rdar://problem/8264008>

define linkonce_odr arm_apcscc void @func1() personality ptr @__gxx_personality_sj0 {
entry:
  %save_filt.936 = alloca i32                     ; <ptr> [#uses=2]
  %save_eptr.935 = alloca ptr                     ; <ptr> [#uses=2]
  %eh_exception = alloca ptr                      ; <ptr> [#uses=5]
  %eh_selector = alloca i32                       ; <ptr> [#uses=3]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call arm_apcscc  void @func2()
  br label %return

bb:                                               ; No predecessors!
  %eh_select = load i32, ptr %eh_selector             ; <i32> [#uses=1]
  store i32 %eh_select, ptr %save_filt.936, align 4
  %eh_value = load ptr, ptr %eh_exception             ; <ptr> [#uses=1]
  store ptr %eh_value, ptr %save_eptr.935, align 4
  invoke arm_apcscc  void @func3()
          to label %invcont unwind label %lpad

invcont:                                          ; preds = %bb
  %tmp6 = load ptr, ptr %save_eptr.935, align 4          ; <ptr> [#uses=1]
  store ptr %tmp6, ptr %eh_exception, align 4
  %tmp7 = load i32, ptr %save_filt.936, align 4          ; <i32> [#uses=1]
  store i32 %tmp7, ptr %eh_selector, align 4
  br label %Unwind

bb12:                                             ; preds = %ppad
  call arm_apcscc  void @_ZSt9terminatev() noreturn nounwind
  unreachable

return:                                           ; preds = %entry
  ret void

lpad:                                             ; preds = %bb
  %eh_ptr = landingpad { ptr, i32 }
              cleanup
  %exn = extractvalue { ptr, i32 } %eh_ptr, 0
  store ptr %exn, ptr %eh_exception
  %eh_ptr13 = load ptr, ptr %eh_exception             ; <ptr> [#uses=1]
  %eh_select14 = extractvalue { ptr, i32 } %eh_ptr, 1
  store i32 %eh_select14, ptr %eh_selector
  br label %ppad

ppad:
  br label %bb12

Unwind:
  %eh_ptr15 = load ptr, ptr %eh_exception
  call arm_apcscc  void @_Unwind_SjLj_Resume(ptr %eh_ptr15)
  unreachable
}

declare arm_apcscc void @func2()

declare arm_apcscc void @_ZSt9terminatev() noreturn nounwind

declare arm_apcscc void @_Unwind_SjLj_Resume(ptr)

declare arm_apcscc void @func3()

declare arm_apcscc i32 @__gxx_personality_sj0(...)
