; RUN: llc -mtriple aarch64_be-linux-gnu -filetype obj < %s | llvm-objdump -s - | FileCheck %s

; ARM EHABI for big endian
; This test case checks whether CIE length record is laid out in big endian format.
;
; This is the LLVM assembly generated from following C++ code:
;
; extern void foo(int);
; void test(int a, int b) {
;   try {
;   foo(a);
; } catch (...) {
;   foo(b);
; }
;}

define void @_Z4testii(i32 %a, i32 %b) #0 personality ptr @__gxx_personality_v0 {
entry:
  invoke void @_Z3fooi(i32 %a)
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          catch ptr null
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = tail call ptr @__cxa_begin_catch(ptr %1) #2
  invoke void @_Z3fooi(i32 %b)
          to label %invoke.cont2 unwind label %lpad1

invoke.cont2:                                     ; preds = %lpad
  tail call void @__cxa_end_catch()
  br label %try.cont

try.cont:                                         ; preds = %entry, %invoke.cont2
  ret void

lpad1:                                            ; preds = %lpad
  %3 = landingpad { ptr, i32 }
          cleanup
  invoke void @__cxa_end_catch()
          to label %eh.resume unwind label %terminate.lpad

eh.resume:                                        ; preds = %lpad1
  resume { ptr, i32 } %3

terminate.lpad:                                   ; preds = %lpad1
  %4 = landingpad { ptr, i32 }
          catch ptr null
  %5 = extractvalue { ptr, i32 } %4, 0
  tail call void @__clang_call_terminate(ptr %5) #3
  unreachable
}

declare void @_Z3fooi(i32) #0

declare i32 @__gxx_personality_v0(...)

declare ptr @__cxa_begin_catch(ptr)

declare void @__cxa_end_catch()

; Function Attrs: noinline noreturn nounwind
define linkonce_odr hidden void @__clang_call_terminate(ptr) #1 {
  %2 = tail call ptr @__cxa_begin_catch(ptr %0) #2
  tail call void @_ZSt9terminatev() #3
  unreachable
}

declare void @_ZSt9terminatev()

; CHECK-LABEL: Contents of section .eh_frame:
; CHECK-NEXT: {{^ 0000}}
; CHECK-NEXT: {{^ 0010}}
; CHECK-NEXT: 0020 0000000c 00440e10 9e040000 0000001c .....D..........
; CHECK-NEXT: 0030 00000000 017a504c 5200017c 1e0b9c00 .....zPLR..|....

