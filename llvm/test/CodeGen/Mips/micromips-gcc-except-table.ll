; RUN: llc -mtriple=mips-linux-gnu -mcpu=mips32r2 -mattr=+micromips -O3 -filetype=obj < %s | llvm-objdump -s -j .gcc_except_table - | FileCheck %s

; CHECK: Contents of section .gcc_except_table:
; CHECK-NEXT: 0000 ff9b1501 0c001000 00100e1e 011e1800
; CHECK-NEXT: 0010 00010000 00000000

@_ZTIi = external constant ptr

define dso_local i32 @main() local_unnamed_addr norecurse personality ptr @__gxx_personality_v0 {
entry:
  %exception.i = tail call ptr @__cxa_allocate_exception(i32 4) nounwind
  store i32 5, ptr %exception.i, align 16
  invoke void @__cxa_throw(ptr %exception.i, ptr @_ZTIi, ptr null) noreturn
          to label %.noexc unwind label %return

.noexc:
  unreachable

return:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = tail call ptr @__cxa_begin_catch(ptr %1) nounwind
  tail call void @__cxa_end_catch()
  ret i32 0
}

declare i32 @__gxx_personality_v0(...)

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

declare void @__cxa_end_catch() local_unnamed_addr

declare ptr @__cxa_allocate_exception(i32) local_unnamed_addr

declare void @__cxa_throw(ptr, ptr, ptr) local_unnamed_addr
