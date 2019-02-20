; RUN: llc -mtriple mips-unknown-linux-gnu -mattr=+micromips -O3 -filetype=obj -o - %s | llvm-readelf -r | FileCheck %s

; CHECK: .rel.eh_frame
; CHECK: DW.ref.__gxx_personality_v0
; CHECK-NEXT: .text
; CHECK-NEXT: .gcc_except_table

@_ZTIi = external constant i8*

define dso_local i32 @main() local_unnamed_addr personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %exception.i = tail call i8* @__cxa_allocate_exception(i32 4) nounwind
  %0 = bitcast i8* %exception.i to i32*
  store i32 5, i32* %0, align 16
  invoke void @__cxa_throw(i8* %exception.i, i8* bitcast (i8** @_ZTIi to i8*), i8* null) noreturn
          to label %.noexc unwind label %return

.noexc:
  unreachable

return:
  %1 = landingpad { i8*, i32 }
          catch i8* null
  %2 = extractvalue { i8*, i32 } %1, 0
  %3 = tail call i8* @__cxa_begin_catch(i8* %2) nounwind
  tail call void @__cxa_end_catch()
  ret i32 0
}

declare i32 @__gxx_personality_v0(...)

declare i8* @__cxa_begin_catch(i8*) local_unnamed_addr

declare void @__cxa_end_catch() local_unnamed_addr

declare i8* @__cxa_allocate_exception(i32) local_unnamed_addr

declare void @__cxa_throw(i8*, i8*, i8*) local_unnamed_addr
