; RUN: split-file %s %t

; RUN: llc -O0 -mtriple=x86_64-unknown-linux-gnu %t/dwarf.ll -o - | FileCheck %s --check-prefix=DWARF
; RUN: llc -O0 -mtriple=x86_64-unknown-linux-gnu %t/sjlj.ll -o - | FileCheck %s --check-prefix=SJLJ
; RUN: llc -O0 -mtriple=x86_64-pc-windows-msvc %t/wineh.ll -o - | FileCheck %s --check-prefix=WINEH

; DWARF: .cfi_lsda
; DWARF: GCC_except_table

; SJLJ: _Unwind_SjLj_Register

; WINEH: __CxxFrameHandler3
; WINEH: .seh_handlerdata

;--- dwarf.ll
declare void @g()
declare i32 @__gxx_personality_v0(...)

define void @f() personality ptr @__gxx_personality_v0 {
  invoke void @g() to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %p = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %p
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"dwarf"}

;--- sjlj.ll
declare void @g()
declare i32 @__gxx_personality_sj0(...)

define void @f() personality ptr @__gxx_personality_sj0 {
  invoke void @g() to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %p = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %p
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"sjlj"}

;--- wineh.ll
declare void @g()
declare i32 @__CxxFrameHandler3(...)

define void @f() personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @g() to label %cont unwind label %catch.dispatch
catch.dispatch:
  %cs = catchswitch within none [label %catch] unwind to caller
catch:
  %cp = catchpad within %cs [ptr null, i32 64, ptr null]
  catchret from %cp to label %cont
cont:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"exception-model", !"wineh"}
