; RUN: llc -mtriple x86_64-pc-linux-gnu < %s | FileCheck %s

; This test demonstrates that it is possible to use functions for typeinfo
; instead of global variables. While __gxx_personality_v0 would never know what
; to do with them, other EH schemes such as SEH might use them.

declare i32 @__gxx_personality_v0(...)
declare void @filt0()
declare void @filt1()
declare void @_Z1fv()
declare i32 @llvm.eh.typeid.for(ptr)

define i32 @main() uwtable personality ptr @__gxx_personality_v0 {
entry:
  invoke void @_Z1fv()
          to label %try.cont unwind label %lpad

try.cont:
  ret i32 0

lpad:
  %0 = landingpad { ptr, i32 }
          cleanup
          catch ptr @filt0
          catch ptr @filt1
  %sel = extractvalue { ptr, i32 } %0, 1
  %id0 = call i32 @llvm.eh.typeid.for(ptr @filt0)
  %is_f0 = icmp eq i32 %sel, %id0
  br i1 %is_f0, label %try.cont, label %check_f1

check_f1:
  %id1 = call i32 @llvm.eh.typeid.for(ptr @filt1)
  %is_f1 = icmp eq i32 %sel, %id1
  br i1 %is_f1, label %try.cont, label %eh.resume

eh.resume:
  resume { ptr, i32 } %0
}

; CHECK-LABEL: main:
; CHECK: .cfi_startproc
; CHECK: .cfi_personality 3, __gxx_personality_v0
; CHECK: .cfi_lsda 3, .Lexception0
; CHECK: .cfi_def_cfa_offset 16
; CHECK: callq _Z1fv
; CHECK: retq
; CHECK: cmpl $2, %edx
; CHECK: je
; CHECK: cmpl $1, %edx
; CHECK: je
; CHECK: callq _Unwind_Resume
; CHECK: .cfi_endproc
; CHECK: GCC_except_table0:
; CHECK: Lexception0:
