; RUN: llc < %s -verify-machineinstrs -mtriple=s390x-linux-gnu | FileCheck -check-prefix=CHECK-FUNC %s
; RUN: llc < %s -verify-machineinstrs -mtriple=s390x-linux-gnu | FileCheck -check-prefix=CHECK-ET %s
; RUN: llc < %s -verify-machineinstrs -mtriple=s390x-linux-gnu -relocation-model=pic | FileCheck -check-prefix=CHECK-REF %s

declare i32 @__gxx_personality_v0(...)

declare void @bar()

define i64 @foo(i64 %lhs, i64 %rhs) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  invoke void @bar() to label %end unwind label %clean
end:
 ret i64 0

clean:
  %tst = landingpad { i8*, i32 } cleanup
  ret i64 42
}

; CHECK-FUNC: foo:
; CHECK-FUNC: .cfi_startproc
; CHECK-FUNC: .cfi_personality 0, __gxx_personality_v0
; CHECK-FUNC: .cfi_lsda 0, .Lexception0
; CHECK-FUNC: stmg	%r14, %r15, 112(%r15)
; CHECK-FUNC: .cfi_offset %r14, -48
; CHECK-FUNC: .cfi_offset %r15, -40
; CHECK-FUNC: aghi	%r15, -160
; CHECK-FUNC: .cfi_def_cfa_offset 320
; ...main function...
; CHECK-FUNC: .cfi_endproc
;
; CHECK-ET: .section	.gcc_except_table,"a",@progbits
; CHECK-ET-NEXT: .p2align	2
; CHECK-ET-NEXT: GCC_except_table0:
; CHECK-ET-NEXT: .Lexception0:
;
; CHECK-REF: .cfi_personality 155, DW.ref.__gxx_personality_v0
; CHECK-REF: .cfi_lsda 27, .Lexception0
; CHECK-REF: .hidden	DW.ref.__gxx_personality_v0
; CHECK-REF: .weak	DW.ref.__gxx_personality_v0
; CHECK-REF: .section	.data.DW.ref.__gxx_personality_v0,"aGw",@progbits,DW.ref.__gxx_personality_v0,comdat
; CHECK-REF-NEXT: .p2align	3
; CHECK-REF-NEXT: .type	DW.ref.__gxx_personality_v0,@object
; CHECK-REF-NEXT: .size	DW.ref.__gxx_personality_v0, 8
; CHECK-REF-NEXT: DW.ref.__gxx_personality_v0:
; CHECK-REF-NEXT: .quad	__gxx_personality_v0
