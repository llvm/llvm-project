; RUN: llc < %s -mtriple=x86_64 -function-sections -basic-block-sections=labels -yk-extended-llvmbbaddrmap-section -emulated-tls | FileCheck %s

@G = thread_local global i32 0

declare void @foo(ptr)

define void @bar() noinline {
  ret void
}

declare void @baz()

define dso_local void @the_func(ptr %0) {
  ; Note that the emulated TLS access will make an extra direct call with an
  ; unknown target.
  call void @foo(ptr @G)
  call void @bar()
  call void %0()
  ret void
}

; CHECK:		.section .llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text.the_func{{$}}
; CHECK-NEXT:		.byte 2			    # version
; CHECK-NEXT:		.byte 0			    # feature
; CHECK-NEXT:		.quad .Lfunc_begin1 # function address
; CHECK-NEXT:	    .byte 1             # number of basic blocks
; CHECK-NEXT:		.byte 0             # BB id
; CHECK-NEXT:		.uleb128 .Lfunc_begin1-.Lfunc_begin1
; CHECK-NEXT:       .uleb128 .LBB_END1_0-.Lfunc_begin1
; CHECK-NEXT:		.byte 1
; CHECK-NEXT:		.byte 1               # num corresponding blocks
; CHECK-NEXT:		.byte 0               # corresponding block
; CHECK-NEXT:		.byte 4               # num calls
; CHECK-NEXT:		.quad .Lyk_precall0   # call offset
; CHECK-NEXT:		.quad .Lyk_postcall0  # return offset
; CHECK-NEXT:		.quad 0               # target offset
; CHECK-NEXT:		.byte 1               # direct?
; CHECK-NEXT:		.quad .Lyk_precall1   # call offset
; CHECK-NEXT:		.quad .Lyk_postcall1  # return offset
; CHECK-NEXT:		.quad foo             # target offset
; CHECK-NEXT:		.byte 1               # direct?
; CHECK-NEXT:		.quad .Lyk_precall2   # call offset
; CHECK-NEXT:		.quad .Lyk_postcall2  # return offset
; CHECK-NEXT:		.quad bar             # target offset
; CHECK-NEXT:		.byte 1               # direct?
; CHECK-NEXT:		.quad .Lyk_precall3   # call offset
; CHECK-NEXT:		.quad .Lyk_postcall3  # return offset
; CHECK-NEXT:		.quad 0               # target offset
; CHECK-NEXT:		.byte 0               # direct?

; FIXME: test our other extensions to the blockmap.
