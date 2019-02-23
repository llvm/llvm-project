; RUN: llc < %s -asm-verbose=false -wasm-keep-registers | FileCheck --check-prefix=CHECK --check-prefix=FINI --check-prefix=NULL %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; Test that @llvm.global_dtors is properly lowered into @llvm.global_ctors,
; grouping dtor calls by priority and associated symbol.

declare void @orig_ctor()
declare void @orig_dtor0()
declare void @orig_dtor1a()
declare void @orig_dtor1b()
declare void @orig_dtor1c0()
declare void @orig_dtor1c1a()
declare void @orig_dtor1c1b()
declare void @orig_dtor65536()
declare void @after_the_null()

@associated1c0 = external global i8
@associated1c1 = external global i8

@llvm.global_ctors = appending global
[1 x { i32, void ()*, i8* }]
[
  { i32, void ()*, i8* } { i32 200, void ()* @orig_ctor, i8* null }
]

@llvm.global_dtors = appending global
[9 x { i32, void ()*, i8* }]
[
  { i32, void ()*, i8* } { i32 0, void ()* @orig_dtor0, i8* null },
  { i32, void ()*, i8* } { i32 1, void ()* @orig_dtor1a, i8* null },
  { i32, void ()*, i8* } { i32 1, void ()* @orig_dtor1b, i8* null },
  { i32, void ()*, i8* } { i32 1, void ()* @orig_dtor1c0, i8* @associated1c0 },
  { i32, void ()*, i8* } { i32 1, void ()* @orig_dtor1c1a, i8* @associated1c1 },
  { i32, void ()*, i8* } { i32 1, void ()* @orig_dtor1c1b, i8* @associated1c1 },
  { i32, void ()*, i8* } { i32 65535, void ()* @orig_dtor65536, i8* null },
  { i32, void ()*, i8* } { i32 65535, void ()* null, i8* null },
  { i32, void ()*, i8* } { i32 65535, void ()* @after_the_null, i8* null }
]

; CHECK-LABEL: .Lcall_dtors.0:
; CHECK-NEXT: .functype .Lcall_dtors.0 (i32) -> (){{$}}
; CHECK-NEXT: call            orig_dtor0{{$}}

; CHECK-LABEL: .Lregister_call_dtors.0:
; CHECK:      block
; CHECK-NEXT: i32.const       $push2=, .Lcall_dtors.0{{$}}
; CHECK-NEXT: i32.const       $push1=, 0
; CHECK-NEXT: i32.const       $push0=, __dso_handle
; CHECK-NEXT: i32.call        $push3=, __cxa_atexit, $pop2, $pop1, $pop0{{$}}
; CHECK-NEXT: br_if           0, $pop3
; CHECK-NEXT: return
;      CHECK: end_block
; CHECK-NEXT: unreachable

; CHECK-LABEL: .Lcall_dtors.1:
; CHECK-NEXT: .functype .Lcall_dtors.1 (i32) -> (){{$}}
; CHECK-NEXT: call            orig_dtor1a{{$}}
; CHECK-NEXT: call            orig_dtor1b{{$}}

; CHECK-LABEL: .Lregister_call_dtors.1:
; CHECK:      block
; CHECK-NEXT: i32.const       $push2=, .Lcall_dtors.1{{$}}
; CHECK-NEXT: i32.const       $push1=, 0
; CHECK-NEXT: i32.const       $push0=, __dso_handle
; CHECK-NEXT: i32.call        $push3=, __cxa_atexit, $pop2, $pop1, $pop0{{$}}
; CHECK-NEXT: br_if           0, $pop3
; CHECK-NEXT: return
;      CHECK: end_block
; CHECK-NEXT: unreachable

; CHECK-LABEL: .Lcall_dtors.1.associated1c0:
; CHECK-NEXT: .functype .Lcall_dtors.1.associated1c0 (i32) -> (){{$}}
; CHECK-NEXT: call            orig_dtor1c0{{$}}

; CHECK-LABEL: .Lregister_call_dtors.1.associated1c0:
; CHECK:      block
; CHECK-NEXT: i32.const       $push2=, .Lcall_dtors.1.associated1c0{{$}}
; CHECK-NEXT: i32.const       $push1=, 0
; CHECK-NEXT: i32.const       $push0=, __dso_handle
; CHECK-NEXT: i32.call        $push3=, __cxa_atexit, $pop2, $pop1, $pop0{{$}}
; CHECK-NEXT: br_if           0, $pop3
; CHECK-NEXT: return
;      CHECK: end_block
; CHECK-NEXT: unreachable

; CHECK-LABEL: .Lcall_dtors.1.associated1c1:
; CHECK-NEXT: .functype .Lcall_dtors.1.associated1c1 (i32) -> (){{$}}
; CHECK-NEXT: call            orig_dtor1c1a{{$}}
; CHECK-NEXT: call            orig_dtor1c1b{{$}}

; CHECK-LABEL: .Lregister_call_dtors.1.associated1c1:
; CHECK:      block
; CHECK-NEXT: i32.const       $push2=, .Lcall_dtors.1.associated1c1{{$}}
; CHECK-NEXT: i32.const       $push1=, 0
; CHECK-NEXT: i32.const       $push0=, __dso_handle
; CHECK-NEXT: i32.call        $push3=, __cxa_atexit, $pop2, $pop1, $pop0{{$}}
; CHECK-NEXT: br_if           0, $pop3
; CHECK-NEXT: return
;      CHECK: end_block
; CHECK-NEXT: unreachable

; CHECK-LABEL: .Lcall_dtors:
; CHECK-NEXT: .functype .Lcall_dtors (i32) -> (){{$}}
; CHECK-NEXT: call            orig_dtor65536{{$}}

; CHECK-LABEL: .Lregister_call_dtors:
; CHECK:      block
; CHECK-NEXT: i32.const       $push2=, .Lcall_dtors{{$}}
; CHECK-NEXT: i32.const       $push1=, 0
; CHECK-NEXT: i32.const       $push0=, __dso_handle
; CHECK-NEXT: i32.call        $push3=, __cxa_atexit, $pop2, $pop1, $pop0{{$}}
; CHECK-NEXT: br_if           0, $pop3
; CHECK-NEXT: return
;      CHECK: end_block
; CHECK-NEXT: unreachable

; CHECK-LABEL: .section .init_array.0,"",@
;      CHECK: .int32  .Lregister_call_dtors.0{{$}}
; CHECK-LABEL: .section .init_array.1,"",@
;      CHECK: .int32  .Lregister_call_dtors.1{{$}}
; CHECK-LABEL: .section .init_array.200,"",@
;      CHECK: .int32  orig_ctor{{$}}
; CHECK-LABEL: .section .init_array,"",@
;      CHECK: .int32  .Lregister_call_dtors{{$}}

; CHECK-LABEL: .weak __dso_handle

; CHECK-LABEL: .functype __cxa_atexit (i32, i32, i32) -> (i32){{$}}

; We shouldn't make use of a .fini_array section.

; FINI-NOT: fini_array

; This function is listed after the null terminator, so it should
; be excluded.

; NULL-NOT: after_the_null
