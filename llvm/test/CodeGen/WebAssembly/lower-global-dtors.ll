; RUN: llc < %s -asm-verbose=false -wasm-keep-registers | FileCheck --check-prefix=CHECK --check-prefix=FINI --check-prefix=NULL %s

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
declare void @orig_dtor1c2a()
declare void @orig_dtor1c2b()
declare void @orig_dtor1c3()
declare void @orig_dtor1d()
declare void @orig_dtor65535()
declare void @orig_dtor65535c0()
declare void @after_the_null()

@associatedc0 = external global i8
@associatedc1 = external global i8
@associatedc2 = global i8 42
@associatedc3 = global i8 84

@llvm.global_ctors = appending global
[1 x { i32, ptr, ptr }]
[
  { i32, ptr, ptr } { i32 200, ptr @orig_ctor, ptr null }
]

@llvm.global_dtors = appending global
[14 x { i32, ptr, ptr }]
[
  { i32, ptr, ptr } { i32 0, ptr @orig_dtor0, ptr null },
  { i32, ptr, ptr } { i32 1, ptr @orig_dtor1a, ptr null },
  { i32, ptr, ptr } { i32 1, ptr @orig_dtor1b, ptr null },
  { i32, ptr, ptr } { i32 1, ptr @orig_dtor1c0, ptr @associatedc0 },
  { i32, ptr, ptr } { i32 1, ptr @orig_dtor1c1a, ptr @associatedc1 },
  { i32, ptr, ptr } { i32 1, ptr @orig_dtor1c1b, ptr @associatedc1 },
  { i32, ptr, ptr } { i32 1, ptr @orig_dtor1c2a, ptr @associatedc2 },
  { i32, ptr, ptr } { i32 1, ptr @orig_dtor1c2b, ptr @associatedc2 },
  { i32, ptr, ptr } { i32 1, ptr @orig_dtor1c3, ptr @associatedc3 },
  { i32, ptr, ptr } { i32 1, ptr @orig_dtor1d, ptr null },
  { i32, ptr, ptr } { i32 65535, ptr @orig_dtor65535c0, ptr @associatedc0 },
  { i32, ptr, ptr } { i32 65535, ptr @orig_dtor65535, ptr null },
  { i32, ptr, ptr } { i32 65535, ptr null, ptr null },
  { i32, ptr, ptr } { i32 65535, ptr @after_the_null, ptr null }
]

; CHECK-LABEL: .functype __cxa_atexit (i32, i32, i32) -> (i32){{$}}

; CHECK-LABEL: .Lcall_dtors.0:
; CHECK-NEXT: .functype .Lcall_dtors.0 (i32) -> (){{$}}
; CHECK-NEXT: call            orig_dtor0{{$}}

; CHECK-LABEL: .Lregister_call_dtors.0:
; CHECK:      block
; CHECK-NEXT: i32.const       $push2=, .Lcall_dtors.0{{$}}
; CHECK-NEXT: i32.const       $push1=, 0
; CHECK-NEXT: i32.const       $push0=, __dso_handle
; CHECK-NEXT: call            $push3=, __cxa_atexit, $pop2, $pop1, $pop0{{$}}
; CHECK-NEXT: i32.eqz         $push4=, $pop3
; CHECK-NEXT: br_if           0, $pop4
; CHECK-NEXT: unreachable
;      CHECK: end_block

; CHECK-LABEL: .Lcall_dtors.1$0:
; CHECK-NEXT: .functype .Lcall_dtors.1$0 (i32) -> (){{$}}
; CHECK-NEXT: call            orig_dtor1b{{$}}
; CHECK-NEXT: call            orig_dtor1a{{$}}

; CHECK-LABEL: .Lregister_call_dtors.1$0:
; CHECK:      block
; CHECK-NEXT: i32.const       $push2=, .Lcall_dtors.1$0{{$}}
; CHECK-NEXT: i32.const       $push1=, 0
; CHECK-NEXT: i32.const       $push0=, __dso_handle
; CHECK-NEXT: call            $push3=, __cxa_atexit, $pop2, $pop1, $pop0{{$}}
; CHECK-NEXT: i32.eqz         $push4=, $pop3
; CHECK-NEXT: br_if           0, $pop4
; CHECK-NEXT: unreachable
;      CHECK: end_block

; CHECK-LABEL: .Lcall_dtors.1$1.associatedc0:
; CHECK-NEXT: .functype .Lcall_dtors.1$1.associatedc0 (i32) -> (){{$}}
; CHECK-NEXT: call            orig_dtor1c0{{$}}

; CHECK-LABEL: .Lregister_call_dtors.1$1.associatedc0:
; CHECK:      block
; CHECK-NEXT: i32.const       $push2=, .Lcall_dtors.1$1.associatedc0{{$}}
; CHECK-NEXT: i32.const       $push1=, 0
; CHECK-NEXT: i32.const       $push0=, __dso_handle
; CHECK-NEXT: call            $push3=, __cxa_atexit, $pop2, $pop1, $pop0{{$}}
; CHECK-NEXT: i32.eqz         $push4=, $pop3
; CHECK-NEXT: br_if           0, $pop4
; CHECK-NEXT: unreachable

; CHECK-LABEL: .Lcall_dtors.1$2.associatedc1:
; CHECK-NEXT: .functype .Lcall_dtors.1$2.associatedc1 (i32) -> (){{$}}
; CHECK-NEXT: call            orig_dtor1c1b{{$}}
; CHECK-NEXT: call            orig_dtor1c1a{{$}}

; CHECK-LABEL: .Lregister_call_dtors.1$2.associatedc1:
; CHECK:      block
; CHECK-NEXT: i32.const       $push2=, .Lcall_dtors.1$2.associatedc1{{$}}
; CHECK-NEXT: i32.const       $push1=, 0
; CHECK-NEXT: i32.const       $push0=, __dso_handle
; CHECK-NEXT: call            $push3=, __cxa_atexit, $pop2, $pop1, $pop0{{$}}
; CHECK-NEXT: i32.eqz         $push4=, $pop3
; CHECK-NEXT: br_if           0, $pop4
; CHECK-NEXT: unreachable

; CHECK-LABEL: .Lcall_dtors.1$3.associatedc2:
; CHECK-NEXT: .functype .Lcall_dtors.1$3.associatedc2 (i32) -> (){{$}}
; CHECK-NEXT: call            orig_dtor1c2b{{$}}
; CHECK-NEXT: call            orig_dtor1c2a{{$}}

; CHECK-LABEL: .Lregister_call_dtors.1$3.associatedc2:
; CHECK:      block
; CHECK-NEXT: i32.const       $push2=, .Lcall_dtors.1$3.associatedc2{{$}}
; CHECK-NEXT: i32.const       $push1=, 0
; CHECK-NEXT: i32.const       $push0=, __dso_handle
; CHECK-NEXT: call            $push3=, __cxa_atexit, $pop2, $pop1, $pop0{{$}}
; CHECK-NEXT: i32.eqz         $push4=, $pop3
; CHECK-NEXT: br_if           0, $pop4
; CHECK-NEXT: unreachable

; CHECK-LABEL: .Lcall_dtors.1$4.associatedc3:
; CHECK-NEXT: .functype .Lcall_dtors.1$4.associatedc3 (i32) -> (){{$}}
; CHECK-NEXT: call            orig_dtor1c3{{$}}

; CHECK-LABEL: .Lregister_call_dtors.1$4.associatedc3:
; CHECK:      block
; CHECK-NEXT: i32.const       $push2=, .Lcall_dtors.1$4.associatedc3{{$}}
; CHECK-NEXT: i32.const       $push1=, 0
; CHECK-NEXT: i32.const       $push0=, __dso_handle
; CHECK-NEXT: call            $push3=, __cxa_atexit, $pop2, $pop1, $pop0{{$}}
; CHECK-NEXT: i32.eqz         $push4=, $pop3
; CHECK-NEXT: br_if           0, $pop4
; CHECK-NEXT: unreachable

; CHECK-LABEL: .Lcall_dtors.1$5:
; CHECK-NEXT: .functype .Lcall_dtors.1$5 (i32) -> (){{$}}
; CHECK-NEXT: call            orig_dtor1d{{$}}

; CHECK-LABEL: .Lregister_call_dtors.1$5:
; CHECK:      block
; CHECK-NEXT: i32.const       $push2=, .Lcall_dtors.1$5{{$}}
; CHECK-NEXT: i32.const       $push1=, 0
; CHECK-NEXT: i32.const       $push0=, __dso_handle
; CHECK-NEXT: call            $push3=, __cxa_atexit, $pop2, $pop1, $pop0{{$}}
; CHECK-NEXT: i32.eqz         $push4=, $pop3
; CHECK-NEXT: br_if           0, $pop4
; CHECK-NEXT: unreachable

; CHECK-LABEL: .Lcall_dtors$0.associatedc0:
; CHECK-NEXT: .functype .Lcall_dtors$0.associatedc0 (i32) -> (){{$}}
; CHECK-NEXT: call            orig_dtor65535c0

; CHECK-LABEL: .Lcall_dtors$1:
; CHECK-NEXT: .functype .Lcall_dtors$1 (i32) -> (){{$}}
; CHECK-NEXT: call            orig_dtor65535{{$}}

; CHECK-LABEL: .Lregister_call_dtors$1:
; CHECK:      block
; CHECK-NEXT: i32.const       $push2=, .Lcall_dtors$1{{$}}
; CHECK-NEXT: i32.const       $push1=, 0
; CHECK-NEXT: i32.const       $push0=, __dso_handle
; CHECK-NEXT: call            $push3=, __cxa_atexit, $pop2, $pop1, $pop0{{$}}
; CHECK-NEXT: i32.eqz         $push4=, $pop3
; CHECK-NEXT: br_if           0, $pop4
; CHECK-NEXT: unreachable

; CHECK-LABEL: .section .init_array.0,"",@
;      CHECK: .int32  .Lregister_call_dtors.0{{$}}
; CHECK-LABEL: .section .init_array.1,"",@
;      CHECK: .int32  .Lregister_call_dtors.1$0{{$}}
; CHECK-NEXT: .int32  .Lregister_call_dtors.1$3.associatedc2{{$}}
; CHECK-NEXT: .int32  .Lregister_call_dtors.1$4.associatedc3{{$}}
; CHECK-NEXT: .int32  .Lregister_call_dtors.1$5{{$}}
; CHECK-LABEL: .section .init_array.200,"",@
;      CHECK: .int32  orig_ctor{{$}}
; CHECK-LABEL: .section .init_array,"",@
;      CHECK: .int32  .Lregister_call_dtors$1{{$}}

; CHECK-LABEL: .weak __dso_handle

; We shouldn't make use of a .fini_array section.

; FINI-NOT: fini_array

; This function is listed after the null terminator, so it should
; be excluded.

; NULL-NOT: after_the_null
