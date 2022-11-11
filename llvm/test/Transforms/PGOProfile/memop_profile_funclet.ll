; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s --check-prefix=GEN
; RUN: opt < %s -passes=pgo-instr-gen,instrprof -S | FileCheck %s --check-prefixes=LOWER

; This test is to verify that PGO runtime library calls get created with the
; appropriate operand bundle funclet information when a memory intrinsic
; being value profiled is called within an exception handler.

; Test case based on this source:
;  #include <memory.h>
;
;  extern void may_throw(int);
;
;  #define MSG "0123456789012345\0"
;  unsigned len = 16;
;  char msg[200];
;
;  void run(int count) {
;    try {
;      may_throw(count);
;    }
;    catch (...) {
;      memcpy(msg, MSG, len);
;      throw;
;    }
;  }

%eh.ThrowInfo = type { i32, i32, i32, i32 }

$"??_C@_0BC@CABPINND@Exception?5caught?$AA?$AA@" = comdat any

@"?len@@3IA" = dso_local global i32 16, align 4
@"?msg@@3PADA" = dso_local global [200 x i8] zeroinitializer, align 16
@"??_C@_0BC@CABPINND@Exception?5caught?$AA?$AA@" = linkonce_odr dso_local unnamed_addr constant [18 x i8] c"0123456789012345\00\00", comdat, align 1

define dso_local void @"?run@@YAXH@Z"(i32 %count) personality ptr @__CxxFrameHandler3 {
entry:
  invoke void @"?may_throw@@YAXH@Z"(i32 %count)
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %tmp = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %tmp1 = catchpad within %tmp [ptr null, i32 64, ptr null]
  %tmp2 = load i32, ptr @"?len@@3IA", align 4
  %conv = zext i32 %tmp2 to i64
  call void @llvm.memcpy.p0.p0.i64(
    ptr @"?msg@@3PADA",
    ptr @"??_C@_0BC@CABPINND@Exception?5caught?$AA?$AA@",
    i64 %conv, i1 false)
  call void @_CxxThrowException(ptr null, ptr null) #3 [ "funclet"(token %tmp1) ]
  unreachable

try.cont:                                         ; preds = %entry
  ret void
}

; GEN: catch:
; GEN: call void @llvm.instrprof.value.profile(
; GEN-SAME: [ "funclet"(token %tmp1) ]

; LOWER: catch:
; LOWER: call void @__llvm_profile_instrument_memop(
; LOWER-SAME: [ "funclet"(token %tmp1) ]

declare dso_local void @"?may_throw@@YAXH@Z"(i32)
declare dso_local i32 @__CxxFrameHandler3(...)

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare dso_local void @_CxxThrowException(ptr, ptr)
