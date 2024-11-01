; RUN: opt -passes=verify -S < %s | FileCheck %s

; Tests the name mangling performed by the codepath following
; getMangledTypeStr(). Only tests that code with the various manglings
; run fine: doesn't actually test the mangling with the type of the
; arguments. Meant to serve as an example-document on how the user
; should do name manglings.

; Exercise the most general case, llvm_anyptr_type, using gc.relocate
; and gc.statepoint. Note that it has nothing to do with gc.*
; functions specifically: any function that accepts llvm_anyptr_type
; will serve the purpose.

; function and integer
define i32 addrspace(1)* @test_iAny(i32 addrspace(1)* %v) gc "statepoint-example" {
       %tok = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i32 addrspace(1)* %v)]
       %v-new = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %tok,  i32 0, i32 0)
       ret i32 addrspace(1)* %v-new
}

; float
define float addrspace(1)* @test_fAny(float addrspace(1)* %v) gc "statepoint-example" {
       %tok = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(float addrspace(1)* %v)]
       %v-new = call float addrspace(1)* @llvm.experimental.gc.relocate.p1f32(token %tok,  i32 0, i32 0)
       ret float addrspace(1)* %v-new
}

; array of integers
define [3 x i32] addrspace(1)* @test_aAny([3 x i32] addrspace(1)* %v) gc "statepoint-example" {
       %tok = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"([3 x i32] addrspace(1)* %v)]
       %v-new = call [3 x i32] addrspace(1)* @llvm.experimental.gc.relocate.p1a3i32(token %tok,  i32 0, i32 0)
       ret [3 x i32] addrspace(1)* %v-new
}

; vector of integers
define <3 x i32> addrspace(1)* @test_vAny(<3 x i32> addrspace(1)* %v) gc "statepoint-example" {
       %tok = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(<3 x i32> addrspace(1)* %v)]
       %v-new = call <3 x i32> addrspace(1)* @llvm.experimental.gc.relocate.p1v3i32(token %tok,  i32 0, i32 0)
       ret <3 x i32> addrspace(1)* %v-new
}

%struct.test = type { i32, i1 }

; struct
define %struct.test addrspace(1)* @test_struct(%struct.test addrspace(1)* %v) gc "statepoint-example" {
       %tok = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(%struct.test addrspace(1)* %v)]
       %v-new = call %struct.test addrspace(1)* @llvm.experimental.gc.relocate.p1s_struct.tests(token %tok,  i32 0, i32 0)
       ret %struct.test addrspace(1)* %v-new
}

; literal struct with nested literal struct
define {i64, i64, {i64} } addrspace(1)* @test_literal_struct({i64, i64, {i64}} addrspace(1)* %v) gc "statepoint-example" {
       %tok = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"({i64, i64, {i64}} addrspace(1)* %v)]
       %v-new = call {i64, i64, {i64}} addrspace(1)* @llvm.experimental.gc.relocate.p1sl_i64i64sl_i64ss.test(token %tok,  i32 0, i32 0)
       ret {i64, i64, {i64}} addrspace(1)* %v-new
}
; struct with a horrible name, broken when structs were unprefixed
%i32 = type { i32 }

define %i32 addrspace(1)* @test_i32_struct(%i32 addrspace(1)* %v) gc "statepoint-example" {
entry:
      %tok = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(%i32 addrspace(1)* %v)]
      %v-new = call %i32 addrspace(1)* @llvm.experimental.gc.relocate.p1s_i32s(token %tok,  i32 0, i32 0)
      ret %i32 addrspace(1)* %v-new
}
; completely broken intrinsic naming due to needing remangling. Just use random naming to test

define %i32 addrspace(1)* @test_broken_names(%i32 addrspace(1)* %v) gc "statepoint-example" {
entry:
      %tok = call fastcc token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.deadbeef(i64 0, i32 0, i1 ()* elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(%i32 addrspace(1)* %v)]
; Make sure we do not destroy the calling convention when remangling
; CHECK: fastcc
      %v-new = call %i32 addrspace(1)* @llvm.experimental.gc.relocate.beefdead(token %tok,  i32 0, i32 0)
      ret %i32 addrspace(1)* %v-new
}
declare zeroext i1 @return_i1()
declare token @llvm.experimental.gc.statepoint.p0f_i1f(i64, i32, i1 ()*, i32, i32, ...)
declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token, i32, i32)
declare float addrspace(1)* @llvm.experimental.gc.relocate.p1f32(token, i32, i32)
declare [3 x i32] addrspace(1)* @llvm.experimental.gc.relocate.p1a3i32(token, i32, i32)
declare <3 x i32> addrspace(1)* @llvm.experimental.gc.relocate.p1v3i32(token, i32, i32)
declare %struct.test addrspace(1)* @llvm.experimental.gc.relocate.p1s_struct.tests(token, i32, i32)
declare {i64, i64, {i64}} addrspace(1)* @llvm.experimental.gc.relocate.p1sl_i64i64sl_i64ss.test(token, i32, i32)
declare %i32 addrspace(1)* @llvm.experimental.gc.relocate.p1s_i32s(token, i32, i32)
declare %i32 addrspace(1)* @llvm.experimental.gc.relocate.beefdead(token, i32, i32)
declare token @llvm.experimental.gc.statepoint.deadbeef(i64, i32, i1 ()*, i32, i32, ...)
