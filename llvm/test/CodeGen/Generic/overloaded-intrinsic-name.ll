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
define ptr addrspace(1) @test_iAny(ptr addrspace(1) %v) gc "statepoint-example" {
       %tok = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %v)]
       %v-new = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %tok,  i32 0, i32 0)
       ret ptr addrspace(1) %v-new
}

; float
define ptr addrspace(1) @test_fAny(ptr addrspace(1) %v) gc "statepoint-example" {
       %tok = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %v)]
       %v-new = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %tok,  i32 0, i32 0)
       ret ptr addrspace(1) %v-new
}

; array of integers
define ptr addrspace(1) @test_aAny(ptr addrspace(1) %v) gc "statepoint-example" {
       %tok = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %v)]
       %v-new = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %tok,  i32 0, i32 0)
       ret ptr addrspace(1) %v-new
}

; vector of integers
define ptr addrspace(1) @test_vAny(ptr addrspace(1) %v) gc "statepoint-example" {
       %tok = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %v)]
       %v-new = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %tok,  i32 0, i32 0)
       ret ptr addrspace(1) %v-new
}

%struct.test = type { i32, i1 }

; struct
define ptr addrspace(1) @test_struct(ptr addrspace(1) %v) gc "statepoint-example" {
       %tok = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %v)]
       %v-new = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1.tests(token %tok,  i32 0, i32 0)
       ret ptr addrspace(1) %v-new
}

; literal struct with nested literal struct
define ptr addrspace(1) @test_literal_struct(ptr addrspace(1) %v) gc "statepoint-example" {
       %tok = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %v)]
       %v-new = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1.test(token %tok,  i32 0, i32 0)
       ret ptr addrspace(1) %v-new
}
; struct with a horrible name, broken when structs were unprefixed
%i32 = type { i32 }

define ptr addrspace(1) @test_i32_struct(ptr addrspace(1) %v) gc "statepoint-example" {
entry:
      %tok = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %v)]
      %v-new = call ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token %tok,  i32 0, i32 0)
      ret ptr addrspace(1) %v-new
}
; completely broken intrinsic naming due to needing remangling. Just use random naming to test

define ptr addrspace(1) @test_broken_names(ptr addrspace(1) %v) gc "statepoint-example" {
entry:
      %tok = call fastcc token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.deadbeef(i64 0, i32 0, ptr elementtype(i1 ()) @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(ptr addrspace(1) %v)]
; Make sure we do not destroy the calling convention when remangling
; CHECK: fastcc
      %v-new = call ptr addrspace(1) @llvm.experimental.gc.relocate.beefdead(token %tok,  i32 0, i32 0)
      ret ptr addrspace(1) %v-new
}
declare zeroext i1 @return_i1()
declare token @llvm.experimental.gc.statepoint.p0(i64, i32, ptr, i32, i32, ...)
declare ptr addrspace(1) @llvm.experimental.gc.relocate.p1(token, i32, i32)
declare ptr addrspace(1) @llvm.experimental.gc.relocate.p1.tests(token, i32, i32)
declare ptr addrspace(1) @llvm.experimental.gc.relocate.p1.test(token, i32, i32)
declare ptr addrspace(1) @llvm.experimental.gc.relocate.beefdead(token, i32, i32)
declare token @llvm.experimental.gc.statepoint.deadbeef(i64, i32, ptr, i32, i32, ...)
