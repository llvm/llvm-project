; REQUIRES: x86-registered-target

; Test to ensure that only referenced type ID records are emitted into
; each distributed index file.

; RUN: opt -thinlto-bc -o %t1.o %s
; RUN: opt -thinlto-bc -o %t2.o %p/Inputs/cfi-distributed.ll

; RUN: llvm-lto2 run -thinlto-distributed-indexes %t1.o %t2.o \
; RUN:   -o %t3 \
; RUN:   -r=%t1.o,test,px \
; RUN:   -r=%t1.o,_ZTV1B, \
; RUN:   -r=%t1.o,_ZTV1B,px \
; RUN:   -r=%t1.o,test2, \
; RUN:   -r=%t2.o,test2,px \
; RUN:   -r=%t2.o,_ZTV1B2, \
; RUN:   -r=%t2.o,_ZTV1B2,px

; Since @test calls @test2, the latter should be imported here and the
; first index file should reference both type ids.
; RUN: llvm-dis %t1.o.thinlto.bc -o - | FileCheck %s --check-prefix=INDEX1
; INDEX1: typeid: (name: "_ZTS1A"
; INDEX1: typeid: (name: "_ZTS1A2"

; The second index file, corresponding to @test2, should only contain the
; typeid for _ZTS1A.
; RUN: llvm-dis %t2.o.thinlto.bc -o - | FileCheck %s --check-prefix=INDEX2
; INDEX2-NOT: typeid: (name: "_ZTS1A"
; INDEX2: typeid: (name: "_ZTS1A2"

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.B = type { %struct.A }
%struct.A = type { i32 (...)** }

@_ZTV1B = constant { [3 x i8*] } { [3 x i8*] [i8* undef, i8* undef, i8* undef] }, !type !0

define void @test(i8* %b) {
entry:
  tail call void @test2(i8* %b)
  %0 = bitcast i8* %b to i8**
  %vtable2 = load i8*, i8** %0
  %1 = tail call i1 @llvm.type.test(i8* %vtable2, metadata !"_ZTS1A")
  br i1 %1, label %cont, label %trap

trap:
  tail call void @llvm.trap()
  unreachable

cont:
  ret void
}

declare void @test2(i8*)
declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.trap()

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
