; RUN: llc -verify-machineinstrs <%s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; PR27390 crasher

%typ = type { i32, i32 }

; On release builds, it doesn't crash, spewing nonsense instead.
; To make sure it works, check that rldicl is still alive.
; CHECK: rldicl
; Also, in release, it emits a COPY from a 32-bit register to
; a 64-bit register, which happens to be emitted as cror [!]
; by the confused CodeGen.  Just to be sure, check there isn't one.
; CHECK-NOT: cror
; Function Attrs: uwtable
define signext i32 @_Z8access_pP1Tc(%typ* %p, i8 zeroext %type) {
  %b = getelementptr inbounds %typ, %typ* %p, i64 0, i32 1
  %1 = load i32, i32* %b, align 4
  %2 = ptrtoint i32* %b to i64
  %3 = and i64 %2, -35184372088833
  %4 = inttoptr i64 %3 to i32*
  %_msld = load i32, i32* %4, align 4
  %zzz = add i32 %1,  %_msld
  ret i32 %zzz
}
