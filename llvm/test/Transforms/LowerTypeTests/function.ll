; RUN: opt -S -lowertypetests -mtriple=i686-unknown-linux-gnu < %s | FileCheck --check-prefixes=X86,X86-LINUX,NATIVE %s
; RUN: opt -S -lowertypetests -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck --check-prefixes=X86,X86-LINUX,NATIVE %s
; RUN: opt -S -lowertypetests -mtriple=i686-pc-win32 < %s | FileCheck --check-prefixes=X86,X86-WIN32,NATIVE %s
; RUN: opt -S -lowertypetests -mtriple=x86_64-pc-win32 < %s | FileCheck --check-prefixes=X86,X86-WIN32,NATIVE %s
; RUN: opt -S -lowertypetests -mtriple=arm-unknown-linux-gnu < %s | FileCheck --check-prefixes=ARM,NATIVE %s
; RUN: opt -S -lowertypetests -mtriple=thumb-unknown-linux-gnu < %s | FileCheck --check-prefixes=THUMB,NATIVE %s
; RUN: opt -S -lowertypetests -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck --check-prefixes=ARM,NATIVE %s
; RUN: opt -S -lowertypetests -mtriple=riscv32-unknown-linux-gnu < %s | FileCheck --check-prefixes=RISCV,NATIVE %s
; RUN: opt -S -lowertypetests -mtriple=riscv64-unknown-linux-gnu < %s | FileCheck --check-prefixes=RISCV,NATIVE %s
; RUN: opt -S -lowertypetests -mtriple=wasm32-unknown-unknown < %s | FileCheck --check-prefix=WASM32 %s

; Tests that we correctly handle bitsets containing 2 or more functions.

target datalayout = "e-p:64:64"


; NATIVE: @0 = private unnamed_addr constant [2 x void (...)*] [void (...)* bitcast (void ()* @f to void (...)*), void (...)* bitcast (void ()* @g to void (...)*)], align 16
@0 = private unnamed_addr constant [2 x void (...)*] [void (...)* bitcast (void ()* @f to void (...)*), void (...)* bitcast (void ()* @g to void (...)*)], align 16

; NATIVE: private constant [0 x i8] zeroinitializer
; WASM32: private constant [0 x i8] zeroinitializer

; NATIVE: @f = alias void (), void ()* @[[JT:.*]]

; X86: @g = internal alias void (), bitcast ([8 x i8]* getelementptr inbounds ([2 x [8 x i8]], [2 x [8 x i8]]* bitcast (void ()* @[[JT]] to [2 x [8 x i8]]*), i64 0, i64 1) to void ()*)
; ARM: @g = internal alias void (), bitcast ([4 x i8]* getelementptr inbounds ([2 x [4 x i8]], [2 x [4 x i8]]* bitcast (void ()* @[[JT]] to [2 x [4 x i8]]*), i64 0, i64 1) to void ()*)
; THUMB: @g = internal alias void (), bitcast ([4 x i8]* getelementptr inbounds ([2 x [4 x i8]], [2 x [4 x i8]]* bitcast (void ()* @[[JT]] to [2 x [4 x i8]]*), i64 0, i64 1) to void ()*)
; RISCV: @g = internal alias void (), bitcast ([8 x i8]* getelementptr inbounds ([2 x [8 x i8]], [2 x [8 x i8]]* bitcast (void ()* @[[JT]] to [2 x [8 x i8]]*), i64 0, i64 1) to void ()*)

; NATIVE: define hidden void @f.cfi()
; WASM32: define void @f() !type !{{[0-9]+}} !wasm.index ![[I0:[0-9]+]]
define void @f() !type !0 {
  ret void
}

; NATIVE: define internal void @g.cfi()
; WASM32: define internal void @g() !type !{{[0-9]+}} !wasm.index ![[I1:[0-9]+]]
define internal void @g() !type !0 {
  ret void
}

!0 = !{i32 0, !"typeid1"}

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone

define i1 @foo(i8* %p) {
  ; NATIVE: sub i64 {{.*}}, ptrtoint (void ()* @[[JT]] to i64)
  ; WASM32: sub i64 {{.*}}, ptrtoint (i8* getelementptr (i8, i8* null, i64 1) to i64)
  ; WASM32: icmp ule i64 {{.*}}, 1
  %x = call i1 @llvm.type.test(i8* %p, metadata !"typeid1")
  ret i1 %x
}

; X86-LINUX:   define private void @[[JT]]() #[[ATTR:.*]] align 8 {
; X86-WIN32:   define private void @[[JT]]() #[[ATTR:.*]] align 8 {
; ARM:   define private void @[[JT]]() #[[ATTR:.*]] align 4 {
; THUMB: define private void @[[JT]]() #[[ATTR:.*]] align 4 {
; RISCV: define private void @[[JT]]() #[[ATTR:.*]] align 8 {

; X86:      jmp ${0:c}@plt
; X86-SAME: int3
; X86-SAME: int3
; X86-SAME: int3
; X86-SAME: jmp ${1:c}@plt
; X86-SAME: int3
; X86-SAME: int3
; X86-SAME: int3

; ARM:      b $0
; ARM-SAME: b $1

; THUMB:      b.w $0
; THUMB-SAME: b.w $1

; RISCV:      tail $0@plt
; RISCV-SAME: tail $1@plt

; NATIVE-SAME: "s,s"(void ()* @f.cfi, void ()* @g.cfi)

; X86-LINUX: attributes #[[ATTR]] = { naked nounwind }
; X86-WIN32: attributes #[[ATTR]] = { nounwind }
; ARM: attributes #[[ATTR]] = { naked nounwind
; THUMB: attributes #[[ATTR]] = { naked nounwind "target-cpu"="cortex-a8" "target-features"="+thumb-mode" }
; RISCV: attributes #[[ATTR]] = { naked nounwind "target-features"="-c,-relax" }

; WASM32: ![[I0]] = !{i64 1}
; WASM32: ![[I1]] = !{i64 2}
