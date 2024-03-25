; RUN: sed -e s/.T1:// %s | not opt -passes=lint -disable-output 2>&1 | FileCheck --check-prefix=CHECK1 %s
; RUN: sed -e s/.T2:// %s | not opt -passes=lint -disable-output 2>&1 | FileCheck --check-prefix=CHECK2 %s

target triple = "x86_64-pc-windows-msvc"

declare void @f()

;T1: declare ptr @llvm.eh.exceptionpointer.p0(i32)
;T1:
;T1: define void @test1() personality ptr @__CxxFrameHandler3 {
;T1:   call ptr @llvm.eh.exceptionpointer.p0(i32 0)
;T1:   ret void
;T1: }
;CHECK1: Intrinsic has incorrect argument type!
;CHECK1-NEXT: ptr @llvm.eh.exceptionpointer.p0

;T2: declare ptr @llvm.eh.exceptionpointer.p0(token)
;T2:
;T2: define void @test2() personality ptr @__CxxFrameHandler3 {
;T2:   call ptr @llvm.eh.exceptionpointer.p0(token undef)
;T2:   ret void
;T2: }
;CHECK2: eh.exceptionpointer argument must be a catchpad
;CHECK2-NEXT:  call ptr @llvm.eh.exceptionpointer.p0(token undef)

declare i32 @__CxxFrameHandler3(...)
