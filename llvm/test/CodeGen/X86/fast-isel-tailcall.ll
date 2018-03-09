; RUN: llc < %s -fast-isel -tailcallopt -mtriple=i686-- | FileCheck %s
; CHECK-NOT: add
; PR4154

; On x86, -tailcallopt changes the ABI so the caller shouldn't readjust
; the stack pointer after the call in this code.

define i32 @stub(i8* %t0) nounwind {
entry:
        %t1 = load i32, i32* inttoptr (i32 139708680 to i32*)         ; <i32> [#uses=1]
        %t2 = bitcast i8* %t0 to i32 (i32)*               ; <i32 (i32)*> [#uses=1]
        %t3 = call fastcc i32 %t2(i32 %t1)         ; <i32> [#uses=1]
        ret i32 %t3
}
