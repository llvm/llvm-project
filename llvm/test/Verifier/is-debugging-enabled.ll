; RUN: split-file %s %t
; RUN: not opt -passes=verify -disable-output %t/wrong-return.ll 2>&1 | FileCheck %s --check-prefix=RETURN
; RUN: not opt -passes=verify -disable-output %t/wrong-arity.ll 2>&1 | FileCheck %s --check-prefix=ARITY

;--- wrong-return.ll

; RETURN: intrinsic return type expected i1, but got i32
; RETURN-NEXT: declare i32 @llvm.is.debugging.enabled()
declare i32 @llvm.is.debugging.enabled()

;--- wrong-arity.ll

; ARITY: intrinsic has incorrect number of args. Expected 0, but got 1
; ARITY-NEXT: declare i1 @llvm.is.debugging.enabled(i1)
declare i1 @llvm.is.debugging.enabled(i1)
