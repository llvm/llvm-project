; RUN: not opt -disable-output %s 2>&1 | FileCheck %s

; Token overloads of llvm_any_ty intrinsics used to crash parse-time intrinsic
; remangling with "Unhandled type" in the name mangler (#210641). They must be
; rejected with a normal signature error, like constrained overloads are.

; CHECK: intrinsic return type (overload type 0) expected any manglable type, but got token
declare token @llvm.ssa.copy.token(token)

; CHECK: intrinsic argument 0 type (overload type 0) expected any manglable type, but got token
declare i1 @llvm.is.constant.token(token)

; CHECK: error: input module is broken!
