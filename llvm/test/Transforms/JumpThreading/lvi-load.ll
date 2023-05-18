; RUN: opt -S -passes=jump-threading,dce < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.4"

%"struct.llvm::PATypeHolder" = type { ptr }
%"struct.llvm::PointerIntPair<llvm::Use**,2u,llvm::Use::PrevPtrTag,llvm::PointerLikeTypeTraits<llvm::Use**> >" = type { i64 }
%"struct.llvm::Type" = type opaque
%"struct.llvm::Use" = type { ptr, ptr, %"struct.llvm::PointerIntPair<llvm::Use**,2u,llvm::Use::PrevPtrTag,llvm::PointerLikeTypeTraits<llvm::Use**> >" }
%"struct.llvm::Value" = type { ptr, i8, i8, i16, %"struct.llvm::PATypeHolder", ptr, ptr }
%"struct.llvm::ValueName" = type opaque

@_ZZN4llvm4castINS_11InstructionEPNS_5ValueEEENS_10cast_rettyIT_T0_E8ret_typeERKS6_E8__func__ = internal constant [5 x i8] c"cast\00", align 8 ; <ptr> [#uses=1]
@.str = private constant [31 x i8] c"include/llvm/Support/Casting.h\00", align 8 ; <ptr> [#uses=1]
@.str1 = private constant [59 x i8] c"isa<X>(Val) && \22cast<Ty>() argument of incompatible type!\22\00", align 8 ; <ptr> [#uses=1]

; CHECK: Z3fooPN4llvm5ValueE
define zeroext i8 @_Z3fooPN4llvm5ValueE(ptr %V) ssp {
entry:
  %0 = getelementptr inbounds %"struct.llvm::Value", ptr %V, i64 0, i32 1 ; <ptr> [#uses=1]
  %1 = load i8, ptr %0, align 8                       ; <i8> [#uses=2]
  %2 = icmp ugt i8 %1, 20                         ; <i1> [#uses=1]
  br i1 %2, label %bb.i, label %bb2

bb.i:                                             ; preds = %entry
  %toBoolnot.i.i = icmp ult i8 %1, 21             ; <i1> [#uses=1]
  br i1 %toBoolnot.i.i, label %bb6.i.i, label %_ZN4llvm8dyn_castINS_11InstructionEPNS_5ValueEEENS_10cast_rettyIT_T0_E8ret_typeERKS6_.exit

; CHECK-NOT: assert
bb6.i.i:                                          ; preds = %bb.i
  tail call void @__assert_rtn(ptr @_ZZN4llvm4castINS_11InstructionEPNS_5ValueEEENS_10cast_rettyIT_T0_E8ret_typeERKS6_E8__func__, ptr @.str, i32 202, ptr @.str1) noreturn
  unreachable

_ZN4llvm8dyn_castINS_11InstructionEPNS_5ValueEEENS_10cast_rettyIT_T0_E8ret_typeERKS6_.exit: ; preds = %bb.i
; CHECK-NOT: null
  %3 = icmp eq ptr %V, null    ; <i1> [#uses=1]
  br i1 %3, label %bb2, label %bb

bb:                                               ; preds = %_ZN4llvm8dyn_castINS_11InstructionEPNS_5ValueEEENS_10cast_rettyIT_T0_E8ret_typeERKS6_.exit
  tail call void @_ZNK4llvm5Value4dumpEv(ptr %V)
; CHECK: ret
  ret i8 1

bb2:                                              ; preds = %entry, %_ZN4llvm8dyn_castINS_11InstructionEPNS_5ValueEEENS_10cast_rettyIT_T0_E8ret_typeERKS6_.exit
  ret i8 0
}

declare void @__assert_rtn(ptr, ptr, i32, ptr) noreturn

declare void @_ZNK4llvm5Value4dumpEv(ptr)
