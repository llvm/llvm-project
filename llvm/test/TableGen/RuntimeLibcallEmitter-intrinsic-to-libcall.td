// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

include "llvm/IR/RuntimeLibcallsImpl.td"

def SIN_F32 : RuntimeLibcall;
def SIN_F64 : RuntimeLibcall;
def SIN_F128 : RuntimeLibcall;
def SIN_V4F32 : RuntimeLibcall;
def SIN_V2F64 : RuntimeLibcall;
def COS_F128 : RuntimeLibcall;
def sinf : RuntimeLibcallImpl<SIN_F32, "sinf">;
def : RuntimeLibcallFamily<"SIN", ["sin", "experimental_constrained_sin"],
    ["V4F32", "V2F64"]>;
def : RuntimeLibcallFamily<"COS", ["cos"]>;

// CHECK: #ifdef GET_RUNTIME_LIBCALL_FP_SELECTORS
// CHECK: RTLIB::Libcall llvm::RTLIB::getCOS(EVT VT) {
// CHECK-NEXT:   return getFPLibCall(VT, RTLIB::UNKNOWN_LIBCALL, RTLIB::UNKNOWN_LIBCALL, RTLIB::UNKNOWN_LIBCALL, RTLIB::COS_F128, RTLIB::UNKNOWN_LIBCALL);
// CHECK-NEXT: }
// CHECK: RTLIB::Libcall llvm::RTLIB::getSIN(EVT VT) {
// CHECK-NEXT:   if (VT.isVector()) {
// CHECK-NEXT:     if (!VT.isSimple())
// CHECK-NEXT:       return RTLIB::UNKNOWN_LIBCALL;
// CHECK-NEXT:     switch (VT.getSimpleVT().SimpleTy) {
// CHECK-NEXT:     case MVT::v4f32:
// CHECK-NEXT:       return RTLIB::SIN_V4F32;
// CHECK-NEXT:     case MVT::v2f64:
// CHECK-NEXT:       return RTLIB::SIN_V2F64;
// CHECK-NEXT:     default:
// CHECK-NEXT:       return RTLIB::UNKNOWN_LIBCALL;
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return getFPLibCall(VT, RTLIB::SIN_F32, RTLIB::SIN_F64, RTLIB::UNKNOWN_LIBCALL, RTLIB::SIN_F128, RTLIB::UNKNOWN_LIBCALL);
// CHECK-NEXT: }

// CHECK: #ifdef GET_RUNTIME_LIBCALL_INTRINSIC_TO_LIBCALL
// CHECK: RTLIB::Libcall llvm::RTLIB::RuntimeLibcallsInfo::getLibcallForIntrinsic(Intrinsic::ID ID, Type *Ty) {
// CHECK-NEXT:   switch (ID) {
// CHECK-NEXT:   case Intrinsic::cos:
// CHECK-NEXT:     if (Ty->isFP128Ty())
// CHECK-NEXT:       return RTLIB::COS_F128;
// CHECK-NEXT:     return RTLIB::UNKNOWN_LIBCALL;
// CHECK-NEXT:   case Intrinsic::experimental_constrained_sin:
// CHECK-NEXT:   case Intrinsic::sin:
// CHECK-NEXT:     if (Ty->isFloatTy())
// CHECK-NEXT:       return RTLIB::SIN_F32;
// CHECK-NEXT:     if (Ty->isDoubleTy())
// CHECK-NEXT:       return RTLIB::SIN_F64;
// CHECK-NEXT:     if (Ty->isFP128Ty())
// CHECK-NEXT:       return RTLIB::SIN_F128;
// CHECK-NEXT:     return RTLIB::UNKNOWN_LIBCALL;
// CHECK-NEXT:   default:
// CHECK-NEXT:     return RTLIB::UNKNOWN_LIBCALL;
// CHECK-NEXT:   }
// CHECK-NEXT: }
