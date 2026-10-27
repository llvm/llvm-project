// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

include "llvm/IR/RuntimeLibcallsImpl.td"

foreach S = ["V2F64", "NXV2F64"] in {
  defvar VT = !cast<FuncArgType>(S);
  defm ACOS : VectorMathLibcall<S>;
  defm POW : VectorMathLibcall<S, VT, [VT, VT]>;
}

def _ZGVsMxv_acos : RuntimeLibcallImpl<ACOS_MASKED_NXV2F64>;
def _ZGVnN2vv_pow : RuntimeLibcallImpl<POW_V2F64>;
def _ZGVsMxvv_pow : RuntimeLibcallImpl<POW_MASKED_NXV2F64>;
def Sleef_powdx_u10rvvm2 : RuntimeLibcallImpl<POW_NXV2F64>;

def Lib : LibcallLibrary<"lib", (add _ZGVsMxv_acos, _ZGVnN2vv_pow, _ZGVsMxvv_pow,
                                     Sleef_powdx_u10rvvm2)>;

def TheSystemLibrary : SystemRuntimeLibrary<AlwaysAvailable, ["lib"]>;

// CHECK:      enum Libcall : unsigned short {
// CHECK-NEXT:   ACOS_MASKED_NXV2F64 = 0,
// CHECK-NEXT:   ACOS_NXV2F64 = 1,
// CHECK-NEXT:   ACOS_V2F64 = 2,
// CHECK-NEXT:   POW_MASKED_NXV2F64 = 3,
// CHECK-NEXT:   POW_NXV2F64 = 4,
// CHECK-NEXT:   POW_V2F64 = 5,

// CHECK:      const RTLIB::Libcall llvm::RTLIB::RuntimeLibcallsInfo::ImplToLibcall[RTLIB::NumLibcallImpls] = {
// CHECK-NEXT:   RTLIB::UNKNOWN_LIBCALL, // RTLIB::Unsupported
// CHECK-NEXT:   RTLIB::POW_NXV2F64, // RTLIB::impl_Sleef_powdx_u10rvvm2
// CHECK-NEXT:   RTLIB::POW_V2F64, // RTLIB::impl__ZGVnN2vv_pow
// CHECK-NEXT:   RTLIB::ACOS_MASKED_NXV2F64, // RTLIB::impl__ZGVsMxv_acos
// CHECK-NEXT:   RTLIB::POW_MASKED_NXV2F64, // RTLIB::impl__ZGVsMxvv_pow
// CHECK-NEXT: };

// CHECK:      const FuncArgTypeID RTLIB::RuntimeLibcallsInfo::SignatureTable[] = {
// CHECK-NEXT:   /* 0 */ NXV2F64, NXV2F64, NXV2F64, NoFuncArgType,
// CHECK-NEXT:   /* 4 */ NXV2F64, NXV2F64, NXV2F64, NXV2I1, NoFuncArgType,
// CHECK-NEXT:   /* 9 */ V2F64, V2F64, V2F64, NoFuncArgType,
// CHECK-NEXT: };

// CHECK:      const uint16_t RTLIB::RuntimeLibcallsInfo::SignatureOffset[] = {
// CHECK-NEXT:   3, // ACOS_MASKED_NXV2F64
// CHECK-NEXT:   3, // ACOS_NXV2F64
// CHECK-NEXT:   3, // ACOS_V2F64
// CHECK-NEXT:   4, // POW_MASKED_NXV2F64
// CHECK-NEXT:   0, // POW_NXV2F64
// CHECK-NEXT:   9, // POW_V2F64
// CHECK-NEXT: };

// CHECK:      #ifdef GET_RUNTIME_LIBCALL_MASKED_VECTOR_LIBCALL_MAPPING
// CHECK-NEXT: #undef GET_RUNTIME_LIBCALL_MASKED_VECTOR_LIBCALL_MAPPING
// CHECK-EMPTY:
// CHECK-NEXT: RTLIB::Libcall RTLIB::RuntimeLibcallsInfo::getUnmaskedVectorLibcall(RTLIB::Libcall LC) {
// CHECK-NEXT:   switch (LC) {
// CHECK-NEXT:   case RTLIB::ACOS_MASKED_NXV2F64:
// CHECK-NEXT:     return RTLIB::ACOS_NXV2F64;
// CHECK-NEXT:   case RTLIB::POW_MASKED_NXV2F64:
// CHECK-NEXT:     return RTLIB::POW_NXV2F64;
// CHECK-NEXT:   default:
// CHECK-NEXT:     return RTLIB::UNKNOWN_LIBCALL;
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: RTLIB::Libcall RTLIB::RuntimeLibcallsInfo::getMaskedVectorLibcall(RTLIB::Libcall LC) {
// CHECK-NEXT:   switch (LC) {
// CHECK-NEXT:   case RTLIB::ACOS_NXV2F64:
// CHECK-NEXT:     return RTLIB::ACOS_MASKED_NXV2F64;
// CHECK-NEXT:   case RTLIB::POW_NXV2F64:
// CHECK-NEXT:     return RTLIB::POW_MASKED_NXV2F64;
// CHECK-NEXT:   default:
// CHECK-NEXT:     return RTLIB::UNKNOWN_LIBCALL;
// CHECK-NEXT:   }
// CHECK-NEXT: }
