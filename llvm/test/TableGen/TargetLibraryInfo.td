// RUN: llvm-tblgen -gen-target-library-info -I %p/../../include %s | FileCheck %s

include "llvm/Analysis/TargetLibraryInfoImpl.td"

def cosf : TargetLibCall< "cosf", Flt, [Flt]>;
def sinf : TargetLibCall< "sinf", Flt, [Flt]>;

def fmaxf : TargetLibCall< "fmaxf", Floating, [Same, Same]>;

def printf : TargetLibCall< "printf", Int, [Ptr, Ellip]>;

def cabs : TargetLibCall< "cabs", ? /* Checked manually. */>;

// CHECK:      #ifdef GET_TARGET_LIBRARY_INFO_ENUM
// CHECK-NEXT: #undef GET_TARGET_LIBRARY_INFO_ENUM
// CHECK-NEXT: enum LibFunc : unsigned {
// CHECK-NEXT:   NotLibFunc = 0,
// CHECK-NEXT:   LibFunc_cosf,
// CHECK-NEXT:   LibFunc_sinf,
// CHECK-NEXT:   LibFunc_fmaxf,
// CHECK-NEXT:   LibFunc_printf,
// CHECK-NEXT:   LibFunc_cabs,
// CHECK-NEXT:   NumLibFuncs,
// CHECK-NEXT:   End_LibFunc = NumLibFuncs,
// CHECK-NEXT:   Begin_LibFunc = LibFunc_cosf,
// CHECK-NEXT: };
// CHECK-NEXT: #endif

// CHECK:      #ifdef GET_TARGET_LIBRARY_INFO_STRING_TABLE
// CHECK-NEXT: #undef GET_TARGET_LIBRARY_INFO_STRING_TABLE

// CHECK:      #ifdef __GNUC__
// CHECK-NEXT: #pragma GCC diagnostic push
// CHECK-NEXT: #pragma GCC diagnostic ignored "-Woverlength-strings"
// CHECK-NEXT: #endif
// CHECK-NEXT:  constexpr char StandardNamesStrTableStorage[] =
// CHECK-NEXT:   "\0"
// CHECK-NEXT:   "cosf\0"
// CHECK-NEXT:   "sinf\0"
// CHECK-NEXT:   "fmaxf\0"
// CHECK-NEXT:   "printf\0"
// CHECK-NEXT:   "cabs\0"
// CHECK-NEXT:   ;
// CHECK-NEXT: #ifdef __GNUC__
// CHECK-NEXT: #pragma GCC diagnostic pop
// CHECK-NEXT: #endif

// CHECK:      const llvm::StringTable
// CHECK-NEXT: TargetLibraryInfoImpl::StandardNamesStrTable = StandardNamesStrTableStorage;

// CHECK:      const llvm::StringTable::Offset TargetLibraryInfoImpl::StandardNamesOffsets[6] = {
// CHECK-NEXT:   0, //
// CHECK-NEXT:   1, // cosf
// CHECK-NEXT:   6, // sinf
// CHECK-NEXT:   11, // fmaxf
// CHECK-NEXT:   17, // printf
// CHECK-NEXT:   24, // cabs
// CHECK-NEXT: };
// CHECK-NEXT: const uint8_t TargetLibraryInfoImpl::StandardNamesSizeTable[6] = {
// CHECK-NEXT:   0,
// CHECK-NEXT:   4,
// CHECK-NEXT:   4,
// CHECK-NEXT:   5,
// CHECK-NEXT:   6,
// CHECK-NEXT:   4,
// CHECK-NEXT: };
// CHECK-NEXT: #endif

// CHECK:      #ifdef GET_TARGET_LIBRARY_INFO_IMPL_DECL
// CHECK-NEXT: #undef GET_TARGET_LIBRARY_INFO_IMPL_DECL
// CHECK-NEXT: LLVM_ABI static const llvm::StringTable StandardNamesStrTable;
// CHECK-NEXT: LLVM_ABI static const llvm::StringTable::Offset StandardNamesOffsets[6];
// CHECK-NEXT: LLVM_ABI static const uint8_t StandardNamesSizeTable[6];
// CHECK-NEXT: #endif

// CHECK:      #ifdef GET_TARGET_LIBRARY_INFO_SIGNATURE_TABLE
// CHECK-NEXT: #undef GET_TARGET_LIBRARY_INFO_SIGNATURE_TABLE
// CHECK-NEXT: enum FuncArgTypeID : char {
// CHECK-NEXT:   NoFuncArgType = 0,
// CHECK-NEXT:   Void,
// CHECK-NEXT:   Bool,
// CHECK-NEXT:   Int16,
// CHECK-NEXT:   Int32,
// CHECK-NEXT:   Int,
// CHECK-NEXT:   IntPlus,
// CHECK-NEXT:   Long,
// CHECK-NEXT:   IntX,
// CHECK-NEXT:   Int64,
// CHECK-NEXT:   LLong,
// CHECK-NEXT:   SizeT,
// CHECK-NEXT:   SSizeT,
// CHECK-NEXT:   Flt,
// CHECK-NEXT:   Dbl,
// CHECK-NEXT:   LDbl,
// CHECK-NEXT:   Floating,
// CHECK-NEXT:   Ptr,
// CHECK-NEXT:   Struct,
// CHECK-NEXT:   Ellip,
// CHECK-NEXT:   Same,
// CHECK-NEXT: };
// CHECK-NEXT: static const FuncArgTypeID SignatureTable[] = {
// CHECK-NEXT:   /* 0 */ Int, Ptr, Ellip, NoFuncArgType,
// CHECK-NEXT:   /* 4 */ Flt, Flt, NoFuncArgType,
// CHECK-NEXT:   /* 7 */ Floating, Same, Same, NoFuncArgType,
// CHECK-NEXT:   /* 11 */ Void, NoFuncArgType,
// CHECK-NEXT: };
// CHECK-NEXT: static const uint16_t SignatureOffset[] = {
// CHECK-NEXT:   11, //
// CHECK-NEXT:   4, // cosf
// CHECK-NEXT:   4, // sinf
// CHECK-NEXT:   7, // fmaxf
// CHECK-NEXT:   0, // printf
// CHECK-NEXT:   3, // cabs
// CHECK-NEXT: };
// CHECK-NEXT: #endif
