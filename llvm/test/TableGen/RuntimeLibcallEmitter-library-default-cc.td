// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

// A member CC naming the DefaultCC sentinel resolves to the target default
// the consuming system library passes to the library function.

include "llvm/IR/RuntimeLibcallsImpl.td"

def SOME_CALL : RuntimeLibcall;
def OTHER_CALL : RuntimeLibcall;

def some_impl : RuntimeLibcallImpl<SOME_CALL>;
def other_impl : RuntimeLibcallImpl<OTHER_CALL>;

def IsFoo : LibcallPredicate<[{TT.getArch() == Triple::arm}]>;
def isFoo : RuntimeLibcallAvailability<(all_of IsFoo)>;

// Predicate using the isAAPCS_ABI member helper.
def IsAAPCS : LibcallPredicate<[{isAAPCS_ABI(TT, ABIName)}]>;
def isAAPCS : RuntimeLibcallAvailability<(all_of IsAAPCS)>;

// CC naming the DefaultCC sentinel.
def FooDefaultOrAAPCS : LibcallCallingConv<
  [{isAAPCS_ABI(TT, ABIName) ? static_cast<CallingConv::ID>(CallingConv::ARM_AAPCS) : static_cast<CallingConv::ID>(DefaultCC)}]
>;

// One member gated on isAAPCS_ABI, one whose CC names DefaultCC.
def FooLib : LibcallLibrary<"foo", (add
  LibcallImpls<(add some_impl), isAAPCS>,
  LibcallsWithCC<(add other_impl), FooDefaultOrAAPCS>)>;

def FooSystem : SystemRuntimeLibrary<isFoo, (add FooLib)> {
  let DefaultLibcallCallingConv
    = LibcallCallingConv<[{FloatABI == FloatABI::Hard ? CallingConv::ARM_AAPCS_VFP : CallingConv::ARM_AAPCS}]>;
}

// The library function takes DefaultCC as a parameter.

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_foo(const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, StringRef ABIName, LongDoubleFormat LongDoubleFormat, CallingConv::ID DefaultCC) {

// CHECK: setLibcallImplCallingConv(Impl, isAAPCS_ABI(TT, ABIName) ? static_cast<CallingConv::ID>(CallingConv::ARM_AAPCS) : static_cast<CallingConv::ID>(DefaultCC));
// CHECK: if (isAAPCS_ABI(TT, ABIName)) {

// The dispatcher computes the whole-table DefaultCC and passes it down.

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setTargetRuntimeLibcallSets(
// CHECK: const CallingConv::ID DefaultCC = FloatABI == FloatABI::Hard ? CallingConv::ARM_AAPCS_VFP : CallingConv::ARM_AAPCS;
// CHECK: if (isLibraryAvailable(RTLIB::RuntimeLibrary::foo))
// CHECK-NEXT: setAvailableLibFuncs_foo(TT, ExceptionModel, FloatABI, ABIName, LongDoubleFormat, DefaultCC);
