// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

// An isolated LibcallLibrary shares a linker name with a merged library but is
// emitted as its own setAvailableLibFuncs_<defname> containing only its members.
// Its consumer dispatches to that function, still guarding on the real
// LibraryName.

include "llvm/IR/RuntimeLibcallsImpl.td"

def MEMCPY : RuntimeLibcall;
def MEMSET : RuntimeLibcall;
def MALLOC : RuntimeLibcall;

def memcpy : RuntimeLibcallImpl<MEMCPY>;
def memset : RuntimeLibcallImpl<MEMSET>;
def malloc : RuntimeLibcallImpl<MALLOC>;

def IsX86 : LibcallPredicate<[{TT.getArch() == Triple::x86_64}]>;
def isX86 : RuntimeLibcallAvailability<(all_of IsX86)>;
def IsSPIRV : LibcallPredicate<[{TT.isSPIRV()}]>;
def isSPIRV : RuntimeLibcallAvailability<(all_of IsSPIRV)>;

// The generic libc: the full core, always available.
def Libc : LibcallLibrary<"libc", (add memcpy, memset, malloc)>;

// SPIRV supplies only memset/memcpy from libc; isolated so "libc" does not pull
// in the full merged core (malloc).
def SPIRVLibc : LibcallLibrary<"libc", (add memcpy, memset), isSPIRV> {
  let Isolated = true;
}

def X86SystemLibrary : SystemRuntimeLibrary<isX86, ["libc"]>;
def SPIRVSystemLibrary : SystemRuntimeLibrary<isSPIRV, ["libc"]>;

// The merged libc function carries the full core.

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_libc(const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, StringRef ABIName, LongDoubleFormat LongDoubleFormat, CallingConv::ID DefaultCC) {
// CHECK-NEXT:   {
// CHECK-NEXT:     static const RTLIB::LibcallImpl LibraryCalls[] = {
// CHECK-NEXT:         RTLIB::impl_malloc, // malloc
// CHECK-NEXT:         RTLIB::impl_memcpy, // memcpy
// CHECK-NEXT:         RTLIB::impl_memset, // memset
// CHECK-NEXT:     };
// CHECK-EMPTY:
// CHECK-NEXT:     for (const RTLIB::LibcallImpl Impl : LibraryCalls) {
// CHECK-NEXT:       setAvailable(Impl);
// CHECK-NEXT:     }
// CHECK-EMPTY:
// CHECK-NEXT:   }
// CHECK-NEXT: }

// The isolated SPIRV libc is its own function with only memset/memcpy, no malloc.

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_SPIRVLibc(const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, StringRef ABIName, LongDoubleFormat LongDoubleFormat, CallingConv::ID DefaultCC) {
// CHECK-NEXT:   if (TT.isSPIRV()) {
// CHECK-NEXT:     static const RTLIB::LibcallImpl LibraryCalls[] = {
// CHECK-NEXT:         RTLIB::impl_memcpy, // memcpy
// CHECK-NEXT:         RTLIB::impl_memset, // memset
// CHECK-NEXT:     };
// CHECK-EMPTY:
// CHECK-NEXT:     for (const RTLIB::LibcallImpl Impl : LibraryCalls) {
// CHECK-NEXT:       setAvailable(Impl);
// CHECK-NEXT:     }
// CHECK-EMPTY:
// CHECK-NEXT:   }
// CHECK-NEXT: }

// The SPIRV system defaults to "libc", and the libc case selects the
// isolated function by its predicate.

// CHECK-LABEL: ArrayRef<RTLIB::RuntimeLibrary> llvm::RTLIB::RuntimeLibcallsInfo::getDefaultRuntimeLibraries(const llvm::Triple &TT) {
// CHECK-NEXT:   if (TT.isSPIRV()) {
// CHECK-NEXT:     static constexpr RTLIB::RuntimeLibrary Libs[] = {
// CHECK-NEXT:       RTLIB::RuntimeLibrary::libc,
// CHECK-NEXT:     };
// CHECK-NEXT:     return Libs;
// CHECK-NEXT:   }
// CHECK-NEXT:   if (TT.getArch() == Triple::x86_64) {
// CHECK-NEXT:     static constexpr RTLIB::RuntimeLibrary Libs[] = {
// CHECK-NEXT:       RTLIB::RuntimeLibrary::libc,
// CHECK-NEXT:     };
// CHECK-NEXT:     return Libs;
// CHECK-NEXT:   }
// CHECK-NEXT:   return {};
// CHECK-NEXT: }

// CHECK-LABEL: bool llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibraryLibcalls(
// CHECK:   case RTLIB::RuntimeLibrary::libc:
// CHECK-NEXT:     if (TT.isSPIRV()) {
// CHECK-NEXT:       setAvailableLibFuncs_SPIRVLibc(TT, ExceptionModel, FloatABI, ABIName, LongDoubleFormat, DefaultCC);
// CHECK-NEXT:       return true;
// CHECK-NEXT:     }
// CHECK-NEXT:     setAvailableLibFuncs_libc(TT, ExceptionModel, FloatABI, ABIName, LongDoubleFormat, DefaultCC);
// CHECK-NEXT:     return true;
