// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

include "llvm/IR/RuntimeLibcallsImpl.td"

def SDIVREM_I8 : RuntimeLibcall;
def UDIVREM_I16 : RuntimeLibcall;
def MALLOC : RuntimeLibcall;
def TARGET_OVERRIDE_CC : RuntimeLibcall;

def __divmodqi4 : RuntimeLibcallImpl<SDIVREM_I8>;
def __udivmodhi4 : RuntimeLibcallImpl<UDIVREM_I16>;

// Test a case where a target wants to set a different calling
// convention on a generic builtin
def __target_override_cc : RuntimeLibcallImpl<TARGET_OVERRIDE_CC>;

def malloc : RuntimeLibcallImpl<MALLOC>;

def IsAVR : LibcallPredicate<[{TT.getArch() == Triple::avr}]>;
def IsOSHurd : LibcallPredicate<[{TT.isOSHurd()}]>;
def IsMSP430 : LibcallPredicate<[{TT.getArch() == Triple::msp430}]>;
def IsFooLeaf : LibcallPredicate<[{ isFoo() }]>;
def IsBarLeaf : LibcallPredicate<[{ isBar() }]>;

def isAVR : RuntimeLibcallAvailability<(all_of IsAVR)>;

def isAVRHurd : RuntimeLibcallAvailability<(all_of IsAVR, IsOSHurd)>;

def isMSP430 : RuntimeLibcallAvailability<(all_of IsMSP430)>;

def Libc : LibcallLibrary<"libc", (add malloc)>;

def AVRCompilerRt : LibcallLibrary<"compiler-rt",
  (add LibcallsWithCC<(add __divmodqi4, __udivmodhi4), AVR_BUILTIN>), isAVR>;

def MSP430CompilerRt : LibcallLibrary<"compiler-rt",
  (add LibcallsWithCC<(add __divmodqi4), AVR_BUILTIN, RuntimeLibcallAvailability<(all_of IsFooLeaf)>>,
       LibcallsWithCC<(add __udivmodhi4), MSP430_BUILTIN, RuntimeLibcallAvailability<(all_of IsBarLeaf)>>),
  isMSP430>;

def AVRLibrary : SystemRuntimeLibrary<isAVR, (add Libc, AVRCompilerRt)>;

// Test with default calling convention
def AVRHurdLibrary : SystemRuntimeLibrary<isAVRHurd, (add Libc, AVRCompilerRt)> {
  let DefaultLibcallCallingConv
    = LibcallCallingConv<[{isFoo() ? CallingConv::Fast : CallingConv::GHC}]>;
}

def MSP430LibraryWithCondCC : SystemRuntimeLibrary<isMSP430,
  (add Libc, MSP430CompilerRt)>;

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_compiler_rt(const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, StringRef ABIName, LongDoubleFormat LongDoubleFormat, CallingConv::ID DefaultCC) {
// CHECK-NEXT:   if (TT.getArch() == Triple::avr) {
// CHECK-NEXT:     static const RTLIB::LibcallImpl LibraryCalls_AlwaysAvailable_AVR_BUILTIN[] = {
// CHECK-NEXT:         RTLIB::impl___divmodqi4, // __divmodqi4
// CHECK-NEXT:         RTLIB::impl___udivmodhi4, // __udivmodhi4
// CHECK-NEXT:     };
// CHECK-EMPTY:
// CHECK-NEXT:     for (const RTLIB::LibcallImpl Impl : LibraryCalls_AlwaysAvailable_AVR_BUILTIN) {
// CHECK-NEXT:       setAvailable(Impl);
// CHECK-NEXT:       setLibcallImplCallingConv(Impl, CallingConv::AVR_BUILTIN);
// CHECK-NEXT:     }
// CHECK-EMPTY:
// CHECK-NEXT:   }
// CHECK-NEXT:   if (TT.getArch() == Triple::msp430) {
// CHECK-NEXT:     if ( isFoo() ) {
// CHECK-NEXT:       static const RTLIB::LibcallImpl LibraryCalls_anonymous_1_AVR_BUILTIN[] = {
// CHECK-NEXT:           RTLIB::impl___divmodqi4, // __divmodqi4
// CHECK-NEXT:       };
// CHECK-EMPTY:
// CHECK-NEXT:       for (const RTLIB::LibcallImpl Impl : LibraryCalls_anonymous_1_AVR_BUILTIN) {
// CHECK-NEXT:         setAvailable(Impl);
// CHECK-NEXT:         setLibcallImplCallingConv(Impl, CallingConv::AVR_BUILTIN);
// CHECK-NEXT:       }
// CHECK-EMPTY:
// CHECK-NEXT:     }
// CHECK-EMPTY:
// CHECK-NEXT:     if ( isBar() ) {
// CHECK-NEXT:       static const RTLIB::LibcallImpl LibraryCalls_anonymous_3_MSP430_BUILTIN[] = {
// CHECK-NEXT:           RTLIB::impl___udivmodhi4, // __udivmodhi4
// CHECK-NEXT:       };
// CHECK-EMPTY:
// CHECK-NEXT:       for (const RTLIB::LibcallImpl Impl : LibraryCalls_anonymous_3_MSP430_BUILTIN) {
// CHECK-NEXT:         setAvailable(Impl);
// CHECK-NEXT:         setLibcallImplCallingConv(Impl, CallingConv::MSP430_BUILTIN);
// CHECK-NEXT:       }
// CHECK-EMPTY:
// CHECK-NEXT:     }
// CHECK-EMPTY:
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setAvailableLibFuncs_libc(const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, StringRef ABIName, LongDoubleFormat LongDoubleFormat, CallingConv::ID DefaultCC) {
// CHECK-NEXT:   {
// CHECK-NEXT:     static const RTLIB::LibcallImpl LibraryCalls[] = {
// CHECK-NEXT:         RTLIB::impl_malloc, // malloc
// CHECK-NEXT:     };
// CHECK-EMPTY:
// CHECK-NEXT:     for (const RTLIB::LibcallImpl Impl : LibraryCalls) {
// CHECK-NEXT:       setAvailable(Impl);
// CHECK-NEXT:     }
// CHECK-EMPTY:
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setTargetRuntimeLibcallSets(const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, StringRef ABIName, LongDoubleFormat LongDoubleFormat) {
// CHECK-EMPTY:
// CHECK-NEXT:   if (TT.getArch() == Triple::avr && TT.isOSHurd()) {
// CHECK-NEXT:     const CallingConv::ID DefaultCC = isFoo() ? CallingConv::Fast : CallingConv::GHC;
// CHECK-NEXT:     for (CallingConv::ID &Entry : LibcallImplCallingConvs) {
// CHECK-NEXT:       Entry = DefaultCC;
// CHECK-NEXT:     }
// CHECK-EMPTY:
// CHECK-NEXT:     if (isLibraryAvailable("libc"))
// CHECK-NEXT:       setAvailableLibFuncs_libc(TT, ExceptionModel, FloatABI, ABIName, LongDoubleFormat, DefaultCC);
// CHECK-NEXT:     if (isLibraryAvailable("compiler-rt"))
// CHECK-NEXT:       setAvailableLibFuncs_compiler_rt(TT, ExceptionModel, FloatABI, ABIName, LongDoubleFormat, DefaultCC);
// CHECK-EMPTY:
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-EMPTY:
// CHECK-NEXT:   if (TT.getArch() == Triple::avr) {
// CHECK-NEXT:     if (isLibraryAvailable("libc"))
// CHECK-NEXT:       setAvailableLibFuncs_libc(TT, ExceptionModel, FloatABI, ABIName, LongDoubleFormat, CallingConv::C);
// CHECK-NEXT:     if (isLibraryAvailable("compiler-rt"))
// CHECK-NEXT:       setAvailableLibFuncs_compiler_rt(TT, ExceptionModel, FloatABI, ABIName, LongDoubleFormat, CallingConv::C);
// CHECK-EMPTY:
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-EMPTY:
// CHECK-NEXT:   if (TT.getArch() == Triple::msp430) {
// CHECK-NEXT:     if (isLibraryAvailable("libc"))
// CHECK-NEXT:       setAvailableLibFuncs_libc(TT, ExceptionModel, FloatABI, ABIName, LongDoubleFormat, CallingConv::C);
// CHECK-NEXT:     if (isLibraryAvailable("compiler-rt"))
// CHECK-NEXT:       setAvailableLibFuncs_compiler_rt(TT, ExceptionModel, FloatABI, ABIName, LongDoubleFormat, CallingConv::C);
// CHECK-EMPTY:
// CHECK-NEXT:     return;
// CHECK-NEXT:   }
// CHECK-NEXT:   LLVM_DEBUG(dbgs() << "no system runtime library applied to target \'" << TT.str() << "\'\n");
