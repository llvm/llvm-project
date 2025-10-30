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

def isAVR : RuntimeLibcallPredicate<[{TT.getArch() == Triple::avr}]>;

def isAVRHurd : RuntimeLibcallPredicate<
  [{TT.getArch() == Triple::avr && TT.isOSHurd()}]>;

def AVRLibrary : SystemRuntimeLibrary<isAVR,
  (add malloc, LibcallsWithCC<(add __divmodqi4, __udivmodhi4), AVR_BUILTIN>)
>;

// Test with default calling convention
def AVRHurdLibrary : SystemRuntimeLibrary<isAVRHurd,
  (add malloc, LibcallsWithCC<(add __divmodqi4, __udivmodhi4), AVR_BUILTIN>)> {
  let DefaultLibcallCallingConv
    = LibcallCallingConv<[{isFoo() ? CallingConv::Fast : CallingConv::GHC}]>;
}

def isMSP430 : RuntimeLibcallPredicate<[{TT.getArch() == Triple::msp430}]>;

def MSP430LibraryWithCondCC : SystemRuntimeLibrary<isMSP430,
  (add malloc,
       LibcallsWithCC<(add __divmodqi4), AVR_BUILTIN, RuntimeLibcallPredicate<[{ isFoo() }]>>,
       LibcallsWithCC<(add __udivmodhi4), MSP430_BUILTIN, RuntimeLibcallPredicate<[{ isBar() }]>>)
>;


// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setTargetRuntimeLibcallSets(const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, EABI EABIVersion, StringRef ABIName) {
// CHECK: if (TT.getArch() == Triple::avr && TT.isOSHurd()) {
// CHECK-NEXT:   const CallingConv::ID DefaultCC = isFoo() ? CallingConv::Fast : CallingConv::GHC;
// CHECK-NEXT:   for (CallingConv::ID &Entry : LibcallImplCallingConvs) {
// CHECK-NEXT:     Entry = DefaultCC;
// CHECK-NEXT:   }
// CHECK-EMPTY:
// CHECK-NEXT:   static constexpr LibcallImplBitset SystemAvailableImpls({
// CHECK-NEXT:     0x0000000000001a
// CHECK-NEXT:   });
// CHECK-NEXT:   AvailableLibcallImpls = SystemAvailableImpls;
// CHECK-EMPTY:
// CHECK-NEXT:    static const LibcallImplPair LibraryCalls[] = {
// CHECK-NEXT:        {RTLIB::MALLOC, RTLIB::impl_malloc}, // malloc
// CHECK-NEXT:    };
// CHECK-EMPTY:
// CHECK-NEXT:    for (const auto [Func, Impl] : LibraryCalls) {
// CHECK-NEXT:      setLibcallImpl(Func, Impl);
// CHECK-NEXT:    }
// CHECK-EMPTY:
// CHECK-NEXT:    static const LibcallImplPair LibraryCalls_AlwaysAvailable_AVR_BUILTIN[] = {
// CHECK-NEXT:        {RTLIB::SDIVREM_I8, RTLIB::impl___divmodqi4}, // __divmodqi4
// CHECK-NEXT:        {RTLIB::UDIVREM_I16, RTLIB::impl___udivmodhi4}, // __udivmodhi4
// CHECK-NEXT:    };
// CHECK-EMPTY:
// CHECK-NEXT:    for (const auto [Func, Impl] : LibraryCalls_AlwaysAvailable_AVR_BUILTIN) {
// CHECK-NEXT:      setLibcallImpl(Func, Impl);
// CHECK-NEXT:      setLibcallImplCallingConv(Impl, CallingConv::AVR_BUILTIN);
// CHECK-NEXT:    }
// CHECK-EMPTY:
// CHECK-NEXT:    return;
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT: if (TT.getArch() == Triple::avr) {
// CHECK-NEXT:   static constexpr LibcallImplBitset SystemAvailableImpls({
// CHECK-NEXT:     0x0000000000001a
// CHECK-NEXT:   });
// CHECK-NEXT:   AvailableLibcallImpls = SystemAvailableImpls;
// CHECK-EMPTY:
// CHECK-NEXT:    static const LibcallImplPair LibraryCalls[] = {
// CHECK-NEXT:        {RTLIB::MALLOC, RTLIB::impl_malloc}, // malloc
// CHECK-NEXT:    };
// CHECK-EMPTY:
// CHECK-NEXT:   for (const auto [Func, Impl] : LibraryCalls) {
// CHECK-NEXT:     setLibcallImpl(Func, Impl);
// CHECK-NEXT:   }
// CHECK-EMPTY:
// CHECK-NEXT:   static const LibcallImplPair LibraryCalls_AlwaysAvailable_AVR_BUILTIN[] = {
// CHECK-NEXT:       {RTLIB::SDIVREM_I8, RTLIB::impl___divmodqi4}, // __divmodqi4
// CHECK-NEXT:       {RTLIB::UDIVREM_I16, RTLIB::impl___udivmodhi4}, // __udivmodhi4
// CHECK-NEXT:   };
// CHECK-EMPTY:
// CHECK-NEXT:   for (const auto [Func, Impl] : LibraryCalls_AlwaysAvailable_AVR_BUILTIN) {
// CHECK-NEXT:     setLibcallImpl(Func, Impl);
// CHECK-NEXT:     setLibcallImplCallingConv(Impl, CallingConv::AVR_BUILTIN);
// CHECK-NEXT:   }
// CHECK-EMPTY:
// CHECK-NEXT:   return;
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT:  if (TT.getArch() == Triple::msp430) {
// CHECK-NEXT:    static constexpr LibcallImplBitset SystemAvailableImpls({
// CHECK-NEXT:      0x00000000000010
// CHECK-NEXT:    });
// CHECK-NEXT:    AvailableLibcallImpls = SystemAvailableImpls;
// CHECK-EMPTY:
// CHECK-NEXT:    static const LibcallImplPair LibraryCalls[] = {
// CHECK-NEXT:        {RTLIB::MALLOC, RTLIB::impl_malloc}, // malloc
// CHECK-NEXT:    };
// CHECK-EMPTY:
// CHECK-NEXT:    for (const auto [Func, Impl] : LibraryCalls) {
// CHECK-NEXT:      setLibcallImpl(Func, Impl);
// CHECK-NEXT:    }
// CHECK-EMPTY:
// CHECK-NEXT:    if ( isFoo() ) {
// CHECK-NEXT:      static const LibcallImplPair LibraryCalls_anonymous_3_AVR_BUILTIN[] = {
// CHECK-NEXT:          {RTLIB::SDIVREM_I8, RTLIB::impl___divmodqi4}, // __divmodqi4
// CHECK-NEXT:      };
// CHECK-EMPTY:
// CHECK-NEXT:      for (const auto [Func, Impl] : LibraryCalls_anonymous_3_AVR_BUILTIN) {
// CHECK-NEXT:        setLibcallImpl(Func, Impl);
// CHECK-NEXT:        setLibcallImplCallingConv(Impl, CallingConv::AVR_BUILTIN);
// CHECK-NEXT:      }
// CHECK-EMPTY:
// CHECK-NEXT:    }
// CHECK-EMPTY:
// CHECK-NEXT:    if ( isBar() ) {
// CHECK-NEXT:      static const LibcallImplPair LibraryCalls_anonymous_5_MSP430_BUILTIN[] = {
// CHECK-NEXT:          {RTLIB::UDIVREM_I16, RTLIB::impl___udivmodhi4}, // __udivmodhi4
// CHECK-NEXT:      };
// CHECK-EMPTY:
// CHECK-NEXT:      for (const auto [Func, Impl] : LibraryCalls_anonymous_5_MSP430_BUILTIN) {
// CHECK-NEXT:        setLibcallImpl(Func, Impl);
// CHECK-NEXT:        setLibcallImplCallingConv(Impl, CallingConv::MSP430_BUILTIN);
// CHECK-NEXT:      }
// CHECK-EMPTY:
// CHECK-NEXT:    }
// CHECK-EMPTY:
// CHECK-NEXT:    return;
// CHECK-NEXT:  }
