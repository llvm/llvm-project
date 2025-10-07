// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

include "llvm/IR/RuntimeLibcallsImpl.td"


def SHL_I32 : RuntimeLibcall;
def SRL_I64 : RuntimeLibcall;

def SQRT_F128 : RuntimeLibcall;
def SQRT_F80 : RuntimeLibcall;
def BZERO : RuntimeLibcall;
def MEMCPY : RuntimeLibcall;
def MEMSET : RuntimeLibcall;
def CALLOC : RuntimeLibcall;

// Test default names.
let IsDefault = true in {
  def __ashlsi3 : RuntimeLibcallImpl<SHL_I32>;
  def __lshrdi3 : RuntimeLibcallImpl<SRL_I64>;

  def sqrtl_f128 : RuntimeLibcallImpl<SQRT_F128, "sqrtl">;
  def sqrtl_f80 : RuntimeLibcallImpl<SQRT_F80, "sqrtl">;
}

// Ignore non-default in initDefaultLibCallNames.
def bzero : RuntimeLibcallImpl<BZERO>;

def ___memset : RuntimeLibcallImpl<MEMSET, "___memset">;
def ___memcpy : RuntimeLibcallImpl<MEMCPY, "___memcpy">;

def calloc  : RuntimeLibcallImpl<CALLOC, "calloc">;

def CompilerRTLibcalls : LibcallImpls<(add __ashlsi3, __lshrdi3)>;
def LibmLibcalls : LibcallImpls<(add sqrtl_f80)>;

def isSimpleArch : RuntimeLibcallPredicate<[{TT.getArch() == Triple::simple}]>;
def isFooArch : RuntimeLibcallPredicate<[{TT.getArch() == Triple::foo}]>;


def isZOS : RuntimeLibcallPredicate<[{TT.getOS() == Triple::zos}]>;
def isPPC : RuntimeLibcallPredicate<[{TT.getArch().isPPC()}]>;
def isPPC64 : RuntimeLibcallPredicate<[{TT.getArch().isPPC64()}]>;


def isFoo : RuntimeLibcallPredicate<[{isFOO()}]>;
def isBarOS : RuntimeLibcallPredicate<[{TT.getOS() == Triple::bar}]>;
def isBuzzArch : RuntimeLibcallPredicate<[{TT.getArch() == Triple::buzz}]>;
def isBlahArch : RuntimeLibcallPredicate<[{TT.getArch() == Triple::blah}]>;

def hasCompilerRT : RuntimeLibcallPredicate<[{TT.hasCompilerRT()}]>;


def SimpleLibrary : SystemRuntimeLibrary<isSimpleArch,
  (add LibmLibcalls, calloc, CompilerRTLibcalls)>;


def LibraryWithConditionalFunc : LibcallImpls<(add sqrtl_f128, bzero, AvailableIf<___memset, isBarOS>)>;

def FooLibrary : SystemRuntimeLibrary<isFooArch, (add LibraryWithConditionalFunc)>;

def BuzzLibrary : SystemRuntimeLibrary<isBuzzArch, (add sqrtl_f80, CompilerRTLibcalls)>;



def LibraryWithConditionalSet : LibcallImpls<(add sqrtl_f128, bzero,
  LibcallImpls<(add CompilerRTLibcalls), hasCompilerRT>)>;



def BlahLibrary : SystemRuntimeLibrary<isBlahArch, (add calloc, LibraryWithConditionalSet, AvailableIf<___memset, isBarOS>)>;



// All entries should be emitted in Libcall enum.
// CHECK: #ifdef GET_RUNTIME_LIBCALL_ENUM
// CHECK-NEXT: namespace llvm {
// CHECK-NEXT: namespace RTLIB {
// CHECK-NEXT: enum Libcall : unsigned short {
// CHECK-NEXT: BZERO = 0,
// CHECK-NEXT: CALLOC = 1,
// CHECK-NEXT: MEMCPY = 2,
// CHECK-NEXT: MEMSET = 3,
// CHECK-NEXT: SHL_I32 = 4,
// CHECK-NEXT: SQRT_F80 = 5,
// CHECK-NEXT: SQRT_F128 = 6,
// CHECK-NEXT: SRL_I64 = 7,
// CHECK-NEXT: UNKNOWN_LIBCALL = 8
// CHECK-NEXT: };
// CHECK-EMPTY:
// CHECK-NEXT:enum LibcallImpl : unsigned short {
// CHECK-NEXT:  Unsupported = 0,
// CHECK-NEXT:  impl____memcpy = 1, // ___memcpy
// CHECK-NEXT:  impl____memset = 2, // ___memset
// CHECK-NEXT:  impl___ashlsi3 = 3, // __ashlsi3
// CHECK-NEXT:  impl___lshrdi3 = 4, // __lshrdi3
// CHECK-NEXT:  bzero = 5, // bzero
// CHECK-NEXT:  calloc = 6, // calloc
// CHECK-NEXT:  sqrtl_f128 = 7, // sqrtl
// CHECK-NEXT:  sqrtl_f80 = 8, // sqrtl
// CHECK-NEXT: };
// CHECK-NEXT: constexpr size_t NumLibcallImpls = 9;
// CHECK-NEXT: } // End namespace RTLIB
// CHECK-NEXT: } // End namespace llvm
// CHECK-NEXT: #endif

// CHECK: #ifdef GET_INIT_RUNTIME_LIBCALL_NAMES
// CHECK-EMPTY:
// CHECK-NEXT: #ifdef __GNUC__
// CHECK-NEXT: #pragma GCC diagnostic push
// CHECK-NEXT: #pragma GCC diagnostic ignored "-Woverlength-strings"
// CHECK-NEXT: #endif
// CHECK-NEXT:  constexpr char RTLIB::RuntimeLibcallsInfo::RuntimeLibcallImplNameTableStorage[] =
// CHECK-NEXT:   "\0"
// CHECK-NEXT:   "___memcpy\0"
// CHECK-NEXT:   "___memset\0"
// CHECK-NEXT:   "__ashlsi3\0"
// CHECK-NEXT:   "__lshrdi3\0"
// CHECK-NEXT:   "bzero\0"
// CHECK-NEXT:   "calloc\0"
// CHECK-NEXT:   "sqrtl\0"
// CHECK-NEXT:   ;
// CHECK-NEXT: #ifdef __GNUC__
// CHECK-NEXT: #pragma GCC diagnostic pop
// CHECK-NEXT: #endif
// CHECK-EMPTY:
// CHECK-NEXT: const llvm::StringTable
// CHECK-NEXT: RTLIB::RuntimeLibcallsInfo::RuntimeLibcallImplNameTable = RuntimeLibcallImplNameTableStorage;
// CHECK-EMPTY:
// CHECK-NEXT: const uint16_t RTLIB::RuntimeLibcallsInfo::RuntimeLibcallNameOffsetTable[] = {
// CHECK-NEXT:   0, //
// CHECK-NEXT:   1, // ___memcpy
// CHECK-NEXT:   11, // ___memset
// CHECK-NEXT:   21, // __ashlsi3
// CHECK-NEXT:   31, // __lshrdi3
// CHECK-NEXT:   41, // bzero
// CHECK-NEXT:   47, // calloc
// CHECK-NEXT:   54, // sqrtl
// CHECK-NEXT:   54, // sqrtl
// CHECK-NEXT: };
// CHECK-EMPTY:
// CHECK-NEXT: const uint8_t RTLIB::RuntimeLibcallsInfo::RuntimeLibcallNameSizeTable[] = {
// CHECK-NEXT:   0,
// CHECK-NEXT:   9,
// CHECK-NEXT:   9,
// CHECK-NEXT:   9,
// CHECK-NEXT:   9,
// CHECK-NEXT:   5,
// CHECK-NEXT:   6,
// CHECK-NEXT:   5,
// CHECK-NEXT:   5,
// CHECK-NEXT: };
// CHECK-EMPTY:
// CHECK-NEXT: const RTLIB::Libcall llvm::RTLIB::RuntimeLibcallsInfo::ImplToLibcall[RTLIB::NumLibcallImpls] = {
// CHECK-NEXT: RTLIB::UNKNOWN_LIBCALL, // RTLIB::Unsupported
// CHECK-NEXT: RTLIB::MEMCPY, // RTLIB::impl____memcpy
// CHECK-NEXT: RTLIB::MEMSET, // RTLIB::impl____memset
// CHECK-NEXT: RTLIB::SHL_I32, // RTLIB::impl___ashlsi3
// CHECK-NEXT: RTLIB::SRL_I64, // RTLIB::impl___lshrdi3
// CHECK-NEXT: RTLIB::BZERO, // RTLIB::impl_bzero
// CHECK-NEXT: RTLIB::CALLOC, // RTLIB::impl_calloc
// CHECK-NEXT: RTLIB::SQRT_F128, // RTLIB::impl_sqrtl_f128
// CHECK-NEXT: RTLIB::SQRT_F80, // RTLIB::impl_sqrtl_f80
// CHECK-NEXT: };

// CHECK: #ifdef GET_LOOKUP_LIBCALL_IMPL_NAME_BODY
// CHECK-NEXT: size_t Size = Name.size();
// CHECK-NEXT: if (Size == 0 || Size > 9)
// CHECK-NEXT:   return enum_seq(RTLIB::Unsupported, RTLIB::Unsupported);
// CHECK-NEXT: return lookupLibcallImplNameImpl(Name);
// CHECK-NEXT: #endif

// CHECK: #ifdef DEFINE_GET_LOOKUP_LIBCALL_IMPL_NAME
// CHECK-NEXT: static inline uint64_t hash(StringRef Str) {
// CHECK-NEXT: return static_cast<uint32_t>(xxh3_64bits(Str));
// CHECK-NEXT: }

// CHECK: iota_range<RTLIB::LibcallImpl> RTLIB::RuntimeLibcallsInfo::lookupLibcallImplNameImpl(StringRef Name) {
// CHECK: static constexpr uint16_t HashTableNameToEnum[16] = {
// CHECK: 2,
// CHECK: 0,
// CHECK: 6,
// CHECK: 0,
// CHECK: };

// CHECK: unsigned Idx = (hash(Name) % 8) * 2;
// CHECK: for (int I = 0; I != 2; ++I) {
// CHECK: return libcallImplNameHit(Entry, StrOffset);

// CHECK: return enum_seq(RTLIB::Unsupported, RTLIB::Unsupported);
// CHECK-NEXT: }

// CHECK: void llvm::RTLIB::RuntimeLibcallsInfo::setTargetRuntimeLibcallSets(const llvm::Triple &TT, ExceptionHandling ExceptionModel, FloatABI::ABIType FloatABI, EABI EABIVersion, StringRef ABIName) {
// CHECK-EMPTY:
// CHECK-NEXT: if (TT.getArch() == Triple::blah) {
// CHECK-NEXT:    setLibcallImpl(RTLIB::BZERO, RTLIB::impl_bzero); // bzero
// CHECK-NEXT:    setLibcallImpl(RTLIB::CALLOC, RTLIB::impl_calloc); // calloc
// CHECK-NEXT:    setLibcallImpl(RTLIB::SQRT_F128, RTLIB::impl_sqrtl_f128); // sqrtl
// CHECK-EMPTY:
// CHECK-NEXT:    if (TT.hasCompilerRT()) {
// CHECK-NEXT:      setLibcallImpl(RTLIB::SHL_I32, RTLIB::impl___ashlsi3); // __ashlsi3
// CHECK-NEXT:      setLibcallImpl(RTLIB::SRL_I64, RTLIB::impl___lshrdi3); // __lshrdi3
// CHECK-EMPTY:
// CHECK-NEXT:    }
// CHECK-EMPTY:
// CHECK-NEXT:    if (TT.getOS() == Triple::bar) {
// CHECK-NEXT:      setLibcallImpl(RTLIB::MEMSET, RTLIB::impl____memset); // ___memset
// CHECK-EMPTY:
// CHECK-NEXT:    }
// CHECK-EMPTY:
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: if (TT.getArch() == Triple::buzz) {
// CHECK-NEXT:    setLibcallImpl(RTLIB::SHL_I32, RTLIB::impl___ashlsi3); // __ashlsi3
// CHECK-NEXT:    setLibcallImpl(RTLIB::SQRT_F80, RTLIB::impl_sqrtl_f80); // sqrtl
// CHECK-NEXT:    setLibcallImpl(RTLIB::SRL_I64, RTLIB::impl___lshrdi3); // __lshrdi3
// CHECK-EMPTY:
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK-NEXT: if (TT.getArch() == Triple::foo) {
// CHECK-NEXT:    setLibcallImpl(RTLIB::BZERO, RTLIB::impl_bzero); // bzero
// CHECK-NEXT:    setLibcallImpl(RTLIB::SQRT_F128, RTLIB::impl_sqrtl_f128); // sqrtl
// CHECK-EMPTY:
// CHECK-NEXT:    if (TT.getOS() == Triple::bar) {
// CHECK-NEXT:      setLibcallImpl(RTLIB::MEMSET, RTLIB::impl____memset); // ___memset
// CHECK-EMPTY:
// CHECK-NEXT:    }
// CHECK-EMPTY:
// CHECK-NEXT:    return;
// CHECK-NEXT:  }
// CHECK-EMPTY:
// CHECK-NEXT: if (TT.getArch() == Triple::simple) {
// CHECK-NEXT:    setLibcallImpl(RTLIB::CALLOC, RTLIB::impl_calloc); // calloc
// CHECK-NEXT:    setLibcallImpl(RTLIB::SHL_I32, RTLIB::impl___ashlsi3); // __ashlsi3
// CHECK-NEXT:    setLibcallImpl(RTLIB::SQRT_F80, RTLIB::impl_sqrtl_f80); // sqrtl
// CHECK-NEXT:    setLibcallImpl(RTLIB::SRL_I64, RTLIB::impl___lshrdi3); // __lshrdi3
// CHECK-EMPTY:
// CHECK-NEXT:   return;
// CHECK-NEXT: }
// CHECK-NEXT:  LLVM_DEBUG(dbgs() << "no system runtime library applied to target \'" << TT.str() << "\'\n");
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK: #endif
