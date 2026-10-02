// RUN: llvm-tblgen -gen-runtime-libcalls -I %p/../../include %s | FileCheck %s

// Check logical operators over LibcallPredicate work

include "llvm/IR/RuntimeLibcallsImpl.td"

def FUNC0 : RuntimeLibcall;
def FUNC1 : RuntimeLibcall;
def FUNC2 : RuntimeLibcall;
def FUNC3 : RuntimeLibcall;
def FUNC4 : RuntimeLibcall;
def FUNC5 : RuntimeLibcall;

def impl0 : RuntimeLibcallImpl<FUNC0>;
def impl1 : RuntimeLibcallImpl<FUNC1>;
def impl2 : RuntimeLibcallImpl<FUNC2>;
def impl3 : RuntimeLibcallImpl<FUNC3>;
def impl4 : RuntimeLibcallImpl<FUNC4>;
def impl5 : RuntimeLibcallImpl<FUNC5>;

def IsAArch64        : LibcallPredicate<[{TT.isAArch64()}]>;
def IsOSDarwin       : LibcallPredicate<[{TT.isOSDarwin()}]>;
def IsGNUEnv         : LibcallPredicate<[{TT.isGNUEnvironment()}]>;
def IsWindowsArm64EC : LibcallPredicate<[{TT.isWindowsArm64EC()}]>;

// A leaf whose Cond itself contains an operator; make sure this is
// emitted as !(a == b), rather than !a == b.
def IsX86_32 : LibcallPredicate<[{TT.getArch() == Triple::x86}]>;

def isAArch64_ExceptArm64EC
    : RuntimeLibcallAvailability<(all_of IsAArch64, (not IsWindowsArm64EC))>;

def isDarwinOrGNU : RuntimeLibcallAvailability<(any_of IsOSDarwin, IsGNUEnv)>;

def isNotDarwin : RuntimeLibcallAvailability<(not IsOSDarwin)>;

// Nested any_of inside all_of, plus a negated leaf.
def isAArch64DarwinOrGNU : RuntimeLibcallAvailability<
  (all_of IsAArch64, (any_of IsOSDarwin, IsGNUEnv), (not IsWindowsArm64EC))>;

// Not wrapping a nested any_of.
def isAArch64OrNotDarwinGNU : RuntimeLibcallAvailability<
  (any_of IsAArch64, (not (any_of IsOSDarwin, IsGNUEnv)))>;

// Negated leaf whose Cond contains an operator: `!(TT.getArch() == ...)`.
def isNotX86_32 : RuntimeLibcallAvailability<(not IsX86_32)>;

def IsTargetArch : LibcallPredicate<[{isTargetArch()}]>;
def isTargetArch : RuntimeLibcallAvailability<(all_of IsTargetArch)>;

def TheSystemLibrary : SystemRuntimeLibrary<isTargetArch,
  (add LibcallImpls<(add impl0), isAArch64_ExceptArm64EC>,
       LibcallImpls<(add impl1), isDarwinOrGNU>,
       LibcallImpls<(add impl2), isNotDarwin>,
       LibcallImpls<(add impl3), isAArch64DarwinOrGNU>,
       LibcallImpls<(add impl4), isAArch64OrNotDarwinGNU>,
       LibcallImpls<(add impl5), isNotX86_32>)
>;

// Predicate groups emit sorted by predicate def name

// CHECK: if (TT.isAArch64() && (TT.isOSDarwin() || TT.isGNUEnvironment()) && !(TT.isWindowsArm64EC())) {
// CHECK: RTLIB::impl_impl3
// CHECK: if (TT.isAArch64() || !(TT.isOSDarwin() || TT.isGNUEnvironment())) {
// CHECK: RTLIB::impl_impl4
// CHECK: if (TT.isAArch64() && !(TT.isWindowsArm64EC())) {
// CHECK: RTLIB::impl_impl0
// CHECK: if (TT.isOSDarwin() || TT.isGNUEnvironment()) {
// CHECK: RTLIB::impl_impl1
// CHECK: if (!(TT.isOSDarwin())) {
// CHECK: RTLIB::impl_impl2
// CHECK: if (!(TT.getArch() == Triple::x86)) {
// CHECK: RTLIB::impl_impl5
