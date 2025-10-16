// RUN: llvm-tblgen -gen-register-bank -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def MyTarget : Target;

def R0 : Register<"r0">;
def GR : RegisterClass<"MyTarget", [i32], 32, (add R0)>;

def F0 : Register<"f0">;
def FR : RegisterClass<"MyTarget", [f32], 32, (add F0)>;

def V0 : Register<"V0">;
def VR : RegisterClass<"MyTarget", [v4i8, f32], 32, (add V0)>;

def AllFloatR : RegisterClass<"MyTarget", [f32], 32, (add F0, V0)>;
def AnyR : RegisterClass<"MyTarget", [i32, f32, v4i8], 32, (add R0, F0, V0)>;

def GRRegBank : RegisterBank<"GRB", [GR]>;
def FRRegBank : RegisterBank<"FRB", [FR]>;
def VRRegBank : RegisterBank<"VRB", [VR]>;


// CHECK:      #ifdef GET_TARGET_REGBANK_CLASS
// CHECK:        const RegisterBank &getRegBankFromRegClass(const TargetRegisterClass &RC, LLT Ty) const override;

// CHECK:      #ifdef GET_TARGET_REGBANK_IMPL
// CHECK:      const RegisterBank &
// CHECK-NEXT: MyTargetGenRegisterBankInfo::getRegBankFromRegClass(const TargetRegisterClass &RC, LLT) const {
// CHECK-NEXT:   constexpr uint32_t InvalidRegBankID = uint32_t(MyTarget::InvalidRegBankID) & 3;
// CHECK-NEXT:   static const uint32_t RegClass2RegBank[1] = {
// CHECK-NEXT:     (uint32_t(InvalidRegBankID) << 0) |
// CHECK-NEXT:     (uint32_t(InvalidRegBankID) << 2) |
// CHECK-NEXT:     (uint32_t(MyTarget::FRRegBankID) << 4) | // FRRegClassID
// CHECK-NEXT:     (uint32_t(MyTarget::GRRegBankID) << 6) | // GRRegClassID
// CHECK-NEXT:     (uint32_t(MyTarget::VRRegBankID) << 8) // VRRegClassID
// CHECK-NEXT:   };
// CHECK-NEXT:   const unsigned RegClassID = RC.getID();
// CHECK-NEXT:   if (LLVM_LIKELY(RegClassID < 5)) {
// CHECK-NEXT:     unsigned RegBankID = (RegClass2RegBank[RegClassID / 16] >> ((RegClassID % 16) * 2)) & 3;
// CHECK-NEXT:     if (RegBankID != InvalidRegBankID)
// CHECK-NEXT:       return getRegBank(RegBankID);
// CHECK-NEXT:   }
// CHECK-NEXT:   llvm_unreachable(llvm::Twine("Target needs to handle register class ID 0x").concat(llvm::Twine::utohexstr(RegClassID)).str().c_str());
// CHECK-NEXT: }
