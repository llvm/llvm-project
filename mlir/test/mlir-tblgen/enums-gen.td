// RUN: mlir-tblgen -gen-enum-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-enum-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

// Test bit enums
def None: I32BitEnumAttrCaseNone<"None">;
def Bit0: I32BitEnumAttrCaseBit<"Bit0", 0, "tagged">;
def Bit1: I32BitEnumAttrCaseBit<"Bit1", 1>;
def Bit2: I32BitEnumAttrCaseBit<"Bit2", 2>;
def Bit3: I32BitEnumAttrCaseBit<"Bit3", 3>;

def MyBitEnum: I32BitEnumAttr<"MyBitEnum", "An example bit enum",
                           [None, Bit0, Bit1, Bit2, Bit3]> {
  let genSpecializedAttr = 0;
}

// DECL-LABEL: enum class MyBitEnum : uint32_t
// DECL: None = 0,
// DECL: Bit0 = 1,
// DECL: Bit1 = 2,
// DECL: Bit2 = 4,
// DECL: Bit3 = 8,
// DECL: }

// DECL: ::llvm::Optional<MyBitEnum> symbolizeMyBitEnum(uint32_t);
// DECL: std::string stringifyMyBitEnum(MyBitEnum);
// DECL: ::llvm::Optional<MyBitEnum> symbolizeMyBitEnum(::llvm::StringRef);

// DEF-LABEL: std::string stringifyMyBitEnum
// DEF: auto val = static_cast<uint32_t>
// DEF: if (val == 0) return "None";
// DEF: if (1u == (1u & val))
// DEF-NEXT: push_back("tagged")
// DEF: if (2u == (2u & val))
// DEF-NEXT: push_back("Bit1")

// DEF-LABEL: ::llvm::Optional<MyBitEnum> symbolizeMyBitEnum(::llvm::StringRef str)
// DEF: if (str == "None") return MyBitEnum::None;
// DEF: .Case("tagged", 1)
// DEF: .Case("Bit1", 2)
