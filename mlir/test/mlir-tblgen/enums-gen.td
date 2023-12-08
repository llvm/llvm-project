// RUN: mlir-tblgen -gen-enum-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-enum-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

// Test bit enums
def None: I32BitEnumAttrCaseNone<"None", "none">;
def Bit0: I32BitEnumAttrCaseBit<"Bit0", 0, "tagged">;
def Bit1: I32BitEnumAttrCaseBit<"Bit1", 1>;
def Bit2: I32BitEnumAttrCaseBit<"Bit2", 2>;
def Bit3: I32BitEnumAttrCaseBit<"Bit3", 3>;
def BitGroup: I32BitEnumAttrCaseGroup<"BitGroup", [
  Bit0, Bit1
]>;

def MyBitEnum: I32BitEnumAttr<"MyBitEnum", "An example bit enum",
                           [None, Bit0, Bit1, Bit2, Bit3, BitGroup]> {
  let genSpecializedAttr = 0;
}

// DECL-LABEL: enum class MyBitEnum : uint32_t
// DECL: None = 0,
// DECL: Bit0 = 1,
// DECL: Bit1 = 2,
// DECL: Bit2 = 4,
// DECL: Bit3 = 8,
// DECL: }

// DECL: ::std::optional<MyBitEnum> symbolizeMyBitEnum(uint32_t);
// DECL: std::string stringifyMyBitEnum(MyBitEnum);
// DECL: ::std::optional<MyBitEnum> symbolizeMyBitEnum(::llvm::StringRef);

// DECL: struct FieldParser<::MyBitEnum, ::MyBitEnum> {
// DECL:   template <typename ParserT>
// DECL:   static FailureOr<::MyBitEnum> parse(ParserT &parser) {
// DECL:     // Parse the keyword/string containing the enum.
// DECL:     std::string enumKeyword;
// DECL:     auto loc = parser.getCurrentLocation();
// DECL:     if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
// DECL:       return parser.emitError(loc, "expected keyword for An example bit enum");
// DECL:     // Symbolize the keyword.
// DECL:     if (::std::optional<::MyBitEnum> attr = ::symbolizeEnum<::MyBitEnum>(enumKeyword))
// DECL:       return *attr;
// DECL:     return parser.emitError(loc, "invalid An example bit enum specification: ") << enumKeyword;
// DECL:   }

// DECL: inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::MyBitEnum value) {
// DECL:   auto valueStr = stringifyEnum(value);
// DECL:   switch (value) {
// DECL:   case ::MyBitEnum::BitGroup:
// DECL:     return p << valueStr;
// DECL:   default:
// DECL:     break;
// DECL:   }
// DECL:   auto underlyingValue = static_cast<std::make_unsigned_t<::MyBitEnum>>(value);
// DECL:   if (underlyingValue && !llvm::has_single_bit(underlyingValue))
// DECL:     return p << '"' << valueStr << '"';
// DECL:   return p << valueStr;

// DEF-LABEL: std::string stringifyMyBitEnum
// DEF: auto val = static_cast<uint32_t>
// DEF: if (val == 0) return "none";
// DEF: if (1u == (1u & val))
// DEF-NEXT: push_back("tagged")
// DEF: if (2u == (2u & val))
// DEF-NEXT: push_back("Bit1")

// DEF-LABEL: ::std::optional<MyBitEnum> symbolizeMyBitEnum(::llvm::StringRef str)
// DEF: if (str == "none") return MyBitEnum::None;
// DEF: .Case("tagged", 1)
// DEF: .Case("Bit1", 2)

// Test enum printer generation for non non-keyword enums.

def NonKeywordBit: I32BitEnumAttrCaseBit<"Bit0", 0, "tag-ged">;
def MyMixedNonKeywordBitEnum: I32BitEnumAttr<"MyMixedNonKeywordBitEnum", "An example bit enum", [
    NonKeywordBit,
    Bit1
  ]> {
  let genSpecializedAttr = 0;
}

def MyNonKeywordBitEnum: I32BitEnumAttr<"MyNonKeywordBitEnum", "An example bit enum", [
    NonKeywordBit
  ]> {
  let genSpecializedAttr = 0;
}

// DECL: inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::MyMixedNonKeywordBitEnum value) {
// DECL: auto valueStr = stringifyEnum(value);
// DECL:   switch (value) {
// DECL:   case ::MyMixedNonKeywordBitEnum::Bit1:
// DECL:     break;
// DECL:   default:
// DECL:     return p << '"' << valueStr << '"';
// DECL:   }
// DECL:   return p << valueStr;
// DECL: }

// DECL: inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::MyNonKeywordBitEnum value) {
// DECL:   auto valueStr = stringifyEnum(value);
// DECL:   return p << '"' << valueStr << '"';
