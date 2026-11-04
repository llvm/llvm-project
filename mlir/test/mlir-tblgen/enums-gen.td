// RUN: mlir-tblgen -gen-enum-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-enum-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

// Test bit enums
def None: I32BitEnumCaseNone<"None", "none">;
def Bit0: I32BitEnumCaseBit<"Bit0", 0, "tagged">;
def Bit1: I32BitEnumCaseBit<"Bit1", 1>;
def Bit2: I32BitEnumCaseBit<"Bit2", 2>;
def Bit3: I32BitEnumAttrCaseBit<"Bit3", 3>;
def BitGroup: BitEnumCaseGroup<"BitGroup", [
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
// DECL-NEXT:   static FailureOr<::MyBitEnum> parse(AsmParser &parser);
// DECL-NEXT: };

// DECL: struct FieldParser<std::optional<::MyBitEnum>, std::optional<::MyBitEnum>> {
// DECL:   static constexpr bool isKeyValueCompositional = false;
// DECL:   static FailureOr<std::optional<::MyBitEnum>> parse(AsmParser &parser);
// DECL-NEXT: };

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

// DECL: struct FieldParser<::MyCommaSeparatedBitEnum, ::MyCommaSeparatedBitEnum> {
// DECL:   static constexpr bool isKeyValueCompositional = false;
// DECL: struct FieldParser<std::optional<::MyCommaSeparatedBitEnum>, std::optional<::MyCommaSeparatedBitEnum>> {
// DECL:   static constexpr bool isKeyValueCompositional = false;

// DECL: enum class MyI8Enum : uint8_t {
// DECL:   a = 254,
// DECL:   b = 255,
// DECL: };

// DECL: ::std::optional<MyI8Enum> symbolizeMyI8Enum(uint8_t);

// DECL: inline constexpr unsigned getMaxEnumValForMyI8Enum() {
// DECL:   return 255;
// DECL: }

// DECL: enum class MyI16Enum : uint16_t {
// DECL:   a = 65534,
// DECL:   b = 65535,
// DECL: };

// DECL: ::std::optional<MyI16Enum> symbolizeMyI16Enum(uint16_t);

// DECL: inline constexpr unsigned getMaxEnumValForMyI16Enum() {
// DECL:   return 65535;
// DECL: }

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

// DEF-LABEL: FailureOr<::MyBitEnum> FieldParser<::MyBitEnum, ::MyBitEnum>::parse(AsmParser &parser) {
// DEF:   // Parse the keyword/string containing the enum.
// DEF:   std::string enumKeyword;
// DEF:   auto loc = parser.getCurrentLocation();
// DEF:   if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
// DEF:     return parser.emitError(loc, "expected keyword for An example bit enum");
// DEF:   // Symbolize the keyword.
// DEF:   if (::std::optional<::MyBitEnum> attr = ::symbolizeEnum<::MyBitEnum>(enumKeyword))
// DEF:     return *attr;
// DEF:   return parser.emitError(loc, "expected one of [none, tagged, Bit1, Bit2, Bit3, BitGroup] for An example bit enum, got: ") << enumKeyword;
// DEF: }

// DEF-LABEL: FailureOr<std::optional<::MyBitEnum>>
// DEF-NEXT: FieldParser<std::optional<::MyBitEnum>, std::optional<::MyBitEnum>>::parse(AsmParser &parser) {
// DEF:   if (failed(parser.parseOptionalKeywordOrString(&enumKeyword, {"none", "tagged", "Bit1", "Bit2", "Bit3", "BitGroup"})))
// DEF-NEXT:     return std::optional<::MyBitEnum>{};

// Test enum printer generation for non non-keyword enums.

def NonKeywordBit: I32BitEnumCaseBit<"Bit0", 0, "tag-ged">;
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

def MyNonQuotedPrintBitEnum
  : I32BitEnum<"MyNonQuotedPrintBitEnum", "Example new-style bit enum",
    [None, Bit0, Bit1, Bit2, Bit3, BitGroup]>;

// DECL: struct FieldParser<::MyNonQuotedPrintBitEnum, ::MyNonQuotedPrintBitEnum> {
// DECL:   static constexpr bool isKeyValueCompositional = true;
// DECL:   static FailureOr<::MyNonQuotedPrintBitEnum> parse(AsmParser &parser);
// DECL-NEXT: };

// DEF-LABEL: FailureOr<::MyNonQuotedPrintBitEnum> FieldParser<::MyNonQuotedPrintBitEnum, ::MyNonQuotedPrintBitEnum>::parse(AsmParser &parser) {
// DEF:   ::MyNonQuotedPrintBitEnum flags = {};
// DEF:   do {
// DEF:     // Parse the keyword containing a part of the enum.
// DEF:     ::llvm::StringRef enumKeyword;
// DEF:     auto loc = parser.getCurrentLocation();
// DEF:     if (failed(parser.parseOptionalKeyword(&enumKeyword))) {
// DEF:       return parser.emitError(loc, "expected keyword for Example new-style bit enum");
// DEF:     }
// DEF:     // Symbolize the keyword.
// DEF:     if (::std::optional<::MyNonQuotedPrintBitEnum> flag = ::symbolizeEnum<::MyNonQuotedPrintBitEnum>(enumKeyword)) {
// DEF:       flags = flags | *flag;
// DEF:     } else {
// DEF:       return parser.emitError(loc, "expected one of [none, tagged, Bit1, Bit2, Bit3, BitGroup] for Example new-style bit enum, got: ") << enumKeyword;
// DEF:     }
// DEF:   } while (::mlir::succeeded(parser.parseOptionalVerticalBar()));
// DEF:   return flags;
// DEF: }

def MyCommaSeparatedBitEnum
  : I32BitEnum<"MyCommaSeparatedBitEnum", "Comma-separated bit enum",
    [None, Bit0, Bit1]> {
  let separator = ", ";
}

// DECL: inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::MyNonQuotedPrintBitEnum value) {
// DECL:   auto valueStr = stringifyEnum(value);
// DECL-NEXT:   return p << valueStr;

def MyI8Enum : I8Enum<"MyI8Enum", "Example I8 enum", [
    I8EnumCase<"a", 254>,
    I8EnumCase<"b", 255>
  ]>;

def MyI16Enum : I16Enum<"MyI16Enum", "Example I16 enum", [
    I16EnumCase<"a", 65534>,
    I16EnumCase<"b", 65535>
  ]>;
