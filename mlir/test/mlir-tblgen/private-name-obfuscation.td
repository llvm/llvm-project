// Verifies that --mlir-private-name-obfuscator replaces the names of items
// in dialects listed via --mlir-private-dialects with deterministic opaque
// identifiers, and leaves items in other dialects unchanged.
//
// RUN: mlir-tblgen -gen-dialect-decls -dialect=publicd -I %S/../../include %s \
// RUN:   | FileCheck %s --check-prefix=PUBDD-OFF
// RUN: mlir-tblgen -gen-dialect-decls -dialect=publicd -I %S/../../include %s \
// RUN:     --mlir-private-name-obfuscator=md5sum --mlir-private-dialects=privd \
// RUN:   | FileCheck %s --check-prefix=PUBDD-ON
// RUN: mlir-tblgen -gen-dialect-decls -dialect=privd -I %S/../../include %s \
// RUN:   | FileCheck %s --check-prefix=PRIVDD-OFF
// RUN: mlir-tblgen -gen-dialect-decls -dialect=privd -I %S/../../include %s \
// RUN:     --mlir-private-name-obfuscator=md5sum --mlir-private-dialects=privd \
// RUN:   | FileCheck %s --check-prefix=PRIVDD-ON
// RUN: mlir-tblgen -gen-op-decls -dialect=privd -I %S/../../include %s \
// RUN:     --mlir-private-name-obfuscator=md5sum --mlir-private-dialects=privd \
// RUN:   | FileCheck %s --check-prefix=PRIVOP-ON
// RUN: mlir-tblgen -gen-op-decls -dialect=publicd -I %S/../../include %s \
// RUN:     --mlir-private-name-obfuscator=md5sum --mlir-private-dialects=privd \
// RUN:   | FileCheck %s --check-prefix=PUBOP-ON
// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s \
// RUN:     --mlir-private-name-obfuscator=md5sum --mlir-private-dialects=privd \
// RUN:   | FileCheck %s --check-prefix=PRIVFORMAT-ON
// RUN: mlir-tblgen -gen-attrdef-decls -attrdefs-dialect=privd -I %S/../../include %s \
// RUN:     --mlir-private-name-obfuscator=md5sum --mlir-private-dialects=privd \
// RUN:   | FileCheck %s --check-prefix=ATTR-PRIV-ON
// RUN: mlir-tblgen -gen-attrdef-decls -attrdefs-dialect=publicd -I %S/../../include %s \
// RUN:     --mlir-private-name-obfuscator=md5sum --mlir-private-dialects=privd \
// RUN:   | FileCheck %s --check-prefix=ATTR-PUB-ON
// RUN: mlir-tblgen -gen-attrdef-defs -attrdefs-dialect=privd -I %S/../../include %s \
// RUN:     --mlir-private-name-obfuscator=md5sum --mlir-private-dialects=privd \
// RUN:   | FileCheck %s --check-prefix=ATTR-PRIV-DEF-ON
// RUN: mlir-tblgen -gen-typedef-decls -typedefs-dialect=privd -I %S/../../include %s \
// RUN:     --mlir-private-name-obfuscator=md5sum --mlir-private-dialects=privd \
// RUN:   | FileCheck %s --check-prefix=TYPE-PRIV-ON
// RUN: mlir-tblgen -gen-typedef-decls -typedefs-dialect=publicd -I %S/../../include %s \
// RUN:     --mlir-private-name-obfuscator=md5sum --mlir-private-dialects=privd \
// RUN:   | FileCheck %s --check-prefix=TYPE-PUB-ON
// RUN: mlir-tblgen -gen-typedef-defs -typedefs-dialect=privd -I %S/../../include %s \
// RUN:     --mlir-private-name-obfuscator=md5sum --mlir-private-dialects=privd \
// RUN:   | FileCheck %s --check-prefix=TYPE-PRIV-DEF-ON
//
// Determinism: rerunning with the same obfuscator produces identical output.
// RUN: mlir-tblgen -gen-dialect-decls -dialect=privd -I %S/../../include %s \
// RUN:     --mlir-private-name-obfuscator=md5sum --mlir-private-dialects=privd > %t.first
// RUN: mlir-tblgen -gen-dialect-decls -dialect=privd -I %S/../../include %s \
// RUN:     --mlir-private-name-obfuscator=md5sum --mlir-private-dialects=privd > %t.second
// RUN: cmp %t.first %t.second
//
// A different obfuscator produces different output.
// RUN: mlir-tblgen -gen-dialect-decls -dialect=privd -I %S/../../include %s \
// RUN:     --mlir-private-name-obfuscator=sha1sum --mlir-private-dialects=privd > %t.alt
// RUN: not cmp %t.first %t.alt

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def PublicDialect : Dialect {
  let name = "publicd";
  let cppNamespace = "::publicd";
}

def PrivateDialect : Dialect {
  let name = "privd";
  let cppNamespace = "::privd";
}

def Public_AddOp : Op<PublicDialect, "add", []>;
def Public_PublicFormatOp : Op<PublicDialect, "format", []> {
  let assemblyFormat = "attr-dict";
}
def Private_MulOp : Op<PrivateDialect, "mul", []>;
def Private_PrivateFormatOp : Op<PrivateDialect, "format", []> {
  let assemblyFormat = "attr-dict";
}

def Public_PublicAttr : AttrDef<PublicDialect, "Public"> {
  let mnemonic = "public_attr";
  let genMnemonicAlias = 1;
}
def Private_PrivateAttr : AttrDef<PrivateDialect, "Private"> {
  let mnemonic = "private_attr";
  let genMnemonicAlias = 1;
}

def Public_PublicType : TypeDef<PublicDialect, "Public"> {
  let mnemonic = "public_type";
  let genMnemonicAlias = 1;
}
def Private_PrivateType : TypeDef<PrivateDialect, "Private"> {
  let mnemonic = "private_type";
  let genMnemonicAlias = 1;
}

// The obfuscated form of each private TableGen name in this file is the
// md5 hash of the name, prefixed with `_`:
//
//   printf %s privd        | md5sum  -> 1e8d1fee65b16e634839dea9f85d3429
//   printf %s mul          | md5sum  -> 353942263d1bedfbe06b7bfa78226253
//   printf %s format       | md5sum  -> 1ddcb92ade31c8fbd370001f9b29a7d9
//   printf %s private_attr | md5sum  -> d0f28b2f0516a6ff810258f8d1b6899d
//   printf %s private_type | md5sum  -> ff03a49c6c57d69cd51f5420a768f9ef

// Public dialect, obfuscation off: original namespace.
// PUBDD-OFF: getDialectNamespace()
// PUBDD-OFF: StringLiteral("publicd")

// Public dialect, obfuscation on: still original namespace.
// PUBDD-ON: getDialectNamespace()
// PUBDD-ON: StringLiteral("publicd")

// Private dialect, obfuscation off: original namespace.
// PRIVDD-OFF: getDialectNamespace()
// PRIVDD-OFF: StringLiteral("privd")

// Private dialect, obfuscation on: md5(privd) with `_` prefix.
// PRIVDD-ON: getDialectNamespace()
// PRIVDD-ON: StringLiteral("_1e8d1fee65b16e634839dea9f85d3429")
// PRIVDD-ON-NOT: StringLiteral("privd")

// Op of private dialect, obfuscation on: both halves obfuscated. The first
// private op encountered (alphabetically) is MulOp.
// PRIVOP-ON: getOperationName()
// PRIVOP-ON: StringLiteral("_1e8d1fee65b16e634839dea9f85d3429._353942263d1bedfbe06b7bfa78226253")
// PRIVOP-ON-NOT: StringLiteral("privd.mul")
// PRIVOP-ON-NOT: StringLiteral("privd.format")

// Op of public dialect, obfuscation on: unchanged.
// PUBOP-ON: getOperationName()
// PUBOP-ON: StringLiteral("publicd.add")

// Declarative assembly formats of private ops are not generated under private
// obfuscation. They therefore fall back to generic printing and reject the
// custom textual syntax.
// PRIVFORMAT-ON-LABEL: class PrivateFormatOp : public
// PRIVFORMAT-ON: _1e8d1fee65b16e634839dea9f85d3429._1ddcb92ade31c8fbd370001f9b29a7d9
// PRIVFORMAT-ON-NOT: static ::mlir::ParseResult parse
// PRIVFORMAT-ON-NOT: void print
// PRIVFORMAT-ON-LABEL: class PublicFormatOp : public
// PRIVFORMAT-ON: publicd.format
// PRIVFORMAT-ON: static ::mlir::ParseResult parse
// PRIVFORMAT-ON: void print

// Attribute names and mnemonics of private dialects are obfuscated, while
// public attributes remain unchanged.
// ATTR-PRIV-ON-LABEL: class PrivateAttr : public
// ATTR-PRIV-ON: static constexpr ::llvm::StringLiteral name = "_1e8d1fee65b16e634839dea9f85d3429._d0f28b2f0516a6ff810258f8d1b6899d";
// ATTR-PRIV-ON: static constexpr ::llvm::StringLiteral dialectName = "_1e8d1fee65b16e634839dea9f85d3429";
// ATTR-PRIV-ON: getMnemonic()
// ATTR-PRIV-ON: return {"_d0f28b2f0516a6ff810258f8d1b6899d"};
// ATTR-PRIV-ON-NOT: private_attr
// ATTR-PRIV-DEF-ON: os << "_d0f28b2f0516a6ff810258f8d1b6899d";
// ATTR-PRIV-DEF-ON-NOT: private_attr
// ATTR-PUB-ON-LABEL: class PublicAttr : public
// ATTR-PUB-ON: static constexpr ::llvm::StringLiteral name = "publicd.public_attr";
// ATTR-PUB-ON: static constexpr ::llvm::StringLiteral dialectName = "publicd";
// ATTR-PUB-ON: getMnemonic()
// ATTR-PUB-ON: return {"public_attr"};

// Type names and mnemonics of private dialects are obfuscated, while public
// types remain unchanged.
// TYPE-PRIV-ON-LABEL: class PrivateType : public
// TYPE-PRIV-ON: static constexpr ::llvm::StringLiteral name = "_1e8d1fee65b16e634839dea9f85d3429._ff03a49c6c57d69cd51f5420a768f9ef";
// TYPE-PRIV-ON: static constexpr ::llvm::StringLiteral dialectName = "_1e8d1fee65b16e634839dea9f85d3429";
// TYPE-PRIV-ON: getMnemonic()
// TYPE-PRIV-ON: return {"_ff03a49c6c57d69cd51f5420a768f9ef"};
// TYPE-PRIV-ON-NOT: private_type
// TYPE-PRIV-DEF-ON: os << "_ff03a49c6c57d69cd51f5420a768f9ef";
// TYPE-PRIV-DEF-ON-NOT: private_type
// TYPE-PUB-ON-LABEL: class PublicType : public
// TYPE-PUB-ON: static constexpr ::llvm::StringLiteral name = "publicd.public_type";
// TYPE-PUB-ON: static constexpr ::llvm::StringLiteral dialectName = "publicd";
// TYPE-PUB-ON: getMnemonic()
// TYPE-PUB-ON: return {"public_type"};
