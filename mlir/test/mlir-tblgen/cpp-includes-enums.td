// RUN: mlir-tblgen -gen-enum-decls -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-enum-defs -I %S/../../include %s | FileCheck %s --check-prefix=DEF
// RUN: mlir-tblgen -gen-enum-decls -DPARSER -I %S/../../include %s | FileCheck %s --check-prefix=PARSER-DECL
// RUN: mlir-tblgen -gen-enum-defs -DPARSER -I %S/../../include %s | FileCheck %s --check-prefix=PARSER-DEF
// RUN: mlir-tblgen -gen-enum-decls -DATTR -I %S/../../include %s | FileCheck %s --check-prefix=ATTR-DECL
// RUN: mlir-tblgen -gen-enum-defs -DATTR -I %S/../../include %s | FileCheck %s --check-prefix=ATTR-DEF

include "mlir/IR/EnumAttr.td"

// Enum declarations and definitions include the headers declaring the C++
// constructs they use. The definitions only include the headers needed beyond
// the declarations.

#ifndef PARSER
#ifndef ATTR

// An enum without an underlying type has no parser and printer.
def CaseA : I32EnumCase<"A", 0>;
def NoParserEnum : I32Enum<"NoParserEnum", "", [CaseA]> {
  let underlyingType = "";
}

// DECL: #include "llvm/ADT/DenseMapInfo.h"
// DECL-NEXT: #include "llvm/ADT/StringRef.h"
// DECL-EMPTY:
// DECL-NOT: #include

// DEF: #include "llvm/ADT/StringSwitch.h"
// DEF-EMPTY:
// DEF-NOT: #include

#endif
#endif

#ifdef PARSER

// The parser and printer need `raw_ostream`, `FailureOr` and `AsmParser`. The
// printer of a bit enum printed as a single keyword also needs
// `llvm::has_single_bit`, and the conversion functions of bit enums split and
// join strings.
def Bit0 : I32BitEnumAttrCaseBit<"Bit0", 0>;
def Bit1 : I32BitEnumAttrCaseBit<"Bit1", 1>;
def BitGroup : BitEnumCaseGroup<"BitGroup", [Bit0, Bit1]>;
def ParserBitEnum : I32BitEnumAttr<"ParserBitEnum", "", [Bit0, Bit1, BitGroup]> {
  let genSpecializedAttr = 0;
}

// PARSER-DECL: #include "llvm/ADT/DenseMapInfo.h"
// PARSER-DECL-NEXT: #include "llvm/ADT/StringRef.h"
// PARSER-DECL-NEXT: #include "llvm/ADT/bit.h"
// PARSER-DECL-NEXT: #include "llvm/Support/raw_ostream.h"
// PARSER-DECL-NEXT: #include "mlir/Support/LLVM.h"
// PARSER-DECL-EMPTY:
// PARSER-DECL-NOT: #include

// PARSER-DEF: #include "llvm/ADT/SmallVector.h"
// PARSER-DEF-NEXT: #include "llvm/ADT/StringExtras.h"
// PARSER-DEF-NEXT: #include "llvm/ADT/StringSwitch.h"
// PARSER-DEF-NEXT: #include "mlir/IR/OpImplementation.h"
// PARSER-DEF-EMPTY:
// PARSER-DEF-NOT: #include

#endif

#ifdef ATTR

// The specialized attribute derives from `IntegerAttr` and is built from an
// `IntegerType`.
def CaseB : I32EnumAttrCase<"B", 0>;
def AttrEnum : I32EnumAttr<"AttrEnum", "", [CaseB]>;

// ATTR-DECL: #include "llvm/ADT/DenseMapInfo.h"
// ATTR-DECL-NEXT: #include "llvm/ADT/StringRef.h"
// ATTR-DECL-NEXT: #include "llvm/Support/raw_ostream.h"
// ATTR-DECL-NEXT: #include "mlir/IR/BuiltinAttributes.h"
// ATTR-DECL-NEXT: #include "mlir/Support/LLVM.h"
// ATTR-DECL-EMPTY:
// ATTR-DECL-NOT: #include

// ATTR-DEF: #include "llvm/ADT/StringSwitch.h"
// ATTR-DEF-NEXT: #include "mlir/IR/BuiltinTypes.h"
// ATTR-DEF-NEXT: #include "mlir/IR/OpImplementation.h"
// ATTR-DEF-EMPTY:
// ATTR-DEF-NOT: #include

#endif
