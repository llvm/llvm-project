//===- SymbolInterfaces.td - Interfaces for symbol ops -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces and traits that can be used to define
// properties of symbol and symbol table operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_SYMBOLINTERFACES
#define MLIR_IR_SYMBOLINTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// SymbolOpInterface
//===----------------------------------------------------------------------===//

def Symbol : OpInterface<"SymbolOpInterface"> {
  let description = [{
    This interface describes an operation that may define a `Symbol`. A `Symbol`
    operation resides immediately within a region that defines a `SymbolTable`.
    See [Symbols and SymbolTables](SymbolsAndSymbolTables.md) for more details
    and constraints on `Symbol` operations.
  }];

  let methods = [
    InterfaceMethod<"Returns the name of this symbol.",
      "StringRef", "getName", (ins), [{
        // Don't rely on the trait implementation as optional symbol operations
        // may override this.
        return mlir::SymbolTable::getSymbolName(op);
      }], /*defaultImplementation=*/[{
        return mlir::SymbolTable::getSymbolName(this->getOperation());
      }]
    >,
    InterfaceMethod<"Sets the name of this symbol.",
      "void", "setName", (ins "StringRef":$name), [{}],
      /*defaultImplementation=*/[{
        this->getOperation()->setAttr(
            mlir::SymbolTable::getSymbolAttrName(),
            StringAttr::get(name, this->getOperation()->getContext()));
      }]
    >,
    InterfaceMethod<"Gets the visibility of this symbol.",
      "mlir::SymbolTable::Visibility", "getVisibility", (ins), [{}],
      /*defaultImplementation=*/[{
        return mlir::SymbolTable::getSymbolVisibility(this->getOperation());
      }]
    >,
    InterfaceMethod<"Sets the visibility of this symbol.",
      "void", "setVisibility", (ins "mlir::SymbolTable::Visibility":$vis), [{}],
      /*defaultImplementation=*/[{
        mlir::SymbolTable::setSymbolVisibility(this->getOperation(), vis);
      }]
    >,
    InterfaceMethod<"Returns true if this symbol has nested visibility.",
      "bool", "isNested", (ins),  [{}],
      /*defaultImplementation=*/[{
        return getVisibility() == mlir::SymbolTable::Visibility::Nested;
      }]
    >,
    InterfaceMethod<"Returns true if this symbol has private visibility.",
      "bool", "isPrivate", (ins),  [{}],
      /*defaultImplementation=*/[{
        return getVisibility() == mlir::SymbolTable::Visibility::Private;
      }]
    >,
    InterfaceMethod<"Returns true if this symbol has public visibility.",
      "bool", "isPublic", (ins),  [{}],
      /*defaultImplementation=*/[{
        return getVisibility() == mlir::SymbolTable::Visibility::Public;
      }]
    >,
    InterfaceMethod<[{
        Get all of the uses of the current symbol that are nested within the
        given operation 'from'.
        Note: See mlir::SymbolTable::getSymbolUses for more details.
      }],
      "Optional<::mlir::SymbolTable::UseRange>", "getSymbolUses",
      (ins "Operation *":$from), [{}],
      /*defaultImplementation=*/[{
        return ::mlir::SymbolTable::getSymbolUses(this->getOperation(), from);
      }]
    >,
    InterfaceMethod<[{
        Return if the current symbol is known to have no uses that are nested
        within the given operation 'from'.
        Note: See mlir::SymbolTable::symbolKnownUseEmpty for more details.
      }],
      "bool", "symbolKnownUseEmpty", (ins "Operation *":$from), [{}],
      /*defaultImplementation=*/[{
        return ::mlir::SymbolTable::symbolKnownUseEmpty(this->getOperation(),
                                                        from);
      }]
    >,
    InterfaceMethod<[{
        Attempt to replace all uses of the current symbol with the provided
        symbol 'newSymbol' that are nested within the given operation 'from'.
        Note: See mlir::SymbolTable::replaceAllSymbolUses for more details.
      }],
      "LogicalResult", "replaceAllSymbolUses", (ins "StringRef":$newSymbol,
                                                    "Operation *":$from), [{}],
      /*defaultImplementation=*/[{
        return ::mlir::SymbolTable::replaceAllSymbolUses(this->getOperation(),
                                                         newSymbol, from);
      }]
    >,
    InterfaceMethod<[{
        Returns true if this operation optionally defines a symbol based on the
        presence of the symbol name.
      }],
      "bool", "isOptionalSymbol", (ins), [{}],
      /*defaultImplementation=*/[{ return false; }]
    >,
    InterfaceMethod<[{
        Returns true if this operation can be discarded if it has no remaining
        symbol uses.
      }],
      "bool", "canDiscardOnUseEmpty", (ins), [{}],
      /*defaultImplementation=*/[{
        // By default, base this on the visibility alone. A symbol can be
        // discarded as long as it is not public. Only public symbols may be
        // visible from outside of the IR.
        return getVisibility() != ::mlir::SymbolTable::Visibility::Public;
      }]
    >,
  ];

  let verify = [{
    // If this is an optional symbol, bail out early if possible.
    auto concreteOp = cast<ConcreteOp>($_op);
    if (concreteOp.isOptionalSymbol()) {
      if(!concreteOp.getAttr(::mlir::SymbolTable::getSymbolAttrName()))
        return success();
    }
    return ::mlir::detail::verifySymbol($_op);
  }];

  let extraClassDeclaration = [{
    /// Custom classof that handles the case where the symbol is optional.
    static bool classof(Operation *op) {
      return Base::classof(op)
        && op->getAttr(::mlir::SymbolTable::getSymbolAttrName());
    }
  }];

  let extraTraitClassDeclaration = [{
    using Visibility = mlir::SymbolTable::Visibility;
  }];
}

//===----------------------------------------------------------------------===//
// Symbol Traits
//===----------------------------------------------------------------------===//

// Op defines a symbol table.
def SymbolTable : NativeOpTrait<"SymbolTable">;

#endif // MLIR_IR_SYMBOLINTERFACES
