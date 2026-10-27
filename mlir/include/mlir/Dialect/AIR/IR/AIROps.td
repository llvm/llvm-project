//===-- AIR.td - AIR dialect definition --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIR_OPS
#define AIR_OPS

include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def AIR_Dialect : Dialect {
  let name = "air";
  let cppNamespace = "::mlir::air";
  let summary = "The AIR dialect that targets the Apple GPU (Metal/AIR) "
                "LLVM backend.";
  let description = [{
    The AIR dialect owns the `#air.target` GPU compilation target attribute,
    making the Apple GPU (Apple Intermediate Representation / Metal) LLVM
    backend a first-class `gpu` dialect target. It carries no ops; lowering to
    AIR happens entirely in the AIR LLVM backend codegen pipeline.
  }];

  let useDefaultAttributePrinterParser = 1;
}

class AIR_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<AIR_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// AIR target attribute.
//===----------------------------------------------------------------------===//

def AIR_TargetAttr : AIR_Attr<"AIRTarget", "target"> {
  let description = [{
    GPU target attribute for controlling compilation of Apple GPU targets. All
    parameters decay into default values if not present.

    Examples:

    1. Target with default values: a `.metallib` for Apple's Metal driver.
    ```
      gpu.module @mymodule [#air.target] attributes {...} {
        ...
      }
    ```

    2. An AGX chip (an LLVM AGX processor, such as `agx-g14g`) as `chip`: an
       AGX object, compiled from the module's AIR by the AGX backend
       (agx-lower-air), with the AGX target features `features`.
    ```
      gpu.module @mymodule [#air.target<chip = "agx-g14g">,
                            #air.target<chip = "agx-g13g", features = "+agx-strict">]
    ```
  }];
  let parameters =
      (ins DefaultValuedParameter<"int", "3",
                                  "Optimization level to apply.">:$O,
          StringRefParameter<"Target triple.",
                             "\"air64_v28-apple-macosx26.0.0\"">:$triple,
          StringRefParameter<"Target chip.", "\"\"">:$chip,
          StringRefParameter<"Target features (AGX chips).", "\"\"">:$features);
  let assemblyFormat = [{
    (`<` struct($O, $triple, $chip, $features)^ `>`)?
  }];
  let builders = [AttrBuilder<
      (ins CArg<"int", "3">:$optLevel,
          CArg<"::llvm::StringRef", "\"air64_v28-apple-macosx26.0.0\"">:$triple,
          CArg<"::llvm::StringRef", "\"\"">:$chip,
          CArg<"::llvm::StringRef", "\"\"">:$features),
      [{
      return Base::get($_ctxt, optLevel, triple, chip, features);
    }]>];
  let extraClassDeclaration = [{
    /// Whether the chip is an AGX processor: the target is then an AGX object.
    bool isAGX() const { return getChip().starts_with("agx-"); }
  }];
  let skipDefaultBuilders = 1;
}

#endif // AIR_OPS
