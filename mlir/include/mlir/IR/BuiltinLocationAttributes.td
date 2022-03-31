//===- BuiltinLocationAttributes.td - Builtin Locations ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the set of builtin MLIR location attributes.
//
//===----------------------------------------------------------------------===//

#ifndef BUILTIN_LOCATION_ATTRIBUTES_TD
#define BUILTIN_LOCATION_ATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"

// Base class for Builtin dialect location attributes.
class Builtin_LocationAttr<string name>
    : AttrDef<Builtin_Dialect, name, [], "::mlir::LocationAttr"> {
  let cppClassName = name;
  let mnemonic = ?;
}

//===----------------------------------------------------------------------===//
// CallSiteLoc
//===----------------------------------------------------------------------===//

def CallSiteLoc : Builtin_LocationAttr<"CallSiteLoc"> {
  let summary = "A callsite source location";
  let description = [{
    Syntax:

    ```
    callsite-location ::= `callsite` `(` location `at` location `)`
    ```

    An instance of this location allows for representing a directed stack of
    location usages. This connects a location of a `callee` with the location
    of a `caller`.

    Example:

    ```mlir
    loc(callsite("foo" at "mysource.cc":10:8))
    ```
  }];
  let parameters = (ins "Location":$callee, "Location":$caller);
  let builders = [
    AttrBuilderWithInferredContext<(ins "Location":$callee,
                                        "Location":$caller), [{
      return $_get(callee->getContext(), callee, caller);
    }]>,
    AttrBuilderWithInferredContext<(ins "Location":$name,
                                        "ArrayRef<Location>":$frames)>
  ];
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// FileLineColLoc
//===----------------------------------------------------------------------===//

def FileLineColLoc : Builtin_LocationAttr<"FileLineColLoc"> {
  let summary = "A file:line:column source location";
  let description = [{
    Syntax:

    ```
    filelinecol-location ::= string-literal `:` integer-literal `:`
                             integer-literal
    ```

    An instance of this location represents a tuple of file, line number, and
    column number. This is similar to the type of location that you get from
    most source languages.

    Example:

    ```mlir
    loc("mysource.cc":10:8)
    ```
  }];
  let parameters = (ins "StringAttr":$filename, "unsigned":$line,
                        "unsigned":$column);
  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$filename,
                                        "unsigned":$line,
                                        "unsigned":$column), [{
      return $_get(filename.getContext(), filename, line, column);
    }]>,
    AttrBuilder<(ins "StringRef":$filename, "unsigned":$line,
                     "unsigned":$column), [{
      return $_get($_ctxt,
                   StringAttr::get($_ctxt, filename.empty() ? "-" : filename),
                   line, column);
    }]>
  ];
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// FusedLoc
//===----------------------------------------------------------------------===//

def FusedLoc : Builtin_LocationAttr<"FusedLoc"> {
  let summary = "A tuple of other source locations";
  let description = [{
    Syntax:

    ```
    fused-location ::= `fused` fusion-metadata? `[` location (location `,`)* `]`
    fusion-metadata ::= `<` attribute-value `>`
    ```

    An instance of a `fused` location represents a grouping of several other
    source locations, with optional metadata that describes the context of the
    fusion. There are many places within a compiler in which several constructs
    may be fused together, e.g. pattern rewriting, that normally result partial
    or even total loss of location information. With `fused` locations, this is
    a non-issue.

    Example:

    ```mlir
    loc(fused["mysource.cc":10:8, "mysource.cc":22:8)
    loc(fused<"CSE">["mysource.cc":10:8, "mysource.cc":22:8])
    ```
  }];
  let parameters = (ins ArrayRefParameter<"Location", "">:$locations,
                        "Attribute":$metadata);
  let extraClassDeclaration = [{
    static Location get(ArrayRef<Location> locs, Attribute metadata,
                        MLIRContext *context);
    static Location get(MLIRContext *context, ArrayRef<Location> locs) {
      return get(locs, Attribute(), context);
    }
  }];
}

//===----------------------------------------------------------------------===//
// NameLoc
//===----------------------------------------------------------------------===//

def NameLoc : Builtin_LocationAttr<"NameLoc"> {
  let summary = "A named source location";
  let description = [{
    Syntax:

    ```
    name-location ::= string-literal (`(` location `)`)?
    ```

    An instance of this location allows for attaching a name to a child location.
    This can be useful for representing the locations of variable, or node,
    definitions.

    Example:

    ```mlir
    loc("CSE"("mysource.cc":10:8))
    ```
  }];
  let parameters = (ins "StringAttr":$name, "Location":$childLoc);
  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$name,
                                        "Location":$childLoc), [{
      return $_get(name.getContext(), name, childLoc);
    }]>,
    AttrBuilderWithInferredContext<(ins "StringAttr":$name), [{
      return $_get(name.getContext(), name,
                   UnknownLoc::get(name.getContext()));
    }]>
  ];
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// OpaqueLoc
//===----------------------------------------------------------------------===//

def OpaqueLoc : Builtin_LocationAttr<"OpaqueLoc"> {
  let summary = "An opaque source location";
  let description = [{
    An instance of this location essentially contains a pointer to some data
    structure that is external to MLIR and an optional location that can be used
    if the first one is not suitable. Since it contains an external structure,
    only the optional location is used during serialization.
  }];
  let parameters = (ins "uintptr_t":$underlyingLocation,
                        "TypeID":$underlyingTypeID,
                        "Location":$fallbackLocation);
  let builders = [
    AttrBuilderWithInferredContext<(ins "uintptr_t":$underlyingLocation,
                                        "TypeID":$underlyingTypeID,
                                        "Location":$fallbackLocation), [{
      return $_get(fallbackLocation->getContext(), underlyingLocation,
                   underlyingTypeID, fallbackLocation);
    }]>
  ];
  let extraClassDeclaration = [{
    /// Returns an instance of opaque location which contains a given pointer to
    /// an object. The corresponding MLIR location is set to UnknownLoc.
    template <typename T>
    static OpaqueLoc get(T underlyingLocation, MLIRContext *context);

    /// Returns an instance of opaque location which contains a given pointer to
    /// an object and an additional MLIR location.
    template <typename T>
    static OpaqueLoc get(T underlyingLocation, Location fallbackLocation) {
      return get(reinterpret_cast<uintptr_t>(underlyingLocation),
                 TypeID::get<T>(), fallbackLocation);
    }

    /// Returns a pointer to some data structure that opaque location stores.
    template <typename T> static T getUnderlyingLocation(Location location) {
      assert(isa<T>(location));
      return reinterpret_cast<T>(
          location.cast<mlir::OpaqueLoc>().getUnderlyingLocation());
    }

    /// Returns a pointer to some data structure that opaque location stores.
    /// Returns nullptr if provided location is not opaque location or if it
    /// contains a pointer of different type.
    template <typename T>
    static T getUnderlyingLocationOrNull(Location location) {
      return isa<T>(location)
                 ? reinterpret_cast<T>(
                       location.cast<mlir::OpaqueLoc>().getUnderlyingLocation())
                 : T(nullptr);
    }

    /// Checks whether provided location is opaque location and contains a
    /// pointer to an object of particular type.
    template <typename T> static bool isa(Location location) {
      auto opaque_loc = location.dyn_cast<OpaqueLoc>();
      return opaque_loc && opaque_loc.getUnderlyingTypeID() == TypeID::get<T>();
    }
  }];
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// UnknownLoc
//===----------------------------------------------------------------------===//

def UnknownLoc : Builtin_LocationAttr<"UnknownLoc"> {
  let summary = "An unspecified source location";
  let description = [{
    Syntax:

    ```
    unknown-location ::= `?`
    ```

    Source location information is an extremely integral part of the MLIR
    infrastructure. As such, location information is always present in the IR,
    and must explicitly be set to unknown. Thus, an instance of the `unknown`
    location represents an unspecified source location.

    Example:

    ```mlir
    loc(?)
    ```
  }];
  let extraClassDeclaration = [{
    static UnknownLoc get(MLIRContext *context);
  }];
}

#endif // BUILTIN_LOCATION_ATTRIBUTES_TD
