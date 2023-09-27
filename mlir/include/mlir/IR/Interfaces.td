//===-- Interfaces.td - Interfaces defination file ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definations for Interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef INTERFACES_TD
#define INTERFACES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Constraints.td"
include "mlir/IR/Traits.td"

//===----------------------------------------------------------------------===//
// Interface definitions
//===----------------------------------------------------------------------===//

// InterfaceTrait corresponds to a specific 'Interface' class defined in C++.
// The purpose to wrap around C++ symbol string with this class is to make
// interfaces specified for ops in TableGen less alien and more integrated.
class InterfaceTrait<string name> : NativeTrait<"", ""> {
  let trait = name # "::Trait";
  let cppNamespace = "";

  // An optional code block containing extra declarations to place in the
  // interface trait declaration.
  code extraTraitClassDeclaration = "";
}

// OpInterfaceTrait corresponds to a specific 'OpInterface' class defined in
// C++. The purpose to wrap around C++ symbol string with this class is to make
// interfaces specified for ops in TableGen less alien and more integrated.
class OpInterfaceTrait<string name, code verifyBody = [{}],
                       list<Trait> traits = []>
    : InterfaceTrait<name> {
  // Specify the body of the verification function. `$_op` will be replaced with
  // the operation being verified.
  code verify = verifyBody;

  // A bit indicating if the verifier needs to access the ops in the regions. If
  // it set to `1`, the region ops will be verified before invoking this
  // verifier.
  bit verifyWithRegions = 0;

  // Specify the list of traits that need to be verified before the verification
  // of this OpInterfaceTrait.
  list<Trait> dependentTraits = traits;
}

// This class represents a single, optionally static, interface method.
// Note: non-static interface methods have an implicit parameter, either
// $_op/$_attr/$_type corresponding to an instance of the derived value.
class InterfaceMethod<string desc, string retTy, string methodName,
                      dag args = (ins), code methodBody = [{}],
                      code defaultImplementation = [{}]> {
  // A human-readable description of what this method does.
  string description = desc;

  // The name of the interface method.
  string name = methodName;

  // The c++ type-name of the return type.
  string returnType = retTy;

  // A dag of string that correspond to the arguments of the method.
  dag arguments = args;

  // An optional body to the method.
  code body = methodBody;

  // An optional default implementation of the method.
  code defaultBody = defaultImplementation;
}

// This class represents a single static interface method.
class StaticInterfaceMethod<string desc, string retTy, string methodName,
                            dag args = (ins), code methodBody = [{}],
                            code defaultImplementation = [{}]>
    : InterfaceMethod<desc, retTy, methodName, args, methodBody,
                      defaultImplementation>;

// Interface represents a base interface.
class Interface<string name, list<Interface> baseInterfacesArg = []> {
  // A human-readable description of what this interface does.
  string description = "";

  // The name given to the c++ interface class.
  string cppInterfaceName = name;

  // The C++ namespace that this interface should be placed into.
  //
  // To specify nested namespaces, use "::" as the delimiter, e.g., given
  // "A::B", ops will be placed in `namespace A { namespace B { <def> } }`.
  string cppNamespace = "";

  // The list of methods defined by this interface.
  list<InterfaceMethod> methods = [];

  // An optional code block containing extra declarations to place in the
  // interface declaration.
  code extraClassDeclaration = "";

  // An optional code block containing extra declarations to place in both
  // the interface and trait declaration.
  code extraSharedClassDeclaration = "";

  // An optional code block for adding additional "classof" logic. This can
  // be used to better enable "optional" interfaces, where an entity only
  // implements the interface if some dynamic characteristic holds.
  // `$_attr`/`$_op`/`$_type` may be used to refer to an instance of the
  // interface instance being checked.
  code extraClassOf = "";

  // An optional set of base interfaces that this interface
  // "derives" from.
  list<Interface> baseInterfaces = baseInterfacesArg;
}

// AttrInterface represents an interface registered to an attribute.
class AttrInterface<string name, list<Interface> baseInterfaces = []>
  : Interface<name, baseInterfaces>, InterfaceTrait<name>,
	  Attr<CPred<"::llvm::isa<"
		  # !if(!empty(cppNamespace),"", cppNamespace # "::") # name # ">($_self)">,
			name # " instance"
    > {
	let storageType = !if(!empty(cppNamespace), "", cppNamespace # "::") # name;
	let returnType = storageType;
	let convertFromStorage = "$_self";
}

// OpInterface represents an interface registered to an operation.
class OpInterface<string name, list<Interface> baseInterfaces = []>
  : Interface<name, baseInterfaces>, OpInterfaceTrait<name>;

// TypeInterface represents an interface registered to a type.
class TypeInterface<string name, list<Interface> baseInterfaces = []>
  : Interface<name, baseInterfaces>, InterfaceTrait<name>,
	  Type<CPred<"::llvm::isa<"
		  # !if(!empty(cppNamespace),"", cppNamespace # "::") # name # ">($_self)">,
			name # " instance",
			!if(!empty(cppNamespace),"", cppNamespace # "::") # name
    >;

// Whether to declare the interface methods in the user entity's header. This
// class simply wraps an Interface but is used to indicate that the method
// declarations should be generated. This class takes an optional set of methods
// that should have declarations generated even if the method has a default
// implementation.
class DeclareInterfaceMethods<list<string> overridenMethods = []> {
    // This field contains a set of method names that should always have their
    // declarations generated. This allows for generating declarations for
    // methods with default implementations that need to be overridden.
    list<string> alwaysOverriddenMethods = overridenMethods;
}
class DeclareAttrInterfaceMethods<AttrInterface interface,
                                  list<string> overridenMethods = []>
      : DeclareInterfaceMethods<overridenMethods>,
        AttrInterface<interface.cppInterfaceName, interface.baseInterfaces> {
    let description = interface.description;
    let cppInterfaceName = interface.cppInterfaceName;
    let cppNamespace = interface.cppNamespace;
    let methods = interface.methods;
    let baseInterfaces = interface.baseInterfaces;
}
class DeclareOpInterfaceMethods<OpInterface interface,
                                list<string> overridenMethods = []>
      : DeclareInterfaceMethods<overridenMethods>,
        OpInterface<interface.cppInterfaceName, interface.baseInterfaces> {
    let description = interface.description;
    let cppInterfaceName = interface.cppInterfaceName;
    let cppNamespace = interface.cppNamespace;
    let methods = interface.methods;
    let baseInterfaces = interface.baseInterfaces;
}
class DeclareTypeInterfaceMethods<TypeInterface interface,
                                  list<string> overridenMethods = []>
      : DeclareInterfaceMethods<overridenMethods>,
        TypeInterface<interface.cppInterfaceName, interface.baseInterfaces> {
    let description = interface.description;
    let cppInterfaceName = interface.cppInterfaceName;
    let cppNamespace = interface.cppNamespace;
    let methods = interface.methods;
    let baseInterfaces = interface.baseInterfaces;
}

#endif // INTERFACES_TD
