//===-- APIDefs.td - Base definitions for Offload tablegen -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the class definitions used to implement the Offload API,
// as well as helper functions used to help populate relevant records.
// See offload/API/README.md for more detailed documentation.
//
//===----------------------------------------------------------------------===//


// Parameter flags
defvar PARAM_IN = 0x1;
defvar PARAM_OUT = 0x2;
defvar PARAM_OPTIONAL = 0x4;

// Prefix for API naming. This could be hard-coded in the future when a value
// is agreed upon.
defvar PREFIX = "OL";
defvar prefix = !tolower(PREFIX);

// Does the type end with '_handle_t'?
class IsHandleType<string Type> {
  // size("_handle_t") == 9
  bit ret = !if(!lt(!size(Type), 9), 0,
                !ne(!find(Type, "_handle_t", !sub(!size(Type), 9)), -1));
}

// Does the type end with '*'?
class IsPointerType<string Type> {
  bit ret = !ne(!find(Type, "*", !sub(!size(Type), 1)), -1);
}

class Param<string Type, string Name, string Desc, bits<3> Flags = 0> {
  string type = Type;
  string name = Name;
  string desc = Desc;
  bits<3> flags = Flags;
  bit IsHandle = IsHandleType<type>.ret;
  bit IsPointer = IsPointerType<type>.ret;
}

class Return<string Value, list<string> Conditions = []> {
  string value = Value;
  list<string> conditions = Conditions;
}

class ShouldCheckHandle<Param P> {
  bit ret = !and(P.IsHandle, !eq(!and(PARAM_OPTIONAL, P.flags), 0));
}

class ShouldCheckPointer<Param P> {
  bit ret = !and(P.IsPointer, !eq(!and(PARAM_OPTIONAL, P.flags), 0));
}

// For a list of returns that contains a specific return code, find and append
// new conditions to that return
class AppendConditionsToReturn<list<Return> Returns, string ReturnValue,
                               list<string> Conditions> {
  list<Return> ret =
      !foreach(Ret, Returns,
               !if(!eq(Ret.value, ReturnValue),
                   Return<Ret.value, Ret.conditions#Conditions>, Ret));
}

// Add null handle checks to a function's return value descriptions
class AddHandleChecksToReturns<list<Param> Params, list<Return> Returns> {
  list<string> handle_params =
      !foreach(P, Params, !if(ShouldCheckHandle<P>.ret, P.name, ""));
  list<string> handle_params_filt =
      !filter(param, handle_params, !ne(param, ""));
  list<string> handle_param_conds =
      !foreach(handle, handle_params_filt, "`NULL == "#handle#"`");

  // Does the list of returns already contain ERROR_INVALID_NULL_HANDLE?
  bit returns_has_inv_handle = !foldl(
      0, Returns, HasErr, Ret,
      !or(HasErr, !eq(Ret.value, PREFIX#"_RESULT_ERROR_INVALID_NULL_HANDLE")));

  list<Return> returns_out = !if(returns_has_inv_handle,
        AppendConditionsToReturn<Returns, PREFIX # "_RESULT_ERROR_INVALID_NULL_HANDLE", handle_param_conds>.ret,
        !listconcat(Returns, [Return<PREFIX # "_RESULT_ERROR_INVALID_NULL_HANDLE", handle_param_conds>])
    );
}

// Add null pointer checks to a function's return value descriptions
class AddPointerChecksToReturns<list<Param> Params, list<Return> Returns> {
  list<string> ptr_params =
      !foreach(P, Params, !if(ShouldCheckPointer<P>.ret, P.name, ""));
  list<string> ptr_params_filt = !filter(param, ptr_params, !ne(param, ""));
  list<string> ptr_param_conds =
      !foreach(ptr, ptr_params_filt, "`NULL == "#ptr#"`");

  // Does the list of returns already contain ERROR_INVALID_NULL_POINTER?
  bit returns_has_inv_ptr = !foldl(
      0, Returns, HasErr, Ret,
      !or(HasErr, !eq(Ret.value, PREFIX#"_RESULT_ERROR_INVALID_NULL_POINTER")));
  list<Return> returns_out = !if(returns_has_inv_ptr,
        AppendConditionsToReturn<Returns, PREFIX # "_RESULT_ERROR_INVALID_NULL_POINTER", ptr_param_conds>.ret,
        !listconcat(Returns, [Return<PREFIX # "_RESULT_ERROR_INVALID_NULL_POINTER", ptr_param_conds>])
    );
}

defvar DefaultReturns = [Return<PREFIX#"_RESULT_SUCCESS">,
                         Return<PREFIX#"_RESULT_ERROR_UNINITIALIZED">,
                         Return<PREFIX#"_RESULT_ERROR_DEVICE_LOST">,
                         Return<PREFIX#"_RESULT_ERROR_ADAPTER_SPECIFIC">];

class APIObject {
  string name;
  string desc;
}

class Function<string Class> : APIObject {
  string api_class = Class;
  list<Param> params;
  list<Return> returns;
  list<string> details = [];
  list<string> analogues = [];

  list<Return> returns_with_def = !listconcat(DefaultReturns, returns);
  list<Return> all_returns = AddPointerChecksToReturns<params,
        AddHandleChecksToReturns<params, returns_with_def>.returns_out>.returns_out;
}

class Etor<string Name, string Desc> {
  string name = Name;
  string desc = Desc;
}

class Enum : APIObject {
  // This refers to whether the enumerator descriptions specify a return
  // type for functions where this enum may be used as an input type.
  // The format is "[$x_some_return_t] Description text"
  // (TODO: This is lifted from UR, is it relevant?)
  bit is_typed = 0;

  list<Etor> etors = [];
}

class StructMember<string Type, string Name, string Desc> {
  string type = Type;
  string name = Name;
  string desc = Desc;
}

defvar DefaultPropStructMembers =
    [StructMember<prefix#"_structure_type_t", "stype",
                  "type of this structure">,
     StructMember<"void*", "pNext", "pointer to extension-specific structure">];

class StructHasInheritedMembers<string BaseClass> {
  bit ret = !or(!eq(BaseClass, prefix#"_base_properties_t"),
                !eq(BaseClass, prefix#"_base_desc_t"));
}

class Struct : APIObject {
  string base_class = "";
  list<StructMember> members;
  list<StructMember> all_members =
      !if(StructHasInheritedMembers<base_class>.ret,
          DefaultPropStructMembers, [])#members;
}

class Typedef : APIObject { string value; }

class FptrTypedef : APIObject {
  list<Param> params;
  list<Return> returns;
}

class Macro : APIObject {
  string value;

  string condition;
  string alt_value;
}

class Handle : APIObject;
