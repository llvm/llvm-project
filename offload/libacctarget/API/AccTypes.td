//===- AccTypes.td - OpenACC runtime enums and macros -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "AccDefs.td"

def VoidTy : AccType<"void", "">;
def IntTy : AccType<"int", "integer(c_int)", "k">;
def HandleTy : AccType<"int64_t", "integer(acc_handle_kind)", "k", "async">;
def DeviceTy : AccType<"acc_device_t", "integer(acc_device_kind)", "k">;
def PropTy : AccType<"acc_device_property_t",
                     "integer(acc_device_property_kind)", "k">;
def SizeTy : AccType<"size_t", "integer(c_size_t)", "k">;
def VoidPtrTy : AccType<"void *", "type(c_ptr)">;
def LogicalTy : AccType<"bool", "logical(c_bool)", "", "logical">;
def HostArrayTy : AccType<"void *", "type(*), dimension(*)">;
def HandleArrayTy : AccType<"int64_t *",
                            "integer(acc_handle_kind), dimension(*)">;
def CfiDescTy : AccType<"const CFI_cdesc_t *", "type(*), dimension(..)">;
def CharacterTy : AccType<"CFI_cdesc_t *", "character(len=*)">;

def AccHandleKind : FortranKind<"acc_handle_kind", "8">;
def AccDeviceKind : FortranKind<"acc_device_kind", "c_int">;
def AccDevicePropertyKind : FortranKind<"acc_device_property_kind", "c_int">;

def : AccEnum<"acc_device_t", AccDeviceKind, [
  Enumerator<"acc_device_none", 0>,
  Enumerator<"acc_device_default", 1>,
  Enumerator<"acc_device_host", 2>,
  Enumerator<"acc_device_not_host", 3>,
  Enumerator<"acc_device_nvidia", 4>,
  Enumerator<"acc_device_radeon", 5>,
  Enumerator<"acc_device_intelgpu", 6>,
  Enumerator<"acc_device_current", 10>
]>;

def : AccEnum<"acc_device_property_t", AccDevicePropertyKind, [
  Enumerator<"acc_property_memory", 1>,
  Enumerator<"acc_property_free_memory", 2>,
  Enumerator<"acc_property_name", 3>,
  Enumerator<"acc_property_vendor", 4>,
  Enumerator<"acc_property_driver", 5>,
  Enumerator<"acc_property_shared_memory_support", 6>
]>;

def : AccConst<"acc_async_sync", AccHandleKind, -1>;
def : AccConst<"acc_async_default", AccHandleKind, -3>;
def : AccConst<"acc_async_noval", AccHandleKind, -4>;

// The value of _OPENACC for OpenACC 3.4, the version of the specification that
// this API follows.
def : FortranInt<"openacc_version", 202506>;
