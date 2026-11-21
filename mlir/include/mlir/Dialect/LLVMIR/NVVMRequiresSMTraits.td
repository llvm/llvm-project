//===-- NVVMRequiresSMTraits.td - NVVM Requires SM Traits --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines traits for the NVVM Dialect in MLIR
//
//===----------------------------------------------------------------------===//

#ifndef NVVM_REQUIRES_SM_TRAITS
#define NVVM_REQUIRES_SM_TRAITS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

// Interface for NVVM Ops with the NVVMRequiresSM parametric trait
def RequiresSMInterface: OpInterface<"RequiresSMInterface"> {
  let cppNamespace = "::mlir::NVVM";
  let methods = [
    InterfaceMethod<
      "Get the target architectures the op is supported on",
      "::mlir::NVVM::NVVMArchSet", "getSupportedArchs"
    >
  ];
}

// sm_A implies sm_B when code written for sm_B also runs on sm_A.

// Keep in sync with the SM features in llvm/lib/Target/NVPTX/NVPTX.td.
defvar NVVMBaseSMVersions = [20, 21, 30, 32, 35, 37, 50, 52, 53, 60,
                             61, 62, 70, 72, 75, 80, 86, 87, 88, 89,
                             90, 100, 103, 107, 110, 120, 121];
defvar NVVMSMFamilies = [[100, 103, 107], [110], [120, 121]];

class NVVMArch<list<NVVMArch> directlyImplies> {
  assert !eq(!substr(NAME, 0, 2), "SM"),
         "architecture record " # NAME # " must be named SM<version>";

  string gpuKind = "::llvm::NVPTX::GK_SM_" # !substr(NAME, 2) #
                   !if(!eq(NAME, "SM32"), "_", "");

  list<NVVMArch> implies =
      !foldl(directlyImplies, directlyImplies, acc, arch,
             !listconcat(acc, !listremove(arch.implies, acc)));
}

foreach sm = NVVMBaseSMVersions in
  def SM#sm : NVVMArch<!foreach(earlier,
                                !filter(v, NVVMBaseSMVersions, !lt(v, sm)),
                                !cast<NVVMArch>("SM"#earlier))>;

multiclass NVVMArchVariants<list<NVVMArch> earlierInFamily> {
  def f : NVVMArch<[!cast<NVVMArch>(NAME)] # earlierInFamily>;
  def a : NVVMArch<[!cast<NVVMArch>(NAME # "f")]>;
}

foreach family = NVVMSMFamilies in
  foreach sm = family in
    defm SM#sm : NVVMArchVariants<
        !foreach(earlier, !filter(v, family, !lt(v, sm)),
                 !cast<NVVMArch>("SM" # earlier # "f"))>;

// sm_90 has no 'f' variant.
def SM90a : NVVMArch<[SM90]>;

defvar NVVMArchs = !instances<NVVMArch>();

// The op is supported on every target that is, or implies, one of `archs`.
class NVVMRequiresSM<list<NVVMArch> archs> :
  ParamNativeOpTrait<"NVVMRequiresSM",
    !interleave(
      !foreach(target,
               // Keeps `target` if `archs` holds it or anything it implies.
               !filter(target, NVVMArchs,
                       !lt(!size(!listremove(archs, [target] # target.implies)),
                           !size(archs))),
               target.gpuKind),
      ", ")> {
  assert !not(!empty(archs)),
         "NVVMRequiresSM requires at least one architecture";
}

#endif //NVVM_REQUIRES_SM_TRAITS
