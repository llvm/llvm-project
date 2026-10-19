//===-- AGXCallingConv.td - AGX calling conventions --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AGX kernels receive their arguments in uniform registers. The layout is
// computed by AGXTargetLowering::LowerFormalArguments (natural alignment in
// 16-bit units, see AGXUsage.rst), not by a CCState, so only the callee saved
// register list is described here. There are no callable functions: all calls
// are inlined before code generation.
//
//===----------------------------------------------------------------------===//

def CSR_AGX_NoRegs : CalleeSavedRegs<(add)>;
