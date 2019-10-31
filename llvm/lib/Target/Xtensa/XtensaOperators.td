//===- XtensaOperators.td - Xtensa-specific operators ---------*- tblgen-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Node definitions
//===----------------------------------------------------------------------===//
def Xtensa_retflag: SDNode<"XtensaISD::RET_FLAG", SDTNone,
                                 [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
