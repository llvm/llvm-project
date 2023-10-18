//===-- PartitionAttrDefs.td - attributes definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PARTITION_ATTRDEFS
#define PARTITION_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/Partition/IR/PartitionBase.td"
include "mlir/IR/TensorEncoding.td"

// All of the partition tensor attributes will extend this class.
class Partition_Attr<string name,
                        list<Trait> traits = []>
    : AttrDef<Partition_Dialect, name, traits>;

// Partition attribute that will always return 1

def PartitionEncodingAttr : Partition_Attr<"PartitionEncoding", []> {
	let mnemonic = "encoding";
	let description = [{
		an attribute to hsow if there is partition or not
	}];
	let parameters = (
		ins
		"unsigned":$partConst
	);
	let genVerifyDecl = 1;
	let assemblyFormat = "`<` `{` `partConst` `=` $partConst `}` `>`";
}

#endif // PARTITION_ATTRDEFS
