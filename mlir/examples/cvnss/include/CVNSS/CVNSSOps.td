// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef CVNSS_OPS
#define CVNSS_OPS
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def CVNSS_Dialect : Dialect {
  let name = "cvnss";
  let cppNamespace = "::mlir::cvnss";
  let summary = "Verified symbolic and balanced ternary representation";
}
class CVNSS_Op<string mnemonic, list<Trait> traits = []>
    : Op<CVNSS_Dialect, mnemonic, traits>;

// ID = 13 + 9*hi + 3*mid + lo. Each trit is in [-1,1].
def CVNSS_Pack27Op : CVNSS_Op<"pack27", [Pure]> {
  let summary = "Pack three constant balanced trits into a Base27 ID";
  let arguments = (ins I32Attr:$hi, I32Attr:$mid, I32Attr:$lo);
  let results = (outs I32:$id);
  let hasVerifier = 1;
}

def CVNSS_Unpack27Op : CVNSS_Op<"unpack27", [Pure]> {
  let summary = "Unpack a constant ID into three balanced trits";
  let arguments = (ins I32Attr:$id);
  let results = (outs I32:$hi, I32:$mid, I32:$lo);
  let hasVerifier = 1;
}

// An explicit audit record, not a text encoder or a proof of reversibility.
def CVNSS_MappingOp : CVNSS_Op<"mapping"> {
  let summary = "Preserve a versioned external symbolic mapping record";
  let arguments = (ins StrAttr:$source, StrAttr:$target,
                       StrAttr:$mode, StrAttr:$mapping_version,
                       StrAttr:$loss_policy);
  let hasVerifier = 1;
}
// Cross-language semantic annotation; does not execute foreign code.
def CVNSS_ReferenceOp : CVNSS_Op<"reference"> {
  let summary = "Versioned semantic reference with source provenance";
  let arguments = (ins StrAttr:$reference_id, StrAttr:$language,
    StrAttr:$source_uri, StrAttr:$source_digest, StrAttr:$symbol,
    StrAttr:$semantic_kind, StrAttr:$schema_version);
  let hasVerifier = 1;
}
#endif
