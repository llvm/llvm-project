//===- AArch64VectorIntrinsicMappings.td - NEON to SVE Mappings -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the mapping from NEON intrinsics to their SVE equivalents
// for re-vectorization purposes.
//
// Note: SearchableTable.td is already included by AArch64SystemOperands.td
//===----------------------------------------------------------------------===//

// Argument mapping specification
// Encodes how to map NEON arguments to SVE arguments
// Format: Comma-separated specifiers where:
//   'T' = True predicate, 'Z' = Zero, 'P' = poison
//   'A:N' = Argument N from NEON
//   'I:N' = Scalarise a constant splat into an immediate

// Base class for intrinsic re-vectorization mappings
class IntrinsicMapping<string neon, string sve, string argMappings> {
  string NEONIntrinsic = neon;
  string SVEIntrinsic = sve;
  string ArgMappings = argMappings;
}
class CustomIntrinsicMapping<string neon> : IntrinsicMapping<neon, "", ""> {}

// Direct mappings - NEON arguments pass through directly to SVE
// udot/sdot: SVE(arg0, arg1, arg2) = NEON(arg0, arg1, arg2)
def : IntrinsicMapping<"llvm.aarch64.neon.udot", "llvm.aarch64.sve.udot",
                       "A:0,A:1,A:2">;
def : IntrinsicMapping<"llvm.aarch64.neon.sdot", "llvm.aarch64.sve.sdot",
                       "A:0,A:1,A:2">;
def : IntrinsicMapping<"llvm.aarch64.neon.usdot", "llvm.aarch64.sve.usdot",
                       "A:0,A:1,A:2">;

// NEON -> SVE ABD (absolute difference). Requires an extra predicate.
// uabd/sabd: SVE(pred_true, arg0, arg1) = NEON(arg0, arg1)
def : IntrinsicMapping<"llvm.aarch64.neon.uabd", "llvm.aarch64.sve.uabd",
                       "T,A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.sabd", "llvm.aarch64.sve.sabd",
                       "T,A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.fabd", "llvm.aarch64.sve.fabd",
                       "T,A:0,A:1">;

// NEON -> SVE abs/sqabs. Requires an extra predicate and twice the Z input.
// Note that NEON has no fabs.
def : IntrinsicMapping<"llvm.aarch64.neon.abs", "llvm.aarch64.sve.abs",
                       "A:0,T,A:0">;
def : IntrinsicMapping<"llvm.aarch64.neon.sqabs", "llvm.aarch64.sve.sqabs",
                       "A:0,T,A:0">;

// NEON -> SVE FACGE/FACGT (compare absolute value). Requires an extra predicate.
def : IntrinsicMapping<"llvm.aarch64.neon.facge", "llvm.aarch64.sve.facge",
                       "T,A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.facgt", "llvm.aarch64.sve.facgt",
                       "T,A:0,A:1">;

// NEON -> FCVTZ
// AFAIU, SVE has no "round to nearest" or "round towards +inf" instruction
def : IntrinsicMapping<"llvm.aarch64.neon.fcvtzu", "llvm.aarch64.sve.fcvtzu",
                       "P,T,A:0">;
def : IntrinsicMapping<"llvm.aarch64.neon.fcvtzs", "llvm.aarch64.sve.fcvtzs",
                       "P,T,A:0">;

// NEON 8.5a -> SVE FRINT32
// Not clear to me how those instructions differ from the normal FRINTX/FRINTZ?
def : IntrinsicMapping<"llvm.aarch64.neon.frint32x", "llvm.aarch64.sve.frintx",
                       "P,T,A:0">;
def : IntrinsicMapping<"llvm.aarch64.neon.frint32z", "llvm.aarch64.sve.frintz",
                       "P,T,A:0">;

// NEON -> SVE FRECP/FRSQRT
def : IntrinsicMapping<"llvm.aarch64.neon.frecpe", "llvm.aarch64.sve.frecpe.x",
                       "A:0">;
def : IntrinsicMapping<"llvm.aarch64.neon.frecps", "llvm.aarch64.sve.frecps.x",
                       "A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.frsqrte", "llvm.aarch64.sve.frsqrte.x",
                       "A:0">;
def : IntrinsicMapping<"llvm.aarch64.neon.frsqrts", "llvm.aarch64.sve.frsqrts.x",
                       "A:0,A:1">;

// Accumulator mappings - Add true predicate and zero accumulator before NEON arguments
// uadalp/sadalp: SVE(pred_true, zero, arg0) = NEON(arg0)
def : IntrinsicMapping<"llvm.aarch64.neon.uaddlp", "llvm.aarch64.sve.uadalp",
                       "T,Z,A:0">;
def : IntrinsicMapping<"llvm.aarch64.neon.saddlp", "llvm.aarch64.sve.sadalp",
                       "T,Z,A:0">;

// NEON -> SVE hadd/hsub. Requires an extra predicate.
def : IntrinsicMapping<"llvm.aarch64.neon.uhadd", "llvm.aarch64.sve.uhadd",
                      "T,A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.shadd", "llvm.aarch64.sve.shadd",
                      "T,A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.uhsub", "llvm.aarch64.sve.uhsub",
                      "T,A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.shsub", "llvm.aarch64.sve.shsub",
                      "T,A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.urhadd", "llvm.aarch64.sve.urhadd",
                       "T,A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.srhadd", "llvm.aarch64.sve.srhadd",
                       "T,A:0,A:1">;

// NEON -> SVE qadd/qsub. Requires an extra predicate.
def : IntrinsicMapping<"llvm.aarch64.neon.uqadd", "llvm.aarch64.sve.uqadd",
                       "T,A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.sqadd", "llvm.aarch64.sve.sqadd",
                       "T,A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.uqsub", "llvm.aarch64.sve.uqsub",
                       "T,A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.sqsub", "llvm.aarch64.sve.sqsub",
                       "T,A:0,A:1">;

// NEON -> SVE qdmul/qrdmul. Direct mapping.
def : IntrinsicMapping<"llvm.aarch64.neon.sqdmulh", "llvm.aarch64.sve.sqdmulh",
                       "A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.sqrdmulh", "llvm.aarch64.sve.sqrdmulh",
                       "A:0,A:1">;

// Shift left (saturating, rounding, both)
def : IntrinsicMapping<"llvm.aarch64.neon.sqshl", "llvm.aarch64.sve.sqshl",
                       "T,A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.uqshl", "llvm.aarch64.sve.uqshl",
                       "T,A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.srshl", "llvm.aarch64.sve.srshl",
                       "T,A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.urshl", "llvm.aarch64.sve.urshl",
                       "T,A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.sqrshl", "llvm.aarch64.sve.sqrshl",
                       "T,A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.uqrshl", "llvm.aarch64.sve.uqrshl",
                       "T,A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.sqshlu", "llvm.aarch64.sve.sqshlu",
                       "T,A:0,I:1">;

// NEON's bidirectional shifts like neon.sshl or neon.ushl cannot be rewritten for
// SVE directly because their equivalents always have rounding/saturation.
def : CustomIntrinsicMapping<"llvm.aarch64.neon.sshl">;
def : CustomIntrinsicMapping<"llvm.aarch64.neon.ushl">;

// Shift right narrow
def : IntrinsicMapping<"llvm.aarch64.neon.rshrn", "llvm.aarch64.sve.rshrnb",
                       "A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.sqrshrn", "llvm.aarch64.sve.sqrshrnb",
                       "A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.sqrshrun", "llvm.aarch64.sve.sqrshrunb",
                       "A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.sqshrn", "llvm.aarch64.sve.sqshrnb",
                       "A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.sqshrun", "llvm.aarch64.sve.sqshrunb",
                       "A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.uqrshrn", "llvm.aarch64.sve.uqrshrnb",
                       "A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.uqshrn", "llvm.aarch64.sve.uqshrnb",
                       "A:0,A:1">;

// Shift and insert
def : IntrinsicMapping<"llvm.aarch64.neon.vsli", "llvm.aarch64.sve.sli",
                       "A:0,A:1,A:2">;
def : IntrinsicMapping<"llvm.aarch64.neon.vsri", "llvm.aarch64.sve.sri",
                       "A:0,A:1,A:2">;

// Other shift instructions that aren't considered:
// - widening shift left (SHLL, SHLL2, USHLL, USHLL2, SSHLL, SSHLL2): Intrinsics are not used
// - SRSHR, SRSRA, USHR, URSHR: No intrinsics

// SVE's MULL takes inputs where every second element is ignored whereas
// NEON's MULL ignores the hi part of the quad.
// This is compatible with the default handling for scaled 64-bit vectors: each element
// gets interleaved with poison values.
def : IntrinsicMapping<"llvm.aarch64.neon.smull", "llvm.aarch64.sve.smullb",
                       "A:0,A:1">;
def : IntrinsicMapping<"llvm.aarch64.neon.umull", "llvm.aarch64.sve.umullb",
                       "A:0,A:1">;

// SVE's ADDP add pairs from src0 into even indices and pairs from src1 into odd indices.
// NEON's ADDP concatenates sources and inserts results "in order" in the result quad/double.
def : CustomIntrinsicMapping<"llvm.aarch64.neon.addp">;
def : CustomIntrinsicMapping<"llvm.aarch64.neon.faddp">;

// i8 4x4 MMLA (Matrix multiply accumulate)
// (Requires SVE+i8mm, i8mm is already required by NEON's MMLA so there's nothing extra to check)
// Note that NEON has no fmmla but SVE has.
def : IntrinsicMapping<"llvm.aarch64.neon.ummla", "llvm.aarch64.sve.ummla",
                       "A:0,A:1,A:2">;
def : IntrinsicMapping<"llvm.aarch64.neon.smmla", "llvm.aarch64.sve.smmla",
                       "A:0,A:1,A:2">;
def : IntrinsicMapping<"llvm.aarch64.neon.usmmla", "llvm.aarch64.sve.usmmla",
                       "A:0,A:1,A:2">;

// Extract narrow
def : IntrinsicMapping<"llvm.aarch64.neon.sqxtn", "llvm.aarch64.sve.sqxtnb",
                       "A:0">;
def : IntrinsicMapping<"llvm.aarch64.neon.sqxtun", "llvm.aarch64.sve.sqxtunb",
                       "A:0">;
def : IntrinsicMapping<"llvm.aarch64.neon.uqxtn", "llvm.aarch64.sve.uqxtnb",
                       "A:0">;
def : IntrinsicMapping<"llvm.aarch64.neon.fcvtxn", "llvm.aarch64.sve.fcvtx.f32f64",
                       "P,T,A:0">;

// NEON's tbl1 can be remapped directly to SVE's tblq, but tbl[2-4] require
// extra tbxq instruction(s)
def : CustomIntrinsicMapping<"llvm.aarch64.neon.tbl1">;
def : CustomIntrinsicMapping<"llvm.aarch64.neon.tbl2">;
def : CustomIntrinsicMapping<"llvm.aarch64.neon.tbl3">;
def : CustomIntrinsicMapping<"llvm.aarch64.neon.tbl4">;

// Searchable table definition
def IntrinsicMappingTable : GenericTable {
  let FilterClass = "IntrinsicMapping";
  let CppTypeName = "NEONToSVEIntrinsicMapping";
  let Fields = ["NEONIntrinsic", "SVEIntrinsic", "ArgMappings"];
}

// Search index for looking up by NEON intrinsic name
def lookupNEONToSVEMappingByNEONIntrinsic : SearchIndex {
  let Table = IntrinsicMappingTable;
  let Key = ["NEONIntrinsic"];
}
