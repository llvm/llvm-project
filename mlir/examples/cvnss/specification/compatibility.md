# CVNSS-CIR cross-language strategy

Contract 0.2: registered cvnss.reference operation, SHA-256 source provenance,
versioned reference IDs and language/kind validation. Reference identity does
not imply semantic equivalence. Existing pack27/unpack27/mapping remain stable.

Levels:
L0 opaque original bytes; L1 language AST/IR; L2 semantic references;
L3 typed domain operations; L4 verified executable subset; L5 ABI-specific target.
No mandatory lowering of metadata or opaque blobs.

Compatibility matrix:
| Language | Implemented | Planned |
|---|---|---|
| BilaScript | explicit manifest | parser/AST emitter |
| C/C++ | explicit manifest | ClangIR native pass |
| JavaScript | explicit manifest | JSIR import/export |
| Rust | explicit manifest | Melior and MIR adapter |
| Python | static top-level AST extraction | xDSL dialect bridge |

Acceptance: preserve source bytes externally; reject unsupported versions;
test deterministic IDs, invalid inputs, non-ASCII strings, native MLIR parser,
and later separately test runtime ABI, round-trip and semantic equivalence.
No claims of complete conversion or hardware performance.
