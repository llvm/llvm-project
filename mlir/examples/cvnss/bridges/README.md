# CVNSS semantic bridge contract 0.2

The registered MLIR namespace remains cvnss. The hub is metadata-first;
it does not translate arbitrary foreign programs or infer Vietnamese semantics.
Reference IDs are deterministic SHA-256 hashes of a versioned JSON tuple:
(schema version, language, source URI, source SHA-256, symbol, kind).
They identify source revisions, not cross-language semantic equivalence.

## Working interface
Python AST extracts top-level function/class declarations without running code.
C, C++, JavaScript, Rust and BilaScript accept explicit symbol manifests:
```json
[{"symbol":"example","kind":"function"}]
```
```sh
python3 bridges/hub.py cpp example.cpp --symbols symbols.json > example.mlir
python3 bridges/hub.py python example.py > python.mlir
cvnss-opt example.mlir
```
The MLIR verifier checks structural validity, not the truth of external claims.
Original source bytes are not embedded. Retain the original file for round-trip.
Do not lower audit metadata into executable code; export it separately.

## Native bridge roadmap
- ClangIR: pin exact monorepo revision, register CIR, annotate selected supported ops;
  preserve ABI, exception behavior, qualifiers, lifetimes, and source locations.
- JSIR: pin external implementation; preserve AST/source trivia separately;
  do not equate JS dynamic semantics to C++ semantics.
- Rust: use Melior for MLIR interop; MIR extraction is compiler-version-specific.
  Pliron interchange requires explicit serialization and semantic checks.
- Python: use AST for static inspection; xDSL interchange requires matching
  operation definitions and textual/bytecode compatibility tests.
- ABI: start with C ABI fixed-width scalars and explicit ownership, errors and
  buffer lengths; runtime-specific values and exceptions cannot cross implicitly.
- Semantic vectors: future model/version/dimension registry; no fabricated vectors.
- Security: never run imported programs or trust declared symbol manifests.
