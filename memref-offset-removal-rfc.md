# RFC: Drop static offset from MemRefType, keep it in ABI and ops

## Status

Draft. Builds on prior discussions:
- [RFC: Removing offset from MemRef Type and Lowering](https://discourse.llvm.org/t/rfc-removing-offset-from-memref-type-and-lowering/82963)
- [RFC: ContiguousLayoutAttr and changing default memref layout](https://discourse.llvm.org/t/rfc-contiguous-permutation-offset-o-layout-and-changing-default-memref-layout/85284)

## Summary

Remove the static offset from `StridedLayoutAttr` (and therefore from `MemRefType`).
Keep offset as a first-class operand/result on `memref.reinterpret_cast`,
`memref.extract_strided_metadata`, and friends. The type system stops carrying
offset information; ops still talk about offsets; lowerings decide what offset
semantics mean at the ABI level.

This is a smaller-blast-radius subset of the original "remove offset
everywhere" proposal: the runtime descriptor keeps the offset slot by default,
so existing lowerings remain bit-identical in behavior.

## Motivation

The static offset slot in `StridedLayoutAttr` has not earned its keep:

- It conflates IR-level shape information with ABI/lowering decisions, leaking
  implementation details into the type system.
- Most `subview` / `reinterpret_cast` chains produce dynamic offsets in
  practice; the static slot is rarely populated meaningfully.
- The "more static offset blocks fold" guard in `canFoldIntoConsumerOp` only
  exists to prevent casts from inventing offset information. Removing the
  source of those lies removes the need for the guard.
- Alternative lowerings (no-offset descriptors, fat pointers) are awkward to
  support while the type insists on a single offset model.
- The original author of the offset mechanism has acknowledged that the
  expected benefits did not materialize (see linked RFC).

## Proposal

### Type level

- Drop the `offset` parameter from `StridedLayoutAttr`. Equivalently: treat
  it as always `ShapedType::kDynamic` and remove the field.
- `MemRefType` no longer carries any static offset information.
- Printer: always omit the `offset:` clause.
- Parser: accept the legacy form for one release for migration ease, then
  remove.

### Op level

Operations keep offset as an explicit IR value:

- `memref.reinterpret_cast` continues to accept an `offset` operand.
  Semantically: "produce a memref view starting at base + offset".
- `memref.extract_strided_metadata` continues to return an `offset` SSA
  value. Semantically: "give me the offset that the lowering commits to".
- `memref.subview` is unchanged at the op level; offset operand remains.

The contract is: offset is a first-class value at the IR level, decoupled
from the type.

### Lowering strategies

Because offset lives on the op, not the type, lowerings can choose freely:

1. **Current descriptor lowering (default).** Keeps the offset slot in the
   LLVM struct. `reinterpret_cast` writes offset to the struct;
   `extract_strided_metadata` reads it. Behavior identical to today.

2. **No-offset lowering.** Collapses offset into the data pointer at
   lowering time:
   - `reinterpret_cast` with non-zero offset emits a GEP immediately; the
     descriptor stores `base + offset`, with no separate offset field.
   - `extract_strided_metadata` returns a constant 0; downstream DCE
     removes any arithmetic on it.
   - LLVM struct loses the offset member.

3. **Fat-pointer lowering.** GEP on the pointer half of the fat pointer;
   descriptor metadata unchanged.

This factoring makes lowering choice an ABI/codegen decision rather than a
type-system commitment.

### Folding and canonicalization

- Delete the "more static offset blocks fold" guard in
  `canFoldIntoConsumerOp` (`mlir/lib/Dialect/MemRef/IR/MemRefOps.cpp`).
  It guards against lies that can no longer be told.
- Delete `offset == 0` fast paths in Vector, SparseTensor, and
  MemRefToLLVM. They exploit information the type no longer carries.
- Folds that currently constant-propagate offsets through
  `reinterpret_cast` / `extract_strided_metadata` move from IR-level
  canonicalization to post-lowering peephole patterns. Pre-lowering, the
  offset is always conservatively dynamic.
- Rename or remove `hasStaticLayout()` (currently "all strides static AND
  offset static"); collapse to "all strides static" or drop entirely.

### API surface

The helper `getStridesAndOffset()` becomes misleading: with no static offset
on the type, the offset out-param is always `ShapedType::kDynamic` and every
caller has to plumb it through and ignore it.

- Rename `getStridesAndOffset()` to `getStrides()`. Keep it returning
  `LogicalResult` so it continues to act as the "is this layout
  strided-representable?" probe.
- Drop the offset out-param.
- Audit ~80 call sites; the rewrite is mechanical.

Edge case: affine-map layouts can in principle compute a static offset
even when `StridedLayoutAttr` cannot carry one. If any consumer relies on
that, expose it through a separate `getStaticOffsetIfAny()` returning
`std::optional<int64_t>` rather than keeping the offset glued to the
strides API. Likely no real consumers exist; verify by grep before
deleting outright.

## Migration plan

Order matters; each step is independently mergeable.

1. **Nuke offset-based folds first.** Keeps the IR sound while the rest of
   the work proceeds, and surfaces any hidden dependence on those folds
   before the type changes.
2. **Strip `offset` from `StridedLayoutAttr`.** Update printer/parser. Fix
   the stale `assert(offset == 0)` at
   `mlir/lib/Conversion/MemRefToLLVM/MemRefToLLVM.cpp:1828`.
3. **Mass-update tests.** Roughly 149 `.mlir` files, ~2348 occurrences.
   Mostly mechanical: `offset: N` becomes omitted.
4. **Audit `getStridesAndOffset()` call sites** (~80). Most already handle
   dynamic offset; a few need adjustment.
5. **Rename `getStridesAndOffset()` to `getStrides()`** and drop the
   offset out-param. Land as a single sweeping change once step 4 has
   identified all consumers.
6. **Optional follow-up.** Introduce a no-offset lowering pipeline option
   to validate the design end-to-end. Not required for the type-level
   change to land.

## Blast radius

- Tests: ~149 `.mlir` files updated (mostly scriptable).
- Code call sites: ~80 `getStridesAndOffset()` sites audited; ~10 fold and
  special-case sites materially changed.
- Lowering: default descriptor path unchanged in behavior. No-offset and
  fat-pointer paths become straightforward to add later.
- Verifier: no new constraints; some constraints removed.
- Estimated effort: 2 to 3 weeks for one experienced contributor.

## Alternatives considered

- **`ContiguousLayoutAttr` (Krzysz00).** Introduces a richer layout
  attribute that explicitly encodes permutations and offset, partially
  reclaiming optimization information that bare strides lose. Largely
  orthogonal to this proposal: this RFC removes offset from the static
  type encoding; `ContiguousLayoutAttr` enriches the dynamic layout
  vocabulary. Both can coexist.

- **Remove offset from the descriptor entirely (original RFC).** More
  invasive; conflicts with SPIR-V and other backends that cannot trivially
  perform pointer arithmetic on opaque pointers. This proposal is the
  smaller-blast-radius subset: keep ABI flexibility, remove only the
  type-level fiction.

- **Status quo with better folding hygiene.** Possible, but does not
  address the fundamental conflation of type and ABI concerns. The same
  bug class returns over time.

## Open questions

- Does `extract_strided_metadata` need an attribute or trait declaring its
  offset semantics for lowerings that disagree, or is "always
  conservatively dynamic pre-lowering" sufficient?
- Do downstream projects (IREE, Triton, others) materially depend on
  static offset propagation through subview chains? If yes, what is their
  migration path?
- Should `hasStaticLayout()` be removed outright or kept as a renamed
  shim?
- Should the parser keep accepting the legacy `offset: N` form for one
  release as a soft migration, or hard-cut?
- Do any in-tree or downstream consumers actually use static offsets
  derived from affine-map layouts via `getStridesAndOffset()`? If yes,
  introduce `getStaticOffsetIfAny()`; if no, drop the concept.

## Non-goals

- Changing the default lowering. Behavior of the existing descriptor
  lowering is preserved.
- Removing offset from the runtime ABI. Out of scope; covered by the
  original RFC if desired later.
- Introducing a new layout attribute. Compatible with, but independent
  of, `ContiguousLayoutAttr`.
