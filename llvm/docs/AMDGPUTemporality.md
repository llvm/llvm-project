(amdgpu-temporality)=

# AMDGPU Temporality

## Overview

The Temporality metadata describes the expected reuse pattern of the accessed
memory. It generalizes the existing `!nontemporal` metadata, a single boolean
applied uniformly across all scopes, into a separate hint for each scope.

Out of scope for Temporality:

`!amdgpu.last.use`
: Last-use semantics are a separate hint and are unchanged.

`volatile`
: Volatile semantics are orthogonal and governed by the `volatile` keyword.

Cache names
: Temporality never names a physical cache ("L1", "L2", "MALL"); only the
  architecture-neutral scopes in {ref}`amdgpu-temporality-grammar` are
  accepted.

### Applicability

Temporality applies to:

- `load` and `store` instructions, including `load atomic` and `store atomic`.
- `atomicrmw` and `cmpxchg` instructions.
- intrinsics that accept an explicit temporality argument.

On `atomicrmw` and `cmpxchg` the hint is currently dropped during code
generation, as `!nontemporal` is.

(amdgpu-temporality-levels)=

## Levels

| Level | Meaning |
|---|---|
| `low` | Low expected reuse. Prefer eviction / streaming behavior. |
| `regular` | Default temporal behavior. |
| `high` | High expected reuse. Prefer retention. |

Levels are ordered `low` < `regular` < `high`. A higher level asks for more
retention.

(amdgpu-temporality-syntax)=

## Syntax

`!mem.cache_hint` is an existing metadata node carrying a generic list of
key-value hints, each associated with a pointer operand. Each key is a string,
while its value may be a string or an integer. Keys use the target-prefixed
form `<target>.<property>`; therefore, the temporality key is
`amdgpu.temporality` and not a bare name.

```llvm
%v = load i32, ptr addrspace(1) %p, align 4, !mem.cache_hint !0

!0 = !{ i32 0, !1 }
!1 = !{ !"amdgpu.temporality", !"workgroup:low:agent:high" }
```

`!1` is a hint node holding one key-value pair, the key `"amdgpu.temporality"`
and the value `"workgroup:low:agent:high"`. `!0` attaches it to operand 0, the
pointer operand of the `load`.

This structure is fixed boilerplate present whenever the hint applies and is
defined by `!mem.cache_hint` itself, not by this document. Temporality defines
only `<temporality-string>`, the value of the `"amdgpu.temporality"` key. The
remainder of this section specifies that string's grammar. Later examples omit
the `!mem.cache_hint`/`operand_no` wrapper and show only the key/value pair for
brevity.

(amdgpu-temporality-grammar)=

### Grammar

```
temporality-string ::= level (':' entry)*
                      | entry (':' entry)*

entry              ::= scope ':' level

level              ::= "low" | "regular" | "high"

scope              ::= "singlethread" | "wavefront" | "workgroup"
                      | "cluster"     | "agent"     | "system"
```

The scopes are the AMDGPU memory scopes defined in {ref}`amdgpu-scopes`, listed
above from smallest to largest.

### Unscoped form

A single level with no scope. It applies that level at every scope:

```llvm
!1 = !{ !"amdgpu.temporality", !"low" }
```

### General form

One or more entries, each pairing a scope with a level. The level becomes
effective at that scope and remains in effect at larger scopes until a larger
entry overrides it:

```llvm
!1 = !{ !"amdgpu.temporality", !"singlethread:regular:workgroup:low" }
```

Entries may be listed in any order, and a scope may appear in at most one
entry.

A string's **profile** is the entries it lists. The unscoped form can be
rewritten in the general form by prefixing the single entry with
`singlethread:`.

A string may also start with a level and continue with entries. The leading
level is an implicit `singlethread` entry; for example, `low:agent:high` can be
expanded as `singlethread:low:agent:high`.

## Semantics

(amdgpu-temporality-effective-level)=

### Effective level

In general, the effective level at any scope `S` is the level named by the
entry with the largest scope that is still less than or equal to `S`. If no
such level can be inferred, then the effective level at `S` defaults to
`regular`, matching the default used when temporality is absent from the
instruction.

```llvm
!"amdgpu.temporality", !"singlethread:low:cluster:high"
```

| Scope | Effective level |
|---|---|
| singlethread | low |
| wavefront | low |
| workgroup | low |
| cluster | high |
| agent | high |
| system | high |

Listing the effective level at every scope, as the table does, gives the
string's **enumerated form**.

A profile's levels are not required to be monotonic. An entry may name a lower
or a higher level than the preceding entry:

```llvm
!"amdgpu.temporality", !"singlethread:high:wavefront:low:agent:regular:system:low"
```

This profile decreases from high to low at wavefront, rises to regular at
agent, and then decreases again to low at system. These three transitions occur
in two directions within a single node.

(amdgpu-temporality-unsupported-scopes)=

### Unsupported Scopes

The enumerated form has one entry per scope, and a target reads only the
entries for the scopes it implements. On AMDGPU only `cluster` can be missing.

(amdgpu-temporality-lowering)=

## Hardware Lowering (AMDGPU)

### Pre-GFX12

Pre-GFX12 cache policy is represented as a binary temporal or non-temporal
choice. The non-temporal setting corresponds to the bits already emitted for
`!nontemporal` and applies uniformly across all relevant caches. These caches
are associated with three architectural scopes: L0/L1 at workgroup, L2 at
agent, and, on GFX11, MALL at system.

The non-temporal setting is enabled if and only if every specified scope among
`workgroup`, `agent`, and `system` specifies `low`.

When these scopes disagree, a single setting cannot follow both, and the rule
keeps data cached rather than streaming it.

(amdgpu-temporality-gfx12)=

### GFX12 and later

GFX12 divides the caches into two groups and gives the temporal-hint (TH) field
one value per group:

```
near  ->  CU cache (workgroup), SE cache (cluster)
far   ->  MALL (agent, system)
```

Each value resolves to the highest level ({ref}`amdgpu-temporality-levels`)
among the scopes of the caches that it covers.

```
near (CU / SE, on-chip) = max(level at workgroup, level at cluster)
far  (MALL, device-shared)= max(level at agent, level at system)
```

GFX12 has no cache private to a wavefront or a thread. The innermost is the
CU's, shared by every wave on it, so the formulas name no smaller scope.

The maximum is used because one value is applied to all the caches in
`near`/`far`. Choosing the highest value guarantees that no requested retention
is discarded. At worst, data is retained longer than requested, incurring a
cache-capacity cost.

Encoding is a separate step that runs once both maxima are known. The hardware
exposes only six encodable pairs. Lookup is therefore defined over a normalized
pair (`near'`, `far'`).

Normalization is applied before table lookup. If the computed pair is not
directly encodable, the `near` component has higher precedence over the `far`
component: `(high, regular)` and `(high, low)` normalize to `(high, high)`,
while `(regular, high)` normalizes to `(regular, regular)`.

| Computed near | Computed far | Lookup pair | Encoding |
|---|---|---|---|
| low | low | (low, low) | `TH_NT` |
| regular | regular | (regular, regular) | `TH_RT` (default) |
| high | low / regular / high | (high, high) | `TH_HT` |
| low | regular | (low, regular) | `TH_NT_RT` |
| regular | low | (regular, low) | `TH_RT_NT` |
| low | high | (low, high) | `TH_NT_HT` |
| regular | high | (regular, regular) | `TH_RT` |


The hardware defines eight `TH_*` values in total. The two not listed above are
value 3 (`TH_LU` on loads, `TH_WB` on stores) and value 7 (`TH_NT_WB` on
stores). Temporality alone never targets them; the two write-back encodings are
reached only through the `amdgpu.write_back` key ({ref}`amdgpu-write-back`).

#### SMEM (scalar loads)

SMEM (scalar-load) instructions use a narrower encoding than the vector memory
instructions covered above: a 2-bit TH field with only three usable values
(`TH_NT`, `TH_RT`, and `TH_HT`) and no near/far split.

With no split there is no pair to make encodable, so the rule above does not
apply here. A single value covers every cache, which makes this the maximum
rule over one group spanning all of them:

```
level = max(level at workgroup, cluster, agent, system)
```

## Interaction with `!nontemporal` and `aux`

### Interaction with `!nontemporal`

`!nontemporal` remains valid and unchanged. It is equivalent to the unscoped
temporality string `"low"`:

```llvm
%v = load i32, ptr %p, !nontemporal !{i32 1}

%v = load i32, ptr %p, !mem.cache_hint !0     ; identical meaning
!0 = !{ i32 0, !1 }
!1 = !{ !"amdgpu.temporality", !"low" }
```

When an instruction carries both `!nontemporal` and a temporality hint,
Temporality takes precedence.

| `!nontemporal` | Temporality | Result |
|---|---|---|
| absent | absent | default (`regular`) behavior |
| present | absent | `!nontemporal` applies, unchanged |
| present | present | Temporality applies; `!nontemporal` is ignored |

### Interaction with `aux`

Some intrinsics accept a separate explicit `i32 cachepolicy` ("aux") argument
encoding cache-policy bits directly. Temporality is defined for this set of
intrinsics. When an instruction carries both a non-default `aux` value and a
temporality hint, Temporality takes precedence and `aux` is ignored. Only the
temporal bits of `aux` are ignored, its other bits such as scope are kept.

```llvm
%v = call float @llvm.amdgcn.raw.ptr.buffer.load(
    ptr addrspace(8) %rsrc, i32 %offset, i32 %soffset, i32 2),
    !mem.cache_hint !0
!0 = !{ i32 0, !1 }
!1 = !{ !"amdgpu.temporality", !"low" }
```

The `aux` argument (`i32 2` above) encodes a cache policy in its own terms,
possibly the opposite of `low`. It is discarded, not merged;
`amdgpu.temporality`'s `"low"` applies instead.

| `aux` | Temporality | Result |
|---|---|---|
| absent / default | absent | default (`regular`) behavior |
| present | absent | `aux` applies, unchanged |
| present | present | Temporality applies; `aux` is ignored |

(amdgpu-write-back)=

## Write-back

`amdgpu.write_back` is a hint for stores. It asks the caches at `agent` and
`system` scope to keep the stored line dirty instead of writing it through to
memory early.

Temporality controls how long a cache keeps a line; write-back controls whether
that line may stay dirty. The two are independent, so write-back is a separate
key. Like temporality, it is a performance hint and may be ignored.

### Syntax

`amdgpu.write_back` goes in the same hint node as `amdgpu.temporality`. Its
value is an `i1` constant; `true` requests write-back and `false` has no effect.

```llvm
store i32 %v, ptr addrspace(1) %p, !mem.cache_hint !0
!0 = !{ i32 1, !1 }
!1 = !{ !"amdgpu.temporality", !"workgroup:low:agent:high",
        !"amdgpu.write_back", i1 true }
```

### Applicability

Write-back applies to `store` instructions, including `store atomic`, and to
store intrinsics. It is ignored on all other instructions.

### Hardware Lowering

Pre-GFX12 targets have no write-back encoding, so the key is ignored.

On GFX12 and later, the encoding depends on the `near` and `far` levels
computed in {ref}`amdgpu-temporality-gfx12`. Only two pairs have a write-back
encoding:

| Computed near | Computed far | Encoding |
|---|---|---|
| low | high | `TH_NT_WB` |
| regular | high | `TH_WB` |

For any other pair the key has no effect, and the store is encoded as if the
key were absent. For example, `(high, high)` stays `TH_HT`, because no
write-back encoding keeps `near` at `high`.

### Examples

```llvm
!1 = !{ !"amdgpu.temporality", !"workgroup:low:agent:high",
        !"amdgpu.write_back", i1 true }
; near = low, far = high          GFX12: TH_NT_WB (TH_NT_HT without the key)

!2 = !{ !"amdgpu.temporality", !"agent:high", !"amdgpu.write_back", i1 true }
; near = regular, far = high      GFX12: TH_WB    (TH_RT without the key)

!3 = !{ !"amdgpu.temporality", !"high", !"amdgpu.write_back", i1 true }
; near = high, far = high         GFX12: TH_HT    (key ignored: no write-back
;                                                  encoding has near at high)
```

## Diagnostics

The AMDGPU backend performs the following checks at codegen time, not during
`opt -verify`. The generic `!mem.cache_hint` verifier only confirms that the
`"amdgpu.temporality"` value is a string
({ref}`amdgpu-temporality-grammar`). A malformed `<temporality-string>` is
therefore caught only when codegen for this target reads the value.

| Condition | Result |
|---|---|
| String does not match the grammar in {ref}`amdgpu-temporality-grammar`, including a missing or extra level or scope, an unrecognized level or scope, or a level and scope in the wrong order | Error |
| The same scope is named by more than one entry, even if the levels are identical | Error |
| Entries are listed out of scope order | Not an error; entry order does not matter |
| Levels vary non-monotonically across entries | Not an error |
| Both `!nontemporal` / `aux` and a temporality hint are present on the same instruction | Not an error; Temporality applies, and the other hint is ignored |
| A scope is not implemented on the target | Not an error; the level at that scope is ignored |
| The `amdgpu.write_back` value is not an `i1` constant | Error |
| `amdgpu.write_back` is on a load, or the computed pair has no write-back encoding | Not an error; the key is ignored |

```llvm
; Valid values of "amdgpu.temporality"
!"low"                                ; unscoped level, no scope qualifier, applies everywhere
!"workgroup:low"                      ; singlethread and wavefront default to regular
!"singlethread:low:cluster:high"      ; general form with two entries
!"cluster:high:singlethread:low"      ; same as above, entry order is irrelevant

; Invalid values of "amdgpu.temporality"
!""                                   ; empty string, satisfies neither grammar alternative
!"normal"                             ; "normal" is not a level
!"LOW"                                ; levels are lowercase
!"low:high"                           ; after the leading level, high is not an entry
!"workgroup:low:high"                 ; the trailing high has no scope, so it is not an entry
!"workgroup:low:workgroup:high"       ; workgroup appears in two entries, duplicate scope
```

## Examples

### Unscoped, low everywhere (equivalent to `!nontemporal`)

```llvm
load i32, ptr %p, !mem.cache_hint !0
!0 = !{ i32 0, !1 }
!1 = !{ !"amdgpu.temporality", !"low" }
; GFX12: TH_NT      Pre-GFX12: non-temporal
```

### Unscoped, high everywhere

```llvm
load i32, ptr %p, !mem.cache_hint !0
!0 = !{ i32 0, !1 }
!1 = !{ !"amdgpu.temporality", !"high" }
; GFX12: TH_HT      Pre-GFX12: no bits (high has no pre-GFX12 encoding)
```

(amdgpu-temporality-example-dual-level)=

### Dual-level: evict near, retain far

```llvm
load i32, ptr %p, !mem.cache_hint !0
!0 = !{ i32 0, !1 }
!1 = !{ !"amdgpu.temporality", !"singlethread:low:agent:high" }
; near = max(workgroup, cluster) = low    far = max(agent, system) = high
; GFX12: TH_NT_HT
; Pre-GFX12: no bits (agent and system are high)
```

### Low from one scope upward

```llvm
load i32, ptr %p, !mem.cache_hint !0
!0 = !{ i32 0, !1 }
!1 = !{ !"amdgpu.temporality", !"workgroup:low" }
; workgroup: low (named)      cluster, agent, system: low (applied upward)
; near = max(low, low) = low      far = max(low, low) = low
; GFX12: TH_NT
; Pre-GFX12: non-temporal (workgroup, agent and system are all low)
```

The `low` is named at `workgroup`, the smallest scope of the near group, so it
covers both `workgroup` and `cluster` and reaches `near`. `singlethread` and
`wavefront` also resolve to `regular` here, but neither belongs to a group, so
neither affects the result.

### A low that does not reach its group's smallest scope

Both profiles below request `low` from `cluster` upward, but they differ below
it: `"cluster:low"` leaves `workgroup` at the default `regular`, while
`"workgroup:low"` names `low` there. Because `near` is the maximum over
`workgroup` and `cluster`, that one scope changes the encoding.

```llvm
!1 = !{ !"amdgpu.temporality", !"cluster:low" }
; workgroup is not named, so it resolves to regular and dominates the near maximum
; near = max(regular, low) = regular      far = max(low, low) = low
; GFX12: TH_RT_NT     Pre-GFX12: non-temporal

!2 = !{ !"amdgpu.temporality", !"workgroup:low" }
; the low now covers both scopes of the near group
; near = max(low, low) = low              far = max(low, low) = low
; GFX12: TH_NT        Pre-GFX12: non-temporal
```

### A request naming only far scopes

```llvm
load i32, ptr %p, !mem.cache_hint !0
!0 = !{ i32 0, !1 }
!1 = !{ !"amdgpu.temporality", !"agent:high" }
; workgroup and cluster are not named, so both resolve to regular
; near = regular, far = high: not directly encodable, so far moves to regular
; GFX12: TH_RT (the default encoding; this profile has no effect there)
; Pre-GFX12: no bits
```

To retain in the MALL while streaming on chip, name a near scope as well:
`"workgroup:low:agent:high"` encodes as `TH_NT_HT`, as in
{ref}`amdgpu-temporality-example-dual-level`.
