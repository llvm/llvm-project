# Key Instructions debug info in LLVM

Key Instructions reduces the jumpiness of optimized code debug stepping. This document explains the feature and how it is implemented in LLVM. For Clang support please see the Clang docs.

## Status

In development - some details may change with little notice.

Tell Clang [not] to produce Key Instructions metadata with `-g[no-]key-instructions`. See the Clang docs for implementation info.

Use LLVM flag `-dwarf-use-key-instructions` to interpret Key Instructions metadata when producing the DWARF line table (Clang passes the flag to LLVM). The behaviour of this flag may change.

The feature improves optimized code stepping; it's intended for the feature to be used with optimisations enabled. Although the feature works at O0 it is not recommended because in some cases the effect of editing variables may not always be immediately realised. (This is a quirk of the current implementation, rather than fundemental limitation, covered in more detail later).

There is currently no plan to support CodeView.

## Problem statement

A lot of the noise in stepping comes from code motion and instruction scheduling. Consider a long expression on a single line. It may involve multiple operations that optimisations move, re-order, and interleave with other instructions that have different line numbers.

DWARF provides a helpful tool the compiler can employ to mitigate this jumpiness, the is_stmt flag, which indicates that an instruction is a recommended breakpoint location. However, LLVM's current approach to deciding is_stmt placement essentially reduces down to "is the associated line number different to the previous instruction's?".

(Note: It's up to the debugger if it wants to interpret is_stmt or not, and at time of writing LLDB doesn't; possibly because LLVM's is_stmts convey no information that can't already be deduced from the rest of the line table.)

## Solution overview

Taking ideas from two papers [1][2] that explore the issue, especially C. Tice's:

From the perspective of a source-level debugger user:

* Source code is made up of interesting constructs; the level of granularity for “interesting” while stepping is typically assignments, calls, control flow. We’ll call these interesting constructs Atoms.

* Atoms usually have one instruction that implements the functionality that a user can observe; once they step “off” that instruction, the atom is finalised. We’ll call that a Key Instruction.

* Communicating where the key instructions are to the debugger (using DWARF’s is_stmt) avoids jumpiness introduced by scheduling non-key instructions without losing source attribution (because non-key instructions retain an associated source location, they’re just ignored for stepping).

## Solution implementation

1. `DILocation` has 2 new fields, `atomGroup` and `atomRank`.
2. Clang creates `DILocations` using the new fields to communicate which instructions are "interesting".
3. There’s some bookkeeping required by optimisations that duplicate control flow.
4. During DWARF emission, the new metadata is collected (linear scan over instructions) to decide is_stmt placements.

1. *The metadata* - The two new `DILocation` fields are `atomGroup` and `atomRank`. Both are unsigned integers. Instructions in the same function with the same `(atomGroup, inlinedAt)` pair are part of the same source atom. `atomRank` determines is_stmt preference within that group, where a lower number is higher precedence. Higher rank instructions act as "backup" is_stmt locations, providing good fallback locations if/when the primary candidate gets optimized away. The default values of 0 indicate the instruction isn’t interesting - it's not an is_stmt candidate.

2. *Clang annotates key instructions* with the new metadata. Variable assignments (stores, memory intrinsics), control flow (branches and their conditions, some unconditional branches), and exception handling instructions are annotated. Calls are ignored as they're unconditionally marked is_stmt.

3. *Throughout optimisation*, the DILocation is propagated normally. Cloned instructions get the original’s DILocation, the new fields get merged in getMergedLocation, etc. However, pass writers need to intercede in cases where a code path is duplicated, e.g. unrolling, jump-threading. In these cases we want to emit key instructions in both the original and duplicated code, so the duplicated must be assigned new `atomGroup` numbers, in a similar way that instruction operands must get remapped. There’s facilities to help this: `mapAtomInstance(const DebugLoc &DL, ValueToValueMapTy &VMap)` adds an entry to `VMap` which can later be used for remapping using `llvm::RemapSourceAtom(Instruction *I, ValueToValueMapTy &VM)`. `mapAtomInstance` is called from `llvm::CloneBasicBlock` and `llvm::RemapSourceAtom` is called from `llvm::RemapInstruction` so in many cases no additional effort is actually needed.

`mapAtomInstance` ensures `LLVMContextImpl::NextAtomGroup` is kept up to date, which is the global “next available atom number”.

The `DILocations` carry over from IR to MIR as normal, without any changes.

4. *DWARF emission* - Iterate over all instructions in a function. For each `(atomGroup, inlinedAt)` pair we find the set of instructions sharing the lowest rank. Only the last of these instructions in each basic block is included in the set. The instructions in this set get is_stmt applied to their source locations. That `is_stmt` then "floats" to the top of contiguous sequence of instructions with the same line number in the same block. That has two benefits when optimisations are enabled. First, this floats `is_stmt` to the top of epilogue instructions (rather than applying it to the `ret` instruction itself) which is important to avoid losing variable location coverage at return statements. Second, it reduces the difference in optimized code stepping behaviour between when Key Instructions is enabled and disabled in “uninteresting” cases. I.e., it appears to generally reduce unnecessary changes in stepping.

We’ve used contiguous line numbers rather than atom membership as the test there because of our choice to represent source atoms with a single integer ID. We can’t have instructions belonging to multiple atom groups or represent any kind of grouping hierarchy. That means we can’t rely on all the call setup instructions being in the same group currently (e.g., if one of the argument expressions contains key functionality such as a store, it will be in its own group).

## Adding the feature to a front end

Front ends that want to use the feature need to do some heavy lifting; they need to annotate Key Instructions and their backups with `DILocations` with the necessary `atomGroup` and `atomRank` values. Currently they also need to tell LLVM to interpret the metadata by passing the `-dwarf-use-key-instructions` flag.

The prototype had LLVM annotate instructions (instead of Clang) using simple heuristics (just looking at kind of instructions). This doesn't exist anywhere upstream, but could be shared if there's interest (e.g., so another front end can try it out before committing to a full implementation ), feel fre to reach out on Discourse (@OCHyams).

## Limitations

### Lack of multiple atom membership

Using a number to represent atom membership is limiting; currently an instruction cannot belong to multiple atoms. Does this come up in practice? Yes. Both in the front end and during optimisations. Consider this C code:
```c
a = b = c;
```
Clang generates this IR:
```llvm
  %0 = load i32, ptr %c.addr, align 4
  store i32 %0, ptr %b.addr, align 4
  store i32 %0, ptr %a.addr, align 4
```
The load of `c` is used by both stores (which are the Key Instructions for each assignment respectively). We can only use it as a backup location for one of the two atoms.

Certain optimisations merge source locations, which presents another case where it might make sense to be able to represent an instruction belonging to multiple atoms. Currently we deterministically pick one (choosing to keep the lower rank one if there is one).

### Disabled at O0

Consider the following code without optimisations:
```
int c =
    a + b;
```
In the current implementation an `is_stmt` won't be generated for the `a + b` instruction, meaning debuggers will likely step over the `add` and stop at the `store` of the result into `c` (which does get `is_stmt`). A user might have hoped to edit `a` or `b` on the previous line in order to alter the result stored to `c`, which they now won't have the chance to do (they'd need to edit the variables on a previous line instead). If the expression was all on one line then they would be able to edit the values before the `add`. For these reasons we're choosing to recommend that the feature should not be enabled at O0.

It should be possible to fix this case if we make a few changes: add all the instructions in the statement (i.e., including the loads) to the atom, and tweak the DwarfEmission code to understand this situation (same atom, different line). So there is room to persue this in the future. Though that gets tricky in some cases due to the [other limitation mentioned above](#lack-of-multiple-atom-membership), e.g.:
```c
int e =        // atom 1
    (a + b)    // atom 1
  * (c = d);   // - atom 2
```
```llvm
  %0 = load i32, ptr %a.addr, align 4     ; atom 1
  %1 = load i32, ptr %b.addr, align 4     ; atom 1
  %add = add nsw i32 %0, %1               ; atom 1
  %2 = load i32, ptr %d.addr, align 4     ; - atom 2
  store i32 %2, ptr %c.addr, align 4      ; - atom 2
  %mul = mul nsw i32 %add, %2             ; atom 1
  store i32 %mul, ptr %e, align 4         ; atom 1
```
Without multiple-atom-membership or some kind of atom hierarchy it's not apparent how to get the `is_stmt` to stick to `a + b`, given the other rules the `is_stmt` placement follows.

O0 isn't a key use-case so solving this is not a priority for the initial implementation. The trade off, smoother stepping at the cost of not being able to edit variables to affect an expression in some cases (and at particular stop points), becomes more attractive when optimisations are enabled (we find that editing variables in the debugger in optimized code often produces unexpected effects, so it's not a big concern that Key Instructions makes it harder sometimes).

---

**References**
* [1] Key Instructions: Solving the Code Location Problem for Optimized Code (C. Tice, . S. L. Graham, 2000)
* [2] Debugging Optimized Code: Concepts and Implementation on DIGITAL Alpha Systems (R. F. Brender et al)
