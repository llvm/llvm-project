# Key Instructions debug info in LLVM and Clang

Key Instructions is an LLVM feature that reduces the jumpiness of optimized code debug stepping by distinguishing the significance of instructions that make up source language statements. This document explains the feature and how it is implemented in [LLVM](#llvm) and [Clang](#clang-and-other-front-ends).

## Status

Feature complete except for coroutines, which fall back to not-key-instructions handling for now but will get support soon (there is no fundamental reason why they cannot be supported, we've just not got to it at time of writing).

Tell Clang [not] to produce Key Instructions metadata with `-g[no-]key-instructions`.

The feature improves optimized code stepping; it's intended for the feature to be used with optimisations enabled. Although the feature works at O0 it is not recommended because in some cases the effect of editing variables may not always be immediately realised. In some cases, debuggers may place a breakpoint after parts of an expression have been evaluated, which limits the ability to have variable edits affect expressions. (This is a quirk of the current implementation, rather than fundamental limitation, covered in more detail [later](#disabled-at-o0).)

This is a DWARF-based feature. There is currently no plan to support CodeView.

Set LLVM flag `-dwarf-use-key-instructions` to `false` to ignore Key Instructions metadata when emitting DWARF.

# LLVM

## Problem statement

A lot of the noise in stepping comes from code motion and instruction scheduling. Consider a long expression on a single line. It may involve multiple operations that optimisations move, re-order, and interleave with other instructions that have different line numbers.

DWARF provides a helpful tool the compiler can employ to mitigate this jumpiness, the `is_stmt` flag, which indicates that an instruction is a recommended breakpoint location. However, LLVM's current approach to deciding `is_stmt` placement essentially reduces down to "is the associated line number different to the previous instruction's?".

(Note: It's up to the debugger if it wants to interpret `is_stmt` or not, and at time of writing LLDB doesn't; possibly because until now LLVM's `is_stmt`s convey no information that can't already be deduced from the rest of the line table.)

## Solution overview

Taking ideas from two papers [1][2] that explore the issue, especially C. Tice's:

From the perspective of a source-level debugger user:

* Source code is made up of interesting constructs; the level of granularity for “interesting” while stepping is typically assignments, calls, control flow. We’ll call these interesting constructs Atoms.

* Atoms usually have one instruction that implements the functionality that a user can observe; once they step “off” that instruction, the atom is finalised. We’ll call that a Key Instruction.

* Communicating where the key instructions are to the debugger (using DWARF’s is_stmt) avoids jumpiness introduced by scheduling non-key instructions without losing source attribution (because non-key instructions retain an associated source location, they’re just ignored for stepping).

## Solution implementation

1. `DILocation` has 2 new fields, `atomGroup` and `atomRank`. `DISubprogram` has a new field `keyInstructions`.
2. Clang creates `DILocations` using the new fields to communicate which instructions are "interesting", and sets `keyInstructions` true in `DISubprogram`s to tell LLVM to interpret the new metadata in those functions.
3. There’s some bookkeeping required by optimisations that duplicate control flow.
4. During DWARF emission, the new metadata is collected (linear scan over instructions) to decide `is_stmt` placements.

Details:

1. *The metadata* - The two new `DILocation` fields are `atomGroup` and `atomRank` and are both are unsigned integers. `atomGroup` is 61 bits and `atomRank` 3 bits. Instructions in the same function with the same `(atomGroup, inlinedAt)` pair are part of the same source atom. `atomRank` determines `is_stmt` preference within that group, where a lower number is higher precedence. Higher rank instructions act as "backup" `is_stmt` locations, providing good fallback locations if/when the primary candidate gets optimized away. The default values of 0 indicate the instruction isn’t interesting - it's not an `is_stmt` candidate. If `keyInstructions` in `DISubprogram` is false (default) then the new `DILocation` metadata is ignored for the function (including inlined instances) when emitting DWARF.

2. *Clang annotates key instructions* with the new metadata. Variable assignments (stores, memory intrinsics), control flow (branches and their conditions, some unconditional branches), and exception handling instructions are annotated. Calls are ignored as they're unconditionally marked `is_stmt`.

3. *Throughout optimisation*, the `DILocation` is propagated normally. Cloned instructions get the original’s `DILocation`, the new fields get merged in `getMergedLocation`, etc. However, pass writers need to intercede in cases where a code path is duplicated, e.g. unrolling, jump-threading. In these cases we want to emit key instructions in both the original and duplicated code, so the duplicated must be assigned new `atomGroup` numbers, in a similar way that instruction operands must get remapped. There are facilities to help this: `mapAtomInstance(const DebugLoc &DL, ValueToValueMapTy &VMap)` adds an entry to `VMap` which can later be used for remapping using `llvm::RemapSourceAtom(Instruction *I, ValueToValueMapTy &VM)`. `mapAtomInstance` is called from `llvm::CloneBasicBlock` and `llvm::RemapSourceAtom` is called from `llvm::RemapInstruction` so in many cases no additional work is actually needed. `mapAtomInstance` ensures `LLVMContextImpl::NextAtomGroup` is kept up to date, which is the global “next available atom number”.
The `DILocations` carry over from IR to MIR as normal, without any changes.

4. *DWARF emission* - Iterate over all instructions in a function. For each `(atomGroup, inlinedAt)` pair we find the set of instructions sharing the lowest rank. Only the last of these instructions in each basic block is included in the set. The instructions in this set get `is_stmt` applied to their source locations. That `is_stmt` then "floats" to the top of contiguous sequence of instructions with the same line number in the same basic block. That has two benefits when optimisations are enabled. First, this floats `is_stmt` to the top of epilogue instructions (rather than applying it to the `ret` instruction itself) which is important to avoid losing variable location coverage at return statements. Second, it reduces the difference in optimized code stepping behaviour between when Key Instructions is enabled and disabled in “uninteresting” cases. I.e., it appears to generally reduce unnecessary changes in stepping.\
We’ve used contiguous line numbers rather than atom membership as the test there because of our choice to represent source atoms with a single integer ID. We can’t have instructions belonging to multiple atom groups or represent any kind of grouping hierarchy. That means we can’t rely on all the call setup instructions being in the same group currently (e.g., if one of the argument expressions contains key functionality such as a store, it will be in its own group).

## Limitations

### Lack of multiple atom membership

Using a number to represent atom membership is limiting; currently an instruction that belongs to multiple source atoms cannot belong to multiple atom groups. This does occur in practice, both in the front end and during optimisations. Consider this C code:
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
```c
int c =
    a + b;
```
In the current implementation an `is_stmt` won't be generated for the `a + b` instruction, meaning debuggers will likely step over the `add` and stop at the `store` of the result into `c` (which does get `is_stmt`). A user might have wished to edit `a` or `b` on the previous line in order to alter the result stored to `c`, which they now won't have the chance to do (they'd need to edit the variables on a previous line instead). If the expression was all on one line then they would be able to edit the values before the `add`. For these reasons we're choosing to recommend that the feature should not be enabled at O0.

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

# Clang and other front ends

Tell Clang [not] to produce Key Instructions metadata with `-g[no-]key-instructions`.

## Implementation

Clang needs to annotate key instructions with the new metadata. Variable assignments (stores, memory intrinsics), control flow (branches and their conditions, some unconditional branches), and exception handling instructions are annotated. Calls are ignored as they're unconditionally marked `is_stmt`. This is achieved with a few simple constructs:

Class `ApplyAtomGroup` - This is a scoped helper similar to `ApplyDebugLocation` that creates a new source atom group which instructions can be added to. It's used during CodeGen to declare that a new source atom has started, e.g. in `CodeGenFunction::EmitBinaryOperatorLValue`.

`CodeGenFunction::addInstToCurrentSourceAtom(llvm::Instruction *KeyInstruction, llvm::Value *Backup)` adds an instruction (and a backup instruction if non-null) to the current "atom group" defined with `ApplyAtomGroup`. The Key Instruction gets rank 1, and backup instructions get higher ranks (the function looks through casts, applying increasing rank as it goes). There are a lot of sites in Clang that need to call this (mostly stores and store-like instructions). Most stores created through `CGBuilderTy` are annotated, but some that don't need to be key are not. It's important to remember that if there's no active atom group, i.e. no active `ApplyAtomGroup` instance, then `addInstToCurrentSourceAtom` does not annotate the instructions.

`CodeGenFunction::addInstToNewSourceAtom(llvm::Instruction *KeyInstruction, llvm::Value *Backup)` adds an instruction (and a backup instruction if non-null) to a new "atom group". Currently mostly used in loop handling code.

`CodeGenFunction::addInstToSpecificSourceAtom(llvm::Instruction *KeyInstruction, llvm::Value *Backup, uint64_t Atom)` adds the instruction (and backup instruction if non-null) to the specific group `Atom`. This is currently only used for `rets` which is explored in the examples below. Special handling is needed due to the fact that an existing atom group needs to be reused in some circumstances, so neither of the other helper functions are appropriate.

## Examples

A simple example walk through:
```c
void fun(int a) {
  int b = a;
}
```

There are two key instructions here, the assignment and the implicit return. We want to emit metadata that looks like this:

```llvm
define hidden void @_Z3funi(i32 noundef %a) #0 !dbg !11 {
entry:
  %a.addr = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  %0 = load i32, ptr %a.addr, align 4, !dbg !DILocation(line: 2, scope: !11, atomGroup: 1, atomRank: 2)
  store i32 %0, ptr %b, align 4,       !dbg !DILocation(line: 2, scope: !11, atomGroup: 1, atomRank: 1)
  ret void,                            !dbg !DILocation(line: 3, scope: !11, atomGroup: 2, atomRank: 1)
}
```

The store is the key instruction for the assignment (`atomGroup` 1). The instruction corresponding to the final (and in this case only) RHS value, the load from `%a.addr`, is a good backup location for `is_stmt` if the store gets optimized away. It's part of the same source atom, but has lower `is_stmt` precedence, so it gets a higher `atomRank`. This is achieved by starting an atom group with `ApplyAtomGroup` for the source atom (in this case a variable init) in `EmitAutoVarInit`. The instructions (both key and backup) are then annotated by call to `addInstToCurrentSourceAtom` called from `EmitStoreOfScalar`.

The implicit return is also key (`atomGroup` 2) so that it's stepped on, to match existing non-key-instructions behaviour. This is achieved by calling  `addInstToNewSourceAtom` from within `EmitFunctionEpilog`.

Explicit return statements are handled uniquely. Rather than emit a `ret` for each `return` Clang, in all but the simplest cases (as in the first example) emits a branch to a dedicated block with a single `ret`. That branch is the key instruction for the return statement. If there's only one branch to that block, because there's only one `return` (as in this example), Clang folds the block into its only predecessor. Handily `EmitReturnBlock` returns the `DebugLoc` associated with the single branch in that case, which is fed into `addInstToSpecificSourceAtom` to ensure the `ret` gets the right group.


## Supporting Key Instructions from another front end

Front ends that want to use the feature need to group and rank instructions according to their source atoms and interingness by attaching `DILocations` with the necessary `atomGroup` and `atomRank` values. They also need to set the `keyInstructions` field to `true` in `DISubprogram`s to tell LLVM to interpret the new metadata in those functions.

The prototype had LLVM annotate instructions (instead of Clang) using simple heuristics (just looking at kind of instructions, e.g., annotating all stores, conditional branches, etc). This doesn't exist anywhere upstream, but could be shared if there's interest (e.g., so another front end can try it out before committing to a full implementation), feel free to reach out on Discourse (@OCHyams, @jmorse).

---

**References**
* [1] Key Instructions: Solving the Code Location Problem for Optimized Code (C. Tice, S. L. Graham, 2000)
* [2] Debugging Optimized Code: Concepts and Implementation on DIGITAL Alpha Systems (R. F. Brender et al)
