# Debug info migration: From intrinsics to records

We're planning on removing debug info intrinsics from LLVM, as they're slow, unwieldy and can confuse optimisation passes if they're not expecting them. Instead of having a sequence of instructions that looks like this:

```text
    %add = add i32 %foo, %bar
    call void @llvm.dbg.value(metadata %add, ...
    %sub = sub i32 %add, %tosub
    call void @llvm.dbg.value(metadata %sub, ...
    call void @a_normal_function()
```

with `dbg.value` intrinsics representing debug info records, it would instead be printed as:

```text
    %add = add i32 %foo, %bar
      #dbg_value(%add, ...
    %sub = sub i32 %add, %tosub
      #dbg_value(%sub, ...
    call void @a_normal_function()
```

The debug records are not instructions, do not appear in the instruction list, and won't appear in your optimisation passes unless you go digging for them deliberately.

# Great, what do I need to do!

Very little -- we've already instrumented all of LLVM to handle these new records ("`DbgRecords`") and behave identically to past LLVM behaviour. This is currently being turned on by default, so that `DbgRecords` will be used by default in memory, IR, and bitcode.

## API Changes

There are two significant changes to be aware of. Firstly, we're adding a single bit of debug relevant data to the `BasicBlock::iterator` class (it's so that we can determine whether ranges intend on including debug info at the beginning of a block or not). That means when writing passes that insert LLVM IR instructions, you need to identify positions with `BasicBlock::iterator` rather than just a bare `Instruction *`. Most of the time this means that after identifying where you intend on inserting something, you must also call `getIterator` on the instruction position -- however when inserting at the start of a block you _must_ use `getFirstInsertionPt`, `getFirstNonPHIIt` or `begin` and use that iterator to insert, rather than just fetching a pointer to the first instruction.

The second matter is that if you transfer sequences of instructions from one place to another manually, i.e. repeatedly using `moveBefore` where you might have used `splice`, then you should instead use the method `moveBeforePreserving`. `moveBeforePreserving` will transfer debug info records with the instruction they're attached to. This is something that happens automatically today -- if you use `moveBefore` on every element of an instruction sequence, then debug intrinsics will be moved in the normal course of your code, but we lose this behaviour with non-instruction debug info.

For a more in-depth overview of how to update existing code to support debug records, see [the guide below](#how-to-update-existing-code).

# C-API changes

All the functions that have been added are temporary and will be deprecated in the future. The intention is that they'll help downstream projects adapt during the transition period.

```
New functions (all to be deprecated)
------------------------------------
LLVMIsNewDbgInfoFormat                      # Returns true if the module is in the new non-instruction mode.
LLVMSetIsNewDbgInfoFormat                   # Convert to the requested debug info format.

LLVMDIBuilderInsertDeclareIntrinsicBefore   # Insert a debug intrinsic (old debug info format).
LLVMDIBuilderInsertDeclareIntrinsicAtEnd    # Same as above.
LLVMDIBuilderInsertDbgValueIntrinsicBefore  # Same as above.
LLVMDIBuilderInsertDbgValueIntrinsicAtEnd   # Same as above.

LLVMDIBuilderInsertDeclareRecordBefore      # Insert a debug record (new debug info format).
LLVMDIBuilderInsertDeclareRecordAtEnd       # Same as above.
LLVMDIBuilderInsertDbgValueRecordBefore     # Same as above.
LLVMDIBuilderInsertDbgValueRecordAtEnd      # Same as above.

Existing functions (behaviour change)
-------------------------------------
LLVMDIBuilderInsertDeclareBefore   # Insert a debug record (new debug info format) instead of a debug intrinsic (old debug info format).
LLVMDIBuilderInsertDeclareAtEnd    # Same as above.
LLVMDIBuilderInsertDbgValueBefore  # Same as above.
LLVMDIBuilderInsertDbgValueAtEnd   # Same as above.
```

# The new "Debug Record" model

Below is a brief overview of the new representation that replaces debug intrinsics; for an instructive guide on updating old code, see [here](#how-to-update-existing-code).

## What exactly have you replaced debug intrinsics with?

We're using a dedicated C++ class called `DbgRecord` to store debug info, with a one-to-one relationship between each instance of a debug intrinsic and each `DbgRecord` object in any LLVM IR program; these `DbgRecord`s are represented in the IR as non-instruction debug records, as described in the [Source Level Debugging](project:SourceLevelDebugging.rst#Debug Records) document. This class has a set of subclasses that store exactly the same information as is stored in debugging intrinsics. Each one also has almost entirely the same set of methods, that behave in the same way:

  https://llvm.org/docs/doxygen/classllvm_1_1DbgRecord.html
  https://llvm.org/docs/doxygen/classllvm_1_1DbgVariableRecord.html
  https://llvm.org/docs/doxygen/classllvm_1_1DbgLabelRecord.html

This allows you to treat a `DbgVariableRecord` as if it's a `dbg.value`/`dbg.declare`/`dbg.assign` intrinsic most of the time, for example in generic (auto-param) lambdas, and the same for `DbgLabelRecord` and `dbg.label`s.

## How do these `DbgRecords` fit into the instruction stream?

Like so:

```text
                 +---------------+          +---------------+
---------------->|  Instruction  +--------->|  Instruction  |
                 +-------+-------+          +---------------+
                         |
                         |
                         |
                         |
                         v
                  +-------------+
          <-------+  DbgMarker  |<-------
         /        +-------------+        \
        /                                 \
       /                                   \
      v                                     ^
 +-------------+    +-------------+   +-------------+
 |  DbgRecord  +--->|  DbgRecord  +-->|  DbgRecord  |
 +-------------+    +-------------+   +-------------+
```

Each instruction has a pointer to a `DbgMarker` (which will become optional), that contains a list of `DbgRecord` objects. No debugging records appear in the instruction list at all. `DbgRecord`s have a parent pointer to their owning `DbgMarker`, and each `DbgMarker` has a pointer back to it's owning instruction.

Not shown are the links from DbgRecord to other parts of the `Value`/`Metadata` hierachy: `DbgRecord` subclasses have tracking pointers to the DIMetadata that they use, and `DbgVariableRecord` has references to `Value`s that are stored in a `DebugValueUser` base class. This refers to a `ValueAsMetadata` object referring to `Value`s, via the `TrackingMetadata` facility.

The various kinds of debug intrinsic (value, declare, assign, label) are all stored in `DbgRecord` subclasses, with a "RecordKind" field distinguishing `DbgLabelRecord`s from `DbgVariableRecord`s, and a `LocationType` field in the `DbgVariableRecord` class further disambiguating the various debug variable intrinsics it can represent.

# How to update existing code

Any existing code that interacts with debug intrinsics in some way will need to be updated to interact with debug records in the same way. A few quick rules to keep in mind when updating code:

- Debug records will not be seen when iterating over instructions; to find the debug records that appear immediately before an instruction, you'll need to iterate over `Instruction::getDbgRecordRange()`.
- Debug records have interfaces that are identical to those of debug intrinsics, meaning that any code that operates on debug intrinsics can be trivially applied to debug records as well. The exceptions for this are `Instruction` or `CallInst` methods that don't logically apply to debug records, and `isa`/`cast`/`dyn_cast` methods, are replaced by methods on the `DbgRecord` class itself.
- Debug records cannot appear in a module that also contains debug intrinsics; the two are mutually exclusive. As debug records are the future format, handling records correctly should be prioritized in new code.
- Until support for intrinsics is no longer present, a valid hotfix for code that only handles debug intrinsics and is non-trivial to update is to convert the module to the intrinsic format using `Module::setIsNewDbgInfoFormat`, and convert it back afterwards.
  - This can also be performed within a lexical scope for a module or an individual function using the class `ScopedDbgInfoFormatSetter`:
  ```
  void handleModule(Module &M) {
    {
      ScopedDbgInfoFormatSetter FormatSetter(M, false);
      handleModuleWithDebugIntrinsics(M);
    }
    // Module returns to previous debug info format after exiting the above block.
  }
  ```

Below is a rough guide on how existing code that currently supports debug intrinsics can be updated to support debug records.

## Creating debug records

Debug records will automatically be created by the `DIBuilder` class when the new format is enabled. As with instructions, it is also possible to call `DbgRecord::clone` to create an unattached copy of an existing record.

## Skipping debug records, ignoring debug-uses of `Values`, stably counting instructions, etc.

This will all happen transparently without needing to think about it!

```
for (Instruction &I : BB) {
  // Old: Skips debug intrinsics
  if (isa<DbgInfoIntrinsic>(&I))
    continue;
  // New: No extra code needed, debug records are skipped by default.
  ...
}
```

## Finding debug records

Utilities such as `findDbgUsers` and the like now have an optional argument that will return the set of `DbgVariableRecord` records that refer to a `Value`. You should be able to treat them the same as intrinsics.

```
// Old:
  SmallVector<DbgVariableIntrinsic *> DbgUsers;
  findDbgUsers(DbgUsers, V);
  for (auto *DVI : DbgUsers) {
    if (DVI->getParent() != BB)
      DVI->replaceVariableLocationOp(V, New);
  }
// New:
  SmallVector<DbgVariableIntrinsic *> DbgUsers;
  SmallVector<DbgVariableRecord *> DVRUsers;
  findDbgUsers(DbgUsers, V, &DVRUsers);
  for (auto *DVI : DbgUsers)
    if (DVI->getParent() != BB)
      DVI->replaceVariableLocationOp(V, New);
  for (auto *DVR : DVRUsers)
    if (DVR->getParent() != BB)
      DVR->replaceVariableLocationOp(V, New);
```

## Examining debug records at positions

Call `Instruction::getDbgRecordRange()` to get the range of `DbgRecord` objects that are attached to an instruction.

```
for (Instruction &I : BB) {
  // Old: Uses a data member of a debug intrinsic, and then skips to the next
  // instruction.
  if (DbgInfoIntrinsic *DII = dyn_cast<DbgInfoIntrinsic>(&I)) {
    recordDebugLocation(DII->getDebugLoc());
    continue;
  }
  // New: Iterates over the debug records that appear before `I`, and treats
  // them identically to the intrinsic block above.
  // NB: This should always appear at the top of the for-loop, so that we
  // process the debug records preceding `I` before `I` itself.
  for (DbgRecord &DR = I.getDbgRecordRange()) {
    recordDebugLocation(DR.getDebugLoc());
  }
  processInstruction(I);
}
```

This can also be passed through the function `filterDbgVars` to specifically
iterate over DbgVariableRecords, which are more commonly used.

```
for (Instruction &I : BB) {
  // Old: If `I` is a DbgVariableIntrinsic we record the variable, and apply
  // extra logic if it is an `llvm.dbg.declare`.
  if (DbgVariableIntrinsic *DVI = dyn_cast<DbgVariableIntrinsic>(&I)) {
    recordVariable(DVI->getVariable());
    if (DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(DVI))
      recordDeclareAddress(DDI->getAddress());
    continue;
  }
  // New: `filterDbgVars` is used to iterate over only DbgVariableRecords.
  for (DbgVariableRecord &DVR = filterDbgVars(I.getDbgRecordRange())) {
    recordVariable(DVR.getVariable());
    // Debug variable records are not cast to subclasses; simply call the
    // appropriate `isDbgX()` check, and use the methods as normal.
    if (DVR.isDbgDeclare())
      recordDeclareAddress(DVR.getAddress());
  }
  // ...
}
```

## Processing individual debug records

In most cases, any code that operates on debug intrinsics can be extracted to a template function or auto lambda (if it is not already in one) that can be applied to both debug intrinsics and debug records - though keep in mind the main exception that `isa`/`cast`/`dyn_cast` do not apply to `DbgVariableRecord` types.

```
// Old: Function that operates on debug variable intrinsics in a BasicBlock, and
// collects llvm.dbg.declares.
void processDbgInfoInBlock(BasicBlock &BB,
                           SmallVectorImpl<DbgDeclareInst*> &DeclareIntrinsics) {
  for (Instruction &I : BB) {
    if (DbgVariableIntrinsic *DVI = dyn_cast<DbgVariableIntrinsic>(&I)) {
      processVariableValue(DebugVariable(DVI), DVI->getValue());
      if (DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(DVI))
        Declares.push_back(DDI);
      else if (!isa<Constant>(DVI->getValue()))
        DVI->setKillLocation();
    }
  }
}

// New: Template function is used to deduplicate handling of intrinsics and
// records.
// An overloaded function is also used to handle isa/cast/dyn_cast operations
// for intrinsics and records, since those functions cannot be directly applied
// to DbgRecords.
DbgDeclareInst *DynCastToDeclare(DbgVariableIntrinsic *DVI) {
  return dyn_cast<DbgDeclareInst>(DVI);
}
DbgVariableRecord *DynCastToDeclare(DbgVariableRecord *DVR) {
  return DVR->isDbgDeclare() ? DVR : nullptr;
}

template<typename DbgVarTy, DbgDeclTy>
void processDbgVariable(DbgVarTy *DbgVar,
                       SmallVectorImpl<DbgDeclTy*> &Declares) {
    processVariableValue(DebugVariable(DbgVar), DbgVar->getValue());
    if (DbgDeclTy *DbgDeclare = DynCastToDeclare(DbgVar))
      Declares.push_back(DbgDeclare);
    else if (!isa<Constant>(DbgVar->getValue()))
      DbgVar->setKillLocation();
};

void processDbgInfoInBlock(BasicBlock &BB,
                           SmallVectorImpl<DbgDeclareInst*> &DeclareIntrinsics,
                           SmallVectorImpl<DbgVariableRecord*> &DeclareRecords) {
  for (Instruction &I : BB) {
    if (DbgVariableIntrinsic *DVI = dyn_cast<DbgVariableIntrinsic>(&I))
      processDbgVariable(DVI, DeclareIntrinsics);
    for (DbgVariableRecord *DVR : filterDbgVars(I.getDbgRecordRange()))
      processDbgVariable(DVR, DeclareRecords);
  }
}
```

## Moving and deleting debug records

You can use `DbgRecord::removeFromParent` to unlink a `DbgRecord` from it's marker, and then `BasicBlock::insertDbgRecordBefore` or `BasicBlock::insertDbgRecordAfter` to re-insert the `DbgRecord` somewhere else. You cannot insert a `DbgRecord` at an arbitary point in a list of `DbgRecord`s (if you're doing this with `llvm.dbg.value`s then it's unlikely to be correct).

Erase `DbgRecord`s by calling `eraseFromParent`.

```
// Old: Move a debug intrinsic to the start of the block, and delete all other intrinsics for the same variable in the block.
void moveDbgIntrinsicToStart(DbgVariableIntrinsic *DVI) {
  BasicBlock *ParentBB = DVI->getParent();
  DVI->removeFromParent();
  for (Instruction &I : ParentBB) {
    if (auto *BlockDVI = dyn_cast<DbgVariableIntrinsic>(&I))
      if (BlockDVI->getVariable() == DVI->getVariable())
        BlockDVI->eraseFromParent();
  }
  DVI->insertBefore(ParentBB->getFirstInsertionPt());
}

// New: Perform the same operation, but for a debug record.
void moveDbgRecordToStart(DbgVariableRecord *DVR) {
  BasicBlock *ParentBB = DVR->getParent();
  DVR->removeFromParent();
  for (Instruction &I : ParentBB) {
    for (auto &BlockDVR : filterDbgVars(I.getDbgRecordRange()))
      if (BlockDVR->getVariable() == DVR->getVariable())
        BlockDVR->eraseFromParent();
  }
  DVR->insertBefore(ParentBB->getFirstInsertionPt());
}
```

## What about dangling debug records?

If you have a block like so:

```text
    foo:
      %bar = add i32 %baz...
      dbg.value(metadata i32 %bar,...
      br label %xyzzy
```

your optimisation pass may wish to erase the terminator and then do something to the block. This is easy to do when debug info is kept in instructions, but with `DbgRecord`s there is no trailing instruction to attach the variable information to in the block above, once the terminator is erased. For such degenerate blocks, `DbgRecord`s are stored temporarily in a map in `LLVMContext`, and are re-inserted when a terminator is reinserted to the block or other instruction inserted at `end()`.

This can technically lead to trouble in the vanishingly rare scenario where an optimisation pass erases a terminator and then decides to erase the whole block. (We recommend not doing that).

## Anything else?

The above guide does not comprehensively cover every pattern that could apply to debug intrinsics; as mentioned at the [start of the guide](#how-to-update-existing-code), you can temporarily convert the target module from debug records to intrinsics as a stopgap measure. Most operations that can be performed on debug intrinsics have exact equivalents for debug records, but if you encounter any exceptions, reading the class docs (linked [here](#what-exactly-have-you-replaced-debug-intrinsics-with)) may give some insight, there may be examples in the existing codebase, and you can always ask for help on the [forums](https://discourse.llvm.org/tag/debuginfo).