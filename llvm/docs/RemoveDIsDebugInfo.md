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

Approximately nothing -- we've already instrumented all of LLVM to handle these new records ("`DbgRecords`") and behave identically to past LLVM behaviour. We plan on turning this on by default some time soon, with IR converted to the intrinsic form of debug info at terminals (textual IR, bitcode) for a short while, before then changing the textual IR and bitcode formats.

There are two significant changes to be aware of. Firstly, we're adding a single bit of debug relevant data to the `BasicBlock::iterator` class (it's so that we can determine whether ranges intend on including debug info at the beginning of a block or not). That means when writing passes that insert LLVM IR instructions, you need to identify positions with `BasicBlock::iterator` rather than just a bare `Instruction *`. Most of the time this means that after identifying where you intend on inserting something, you must also call `getIterator` on the instruction position -- however when inserting at the start of a block you _must_ use `getFirstInsertionPt`, `getFirstNonPHIIt` or `begin` and use that iterator to insert, rather than just fetching a pointer to the first instruction.

The second matter is that if you transfer sequences of instructions from one place to another manually, i.e. repeatedly using `moveBefore` where you might have used `splice`, then you should instead use the method `moveBeforePreserving`. `moveBeforePreserving` will transfer debug info records with the instruction they're attached to. This is something that happens automatically today -- if you use `moveBefore` on every element of an instruction sequence, then debug intrinsics will be moved in the normal course of your code, but we lose this behaviour with non-instruction debug info.

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
```

# Anything else?

Not really, but here's an "old vs new" comparison of how to do certain things and quickstart for how this "new" debug info is structured.

## Skipping debug records, ignoring debug-uses of Values, stably counting instructions...

This will all happen transparently without needing to think about it!

## What exactly have you replaced debug intrinsics with?

We're using a dedicated C++ class called `DbgRecord` to store debug info, with a one-to-one relationship between each instance of a debug intrinsic and each `DbgRecord` object in any LLVM IR program; these `DbgRecord`s are represented in the IR as non-instruction debug records, as described in the [Source Level Debugging](project:SourceLevelDebugging.rst#Debug Records) document. This class has a set of subclasses that store exactly the same information as is stored in debugging intrinsics. Each one also has almost entirely the same set of methods, that behave in the same way:

  https://llvm.org/docs/doxygen/classllvm_1_1DbgRecord.html
  https://llvm.org/docs/doxygen/classllvm_1_1DbgVariableRecord.html
  https://llvm.org/docs/doxygen/classllvm_1_1DPLabel.html

This allows you to treat a `DbgVariableRecord` as if it's a `dbg.value`/`dbg.declare`/`dbg.assign` intrinsic most of the time, for example in generic (auto-param) lambdas, and the same for `DPLabel` and `dbg.label`s.

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
                  +------------+
          <-------+  DPMarker  |<-------
         /        +------------+        \
        /                                \
       /                                  \
      v                                    ^
 +-------------+    +-------------+   +-------------+
 |  DbgRecord  +--->|  DbgRecord  +-->|  DbgRecord  |
 +-------------+    +-------------+   +-------------+
```

Each instruction has a pointer to a `DPMarker` (which will become optional), that contains a list of `DbgRecord` objects. No debugging records appear in the instruction list at all. `DbgRecord`s have a parent pointer to their owning `DPMarker`, and each `DPMarker` has a pointer back to it's owning instruction.

Not shown are the links from DbgRecord to other parts of the `Value`/`Metadata` hierachy: `DbgRecord` subclasses have tracking pointers to the DIMetadata that they use, and `DbgVariableRecord` has references to `Value`s that are stored in a `DebugValueUser` base class. This refers to a `ValueAsMetadata` object referring to `Value`s, via the `TrackingMetadata` facility.

The various kinds of debug intrinsic (value, declare, assign, label) are all stored in `DbgRecord` subclasses, with a "RecordKind" field distinguishing `DPLabel`s from `DbgVariableRecord`s, and a `LocationType` field in the `DbgVariableRecord` class further disambiguating the various debug variable intrinsics it can represent.

## Finding debug info records

Utilities such as `findDbgUsers` and the like now have an optional argument that will return the set of `DbgVariableRecord` records that refer to a `Value`. You should be able to treat them the same as intrinsics.

## Examining debug info records at positions

Call `Instruction::getDbgRecordRange()` to get the range of `DbgRecord` objects that are attached to an instruction.

## Moving around, deleting

You can use `DbgRecord::removeFromParent` to unlink a `DbgRecord` from it's marker, and then `BasicBlock::insertDbgRecordBefore` or `BasicBlock::insertDbgRecordAfter` to re-insert the `DbgRecord` somewhere else. You cannot insert a `DbgRecord` at an arbitary point in a list of `DbgRecord`s (if you're doing this with `dbg.value`s then it's unlikely to be correct).

Erase `DbgRecord`s by calling `eraseFromParent` or `deleteInstr` if it's already been removed.

## What about dangling `DbgRecord`s?

If you have a block like so:

```text
    foo:
      %bar = add i32 %baz...
      dbg.value(metadata i32 %bar,...
      br label %xyzzy
```

your optimisation pass may wish to erase the terminator and then do something to the block. This is easy to do when debug info is kept in instructions, but with `DbgRecord`s there is no trailing instruction to attach the variable information to in the block above, once the terminator is erased. For such degenerate blocks, `DbgRecord`s are stored temporarily in a map in `LLVMContext`, and are re-inserted when a terminator is reinserted to the block or other instruction inserted at `end()`.

This can technically lead to trouble in the vanishingly rare scenario where an optimisation pass erases a terminator and then decides to erase the whole block. (We recommend not doing that).
