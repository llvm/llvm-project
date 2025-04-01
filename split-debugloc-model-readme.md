# Split DebugLoc-Model

This describes the core design and implementation details of the proposed optimized storage model for source locations, to replace DILocations.

## Design

### Background

Under the current model, source locations are stored as `DILocation *` fields on each instruction. DILocations are almost always uniqued, which means that any two instructions that have identical source locations will point to the same DILocation object, which saves memory and simplifies equality checks. DILocations are a kind of metadata, whose total memory footprint is:

- DILocation (40-48):
    - MDNode (32):
        - MDNode::Header (16)
        - Context (8)
        - Metadata (8):
            - SubclassID (1)
            - Line (4)
            - Col (2)
            - Storage + Implicitcode (1)
    - MDNode Ops (8-16):
        - Scope (8)
        - InlinedAt (8, Optional)

In here is a lot of space wasted on scaffolding that is either common to all DILocations or only used during parsing. This scaffolding is convenient for metadata in general, and generally necessary - but DILocations are unique in both volume (the number of DILocations allocated often dwarfs all other metadata combined), and in that they are only ever used within a specific context, e.g. attached to instructions (either directly or via loop metadata). There are also hidden costs here - the LLVMContext must store all DILocations in a DenseMap of keys to DILocation pointers (used for uniquing), which also takes up a substantial amount of space; in some cases this storage alone comprised around 2-3% of LLVM's total memory usage.

A further inefficiency of this approach is that while uniquing ensures we do not store duplicate DILocation instances, we only deduplicate *exact* copies, and otherwise have no storage optimizations. Throughout compilation, we may make small changes to DILocations - setting the line or column field to 0, changing the scope, or adding inlinedAt fields. Every time we do this, we must allocate a complete new DILocation, and we are also unable to deallocate the old DILocation even if it is no longer used anywhere - we never delete metadata until the LLVMContext's destructor, i.e. when exiting the program.

### Proposal

Instead of storing each DILocation as a piece of standalone metadata, we fold them into their relevant context, i.e. the function they are used by. The debug info metadata for a function is stored in the `DISubprogram` class; each function with debug info has a 1-1 relationship with a DISubprogram instance. This makes it suitable for storing function-local metadata, and it is where we store the source location information in this new model. The key points of this model are as follows:

- All source location data for a function is owned by its DISubprogram; this data is stored in vectors, which contain only source location data and no scaffolding or other "context"-related fields.
- We split the source location information into two parts: the context information `(Scope, InlinedAt)`, and the location information `(Line, Column, ImplicitCode)`.
    - In the current model, `InlinedAt` is a pointer to a *distinct* DILocation, meaning one that is not uniqued; this is to ensure that two separate call inlinings, even with identical source locations, will always be distinct from each other (thus the inlined source locations are not confused for each other).
    - In the proposed model, `InlinedAt` is a DebugLoc (see below); for each inlined call, there is a unique entry in the position vector which is used to distinguish that call from any other inlined calls at the same position.
- Instead of each Instruction's `DebugLoc` field storing a DILocation*, it instead stores a pair of 4-byte indexes into the owning DISubprogram's source location vectors.

The key performance benefits of this approach in order of importance (descending) are:

- Removing all redundant context information from source locations, which accounted for 43-50% of memory allocated for DILocations.
- Reducing the number of unique source location entries needed, by no longer requiring a unique allocation for each *combination* of scope+position, only for each scope and position individually.
- Using tightly-allocated vectors that are rarely (if ever) re-allocated to store and reference source locations, instead of using an incredibly large map for references and malloc for storage.

### API

The key difference in usage is that the data attached to an instruction, the DebugLoc, is not sufficient to access the underlying storage information; it is only an index to data stored in the DISubprogram. This means that if we wish to arbitrarily pass around information equivalent to a DILocation*, we need a *fat-pointer* class - which for the prototype, I have named `DILocRef`:

```cpp
class DILocRef {
    DISubprogram *SP;
    DebugLoc Index;
    DILocRef(Instruction &I) : SP(I.getFunction()->getSubprogram()), Index(I.getDebugLoc()) {}
};
```

This class implements all of the methods that a DILocation does; under the hood, it uses the Index to find the relevant data via the SP. One difference to keep in mind is that the DILocation `getInlinedAt()` method returns a DILocation*, while the DILocRef equivalent returns a DILocRef - this makes usage consistent, allowing trivial rewrites of code that traverses a chain of inlined at locations (e.g. `for (DL = ...; DL; DL = DL->getInlinedAt())`). Compared to DILocations, this comes with the cost that we must perform 2 indirections per operation (accessing the subprogram, then looking up the storage vector at the specified index) rather than 1 (accessing the DILocation). Fortunately, operations that examine the data in a DILocation are not very common outside of a few specific points in compilation, which softens this blow.

Another cost of this class is that it is twice as large as a simple pointer. This is a small matter in cases where we pass a single DILocation to a function as a one-off operation, but becomes more relevant in code that deals more extensively with source locations and may store e.g. a large vector of DILocations. In these cases we would be doubling the size of the storage, and almost certainly for no good reason: in essentially all code that deals with source locations, they are examined in the context of a particular function - meaning that if we have a vector of DILocations, they will all be contained within the same function, or otherwise grouped by function. Thus, instead of storing a DILocRef in place of a DILocation*, it is more efficient to store the DISubprogram* outside of the vector, and store only the DebugLocs in the vector itself.

### Comparison

**Using DILocations:**
```llvm
define i32 @f() local_unnamed_addr #0 !dbg !7 {
entry:
  %0 = tail call i64 @llvm.objectsize.i64.p0(ptr inttoptr (i64 1 to ptr), i1 false) #2, !dbg !11
  br label %for.cond, !dbg !18

for.cond:                                         ; preds = %for.cond, %entry
  %call.i = tail call ptr @__memset_chk(ptr null, i32 0, i64 0, i64 %0) #2, !dbg !19
  br label %for.cond, !dbg !20, !llvm.loop !21
}
;...
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !2)
!11 = !DILocation(line: 1, column: 56, scope: !12, inlinedAt: !13)
!13 = distinct !DILocation(line: 1, column: 17, scope: !11)
!18 = !DILocation(line: 1, column: 9, scope: !7)
!19 = !DILocation(line: 1, column: 27, scope: !12, inlinedAt: !13)
!20 = !DILocation(line: 1, column: 9, scope: !14)
```

**Using Split-DebugLocs:**
```llvm
define i32 @f() local_unnamed_addr !dbg !7 {
entry:
  %0 = tail call i64 @llvm.objectsize.i64.p0(ptr inttoptr (i64 1 to ptr), i1 false), !DebugLoc(srcLoc: 4, locScope: 2)
  br label %for.cond, !DebugLoc(srcLoc: 1, locScope: 0)

for.cond:                                         ; preds = %for.cond, %entry
  %call.i = tail call ptr @__memset_chk(ptr null, i32 0, i64 0, i64 %0), !DebugLoc(srcLoc: 3, locScope: 2)
  br label %for.cond, !llvm.loop !16, !DebugLoc(srcLoc: 1, locScope: 1)
}
;...
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2, srcLocs: [
  {line: 0} ; 0
  {line: 1, column: 9} ; 1
  {line: 1, column: 17} ; 2
  {line: 1, column: 27} ; 3
  {line: 1, column: 56} ; 4
], locScopes: [
  {scope: !7} ; 0
  {scope: !14} ; 1
  {scope: !12, inlinedAt: DebugLoc(srcLoc: 2, locScope: 1)} ; 2
])
```

## Implementation

This section lays out in greater detail exactly how the prototype of this change has been implemented, some of the problems encountered (some of which are not yet solved), the transition roadmap, and a general collection of notes and explanations.

### Storage Model

As described in the outline above, we store source location information in two separate vectors in the DISubprogram; to be more accurate however, we actually have *4* vectors, and one field. Some of these require further context to explain, but for reference:

```cpp
  // Stores LocScope (Scope, InlinedAt) data.
  SmallVector<DILocScopeData, 0> FnLocScopes;
  // Stores SrcLoc (Line, Column, IsImplicitCode) data.
  SmallVector<DISrcLocData, 0> FnSrcLocs;
  // Stores non-normalized SrcLoc data with associated InlinedAt indexes.
  SmallVector<std::pair<DISrcLocData, uint32_t>, 0> NonNormalFnSrcLocs;
  // Stores a collection of indices describing where each InlinedAt range in FnSrcLoc starts.
  SmallVector<uint32_t, 0> FnInlinedAtSrcLocRanges;
```

This adds a decent amount to the size of the DISubprogram class, since we mostly only have one of these per-function it isn't too great a price to pay; these fields are all completely unused on *declaration* DISubprograms however, as these correspond to just a function declaration and therefore contain no concrete source locations. It's possible we could subclass DISubprogram to prevent redundant storage (and potentially as a solution to the nodebug issue described below), but I've avoided doing this prematurely for the prototype.

For any DISubprogram that does or *could* contain DILocations, we also initialize two special elements: `FnLocScopes[0]` is always the function scope, meaning it has no InlinedAt location and the scope is the owning DISubprogram; this is a commonly used scope, being the outermost scope that can exist in this function. `FnSrcLocs[0]` is always the line 0 location, the value used for "artificial" or dropped source locations, which is also commonly caused by optimizations. Because of this, we can always rely on `DebugLoc(0, 0)` being a valid location, which is a line-less source location at function scope.

#### Normalized SrcLocs

The FnSrcLocs array is meant to be "normalized" at all times. What this means is that it is tightly-allocated (the reserved capacity is exactly equal to the number of elements), and it is arranged into buckets of source locations, with one bucket for non-inlined source locs and one bucket for each inlined call, where the source locations in each bucket other than the first are sorted. The reason for bucketing by InlinedAt will be explained later, but the sorting exists to ensure faster lookup. As with DILocations, we don't want to store duplicate information, and therefore when we want to get the index for a DILocation, we find the appropriate bucket using the FnInlinedAtSrcLocRanges vector - which gives us the start of each inlinedAt bucket - and binary search within that bucket. This gives us reasonably fast lookup of O(log n) - slower than the existing O(1) lookup time, but this is actually a very uncommon operation after the initial step of generating the IR in-memory from the frontend/parsing, so this is not a big deal.

#### InlinedAt SrcLocs

Generally, the complete set of SrcLoc data is fixed when a function is fully generated - it is very rare that optimizations add new source locations, they only potentially reduce the set by deleting instructions or dropping source locations over time. The major exception to this is inlining, in which we suddenly add a set of source locations from one function into another function. This is major cost in the current model; every DILocation that appears in the callee must have a new DILocation generated with a modified InlinedAt field - including DILocations that were themselves inlined from other functions.

The new model cannot avoid the duplicating approach, but it does simplify it significantly, and manages to perform the full inlining step with only three allocationsby directly appending the callee's location vectors to the caller's, adding a fixed offset to each inlined DebugLoc so that it points to the index of the item in the extended list. In between the caller and callee SrcLocs, we add a new SrcLoc for the inlined call. This SrcLoc uniquely identifies this inlined instance of the callee; if we inline the same function at the same source location again, it will have an identical SrcLoc at a different index (i.e. these are not uniqued).

#### Non-Normalized SrcLocs

Although we usually expect FnSrcLocs to be normalized, this is not always the case; if we ever have to add new SrcLocs, we have to choose between either normalizing immediately - which requires us to reallocate the vector and re-index every DebugLoc in the function - or accepting a temporary non-normal state. The prototype opts for the latter, using an additional vector `NonNormalFnSrcLocs` to store any new SrcLocs. These are not added very often - the only times where this happens are when merging source locations with identical lines but different columns, or when adding a line at the `DISubprogram::getLine()` or `getScopeLine()`, which we do not add to `FnSrcLocs` by default. The non-normal vector is not sorted or tightly allocated, on the basis that it should be small enough for reallocation and linear search to be acceptable costs. Before any operation that requires normality - inlining or printing - we normalize the DISubprogram, taking the expensive steps of merging in the non-normal SrcLocs into sorted positions and re-indexing. This should almost never happen more than once per function per build, and if no non-normal SrcLocs have been created then it doesn't need to happen at all.

### Notable Issues

#### Loop Metadata

It would be convenient if DILocations only appeared attached directly to instructions, but unfortunately there is an extra case to deal with: instructions may have loop metadata attached to them; this metadata is *always* a distinct MDTuple, and its first argument is always itself. The first argument *may* be a DILocation, representing the start of the loop, and if so then the second argument *may* also be a DILocation, representing the end of the loop. This is less convenient than the normal case: as an MDTuple, the loop metadata *must* be comprised of Metadata pointers, which means that we cannot replace the DILocation pointers with a pair of 4-byte indexes. The approach I've taken for this in the prototype is to convert the pair of indexes into a single 8-byte value, and replace the DILocation* arguments with ConstantAsMetadata* of the combined value. This isn't super user-friendly - a user will see a large random number that they need to convert to binary or hex to interpret. The long-term solution to this is probably to replace them with either a raw MDNode containing two constant values for each argument, or to create a specialized Metadata class specifically for this case, to give a more user-friendly view. Either way, this adds extra costs, but loop metadata is not common enough for this to cause performance issues.

#### Merged InlinedAts

In theory, for every inlined function call, there should be exactly one distinct DILocation, or one SrcLoc with this model. This is not always the case however - there is a specific exception when merging partially-matching inlining chains. That is, if we merge two instructions that originated from two different calls to the same function, we may create a *new* inlinedAt DILocation, which has a line 0 location and is *not* distinct, meaning that it does not uniquely correspond to one inlined instance (or one merge of two specific inlined instances). For now, we preserve this behaviour and create a new LocScope with `InlinedAt=(0, <Merged Call Scope>)`. This breaks some assumptions about how we use the `InlinedAt` field to efficiently search for SrcLocs, but it can be handled by special-casing the search logic for `InlinedAt.SrcLocIndex = 0`.

#### nodebug Functions

This model of DebugLocs relies on using a function's DISubprogram to store all source location information contained in that function; however, sometimes a function with no DISubprogram can legitimately contain source locations, which is when a `nodebug` function inlines a function with debug info. This is a case that unfortunately needs to be handled; I've not implemented a solution yet, but a valid possibility would be to create a "dummy" DISubprogram for nodebug functions, which contains no fields except for the source location fields, allowing us to preserve this information in case we need it later without otherwise creating debug info for that function; an approach like this would cause some trickiness with other code that expects `F->getSubprogram()` to be null for all nodebug functions. An alternative would be to store this information directly on the `Function` itself, which simplifies matters significantly *but* adds a memory cost to builds without debug info, which is also undesirable. Finally, we could add a "pseudo" DISubprogram alternative that acts as a base class of DISubprogram; we modify `F->getSubprogram()` to use a `dyn_cast` such that it still returns a nullptr value for the pseudo DISubprogram, but create a different function that gives us access to the pseudo-version when we just need to examine DebugLocs. This is probably the best overall option, but it's also the most complicated to review.

### Other Notes

#### Lossiness

This approach is technically lossy - we reduce all line-0 locations to a single SrcLoc, which means we lose the ability to represent any locations with line=0 and any other non-0 fields, e.g. line 0, column 5. In reality, this is not incorrect behaviour, as we intentionally do not use the fields other than line when line=0 (and we generally zero-out these fields anyway), so this is an "efficient" loss of data.

#### GetAllMetadata

The function `Value::getAllMetadata` returns a vector of all the Metadata* associated with that value; for Instructions, this includes the attached DILocation. This interface is unfortunately impossible for us to match after this rewrite, as source locations no longer are Metadata. This doesn't matter for code that doesn't *need* to observe source locations, but in all other cases it's necessary to substitute this functionality. Typically this means adding new code that handles DebugLocs specifically, which we do when printing; any consumers of LLVM's API must also be updated to handle this, which may include some external projects.

### Future Changes

There are a number of missing features at the moment that must be implemented in future:

- Direct bitcode printing/parsing support.
    - Currently we convert to/from DILocations during bitcode printing and parsing, which is costly in time and memory. Having a direct representation in bitcode would be ideal, and hopefully more efficient than the current model.
- Representing DebugLocs in nodebug functions (see above).
- ISel + MIR Codegen support - a lot of scaffolding has been put in place, but the effort hasn't been put in to make it work yet.

Furthermore, there are some changes to simplify or otherwise improve this approach that are likely to be worthwhile once the core functionality is complete:

- Performing a cleanup of DILocRef uses, so that we always use DebugLoc where there are many stored at once (instead of repeatedly storing the same `DISubprogram` pointer).
- Creating a more efficient alternative of DILocRef: DILocRef has a DISubprogram*, but in reality it would be faster if it had references to the beginning of the location arrays; this makes for an *even-fatter-pointer*, but one that can look up source locations just as efficiently as DILocations (or morseo thanks to data locality!).
- There are things we can do to reduce the costs of the source location vectors.
    - Sorting SrcLocs may be unnecessary - there are *very* few cases where we actually need to search the SrcLoc vector, and the few cases we do look like they can just be handled as special-cases.
    - Bucketing SrcLocs by InlinedAt may be unnecessary - although the buckets naturally extend from our inlining approach, they don't actually serve a purpose if we don't need to sort SrcLocs. We still need to track the inlined-at SrcLocs, but that could easily be done by using the InlinedAtSrcLocs vector. This could also give us opportunities to deduplicate SrcLocs from multipled inlined instances of the same function, further saving memory.
    - LocScopes may be more in need of normalization, because we're more likely to need to look them up as we go along - a viable approach would be either sorting the array s.t. every parent scope appears before every one of its child scopes (this is likely to be the case by default, but it isn't enforced), which gives us some minor savings (we can linear search a smaller section of the array), or more aggressively if we can spare the bits then we could add an index alongside each scope, pointing to the entry for that scope's parent scope.
      - This could also allow for more efficient implementations of certain scope operations, such as "nearest common scope", by allowing us to solve them purely by stepping back through an array without having to actually dereference any scopes. If we make Scopes into function-local metadata as well, then this would be a very natural development.
- There exist other metadata types that are effectively function-local: Loop MDNodes, DIScopes and DIAssignID are each only relevant within a particular function context, and therefore could also be merged into the DISubprogram. These classes are less significant in terms of memory usage than DILocations, but DIScopes and DIAssignID do still consume a measurable % of memory - we may scrape out extra savings by adding them afterwards.
