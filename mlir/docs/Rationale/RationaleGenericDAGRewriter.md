# Generic DAG Rewriter Infrastructure Rationale

This document details the rationale behind a general DAG-to-DAG rewrite
infrastructure for MLIR. For up-to-date documentation on the user facing API,
please look at the main [Pattern Rewriting document](../PatternRewriter.md).

## Introduction and Motivation

The goal of a compiler IR is to represent code - at various levels of
abstraction which pose different sets of tradeoffs in terms of representational
capabilities and ease of transformation. However, the ability to represent code
is not itself very useful - you also need to be able to implement those
transformations.

There are many different types of compiler transformations, but this document
focuses on a particularly important class of transformation that comes up
repeatedly at scale, and is important for the goals of MLIR: matching one DAG of
operations, and replacing with another. This is an integral part of many
compilers and necessary for peephole optimizations like "eliminate identity
nodes" or "replace x+0 with x", a generalized canonicalization framework (e.g.
Instruction Combiner in LLVM), as well as a useful abstraction to implement
optimization algorithms for optimization algorithms for IR at multiple levels.

A particular strength of MLIR (and a major difference vs other compiler
infrastructures like LLVM, GCC, XLA, TensorFlow, etc) is that it uses a single
compiler IR to represent code at multiple levels of abstraction: an MLIR
operation can be a "TensorFlow operation", an "XLA HLO", an Affine Loop Nest, an
LLVM IR instruction (transitively including X86, Lanai, PTX, and other target
specific instructions), or anything else that the MLIR operation system can
reasonably express. Given that MLIR spans such a wide range of different problem
scopes, a single infrastructure for performing graph-to-graph rewrites can help
solve many diverse domain challenges.

[Static single assignment](https://en.wikipedia.org/wiki/Static_single_assignment_form)
(SSA) representations like MLIR make it easy to access the operands and "users"
of an operation. As such, a natural abstraction for these graph-to-graph
rewrites is that of DAG pattern matching: clients define DAG tile patterns
(where a tile is a sequence of operations defining a subgraph of the DAG), and
each pattern includes a result DAG to produce and the cost of the result (or,
inversely, the benefit of doing the replacement). A common infrastructure
efficiently finds and performs the rewrites.

While this concept is simple, the details are more nuanced. This document
defines and explores a set of abstractions that can solve a wide range of
different problems, and be applied to many different sorts of problems that MLIR
is - and is expected to - face over time. We do this by separating the pattern
application algorithm from the "driver" of the computation loop, and make space
for the patterns to be defined declaratively.

### Constant folding

A degenerate but pervasive case of DAG-to-DAG pattern matching is constant
folding: an operation whose operands contain constants can often be folded to a
result constant value.

MLIR operations may override a
[`fold`](../Canonicalization.md/#canonicalizing-with-the-fold-method) routine, which
exposes a simpler API compared to a general DAG-to-DAG pattern matcher, and
allows for it to be applicable in cases that a generic matcher would not. For
example, a DAG-rewrite can remove arbitrary nodes in the current function, which
could invalidate iterators. Constant folding as an API does not remove any
nodes, it just provides a (list of) constant values and allows the clients to
update their data structures as necessary.

## Related Work

There is a huge amount of related work to consider, given that nearly every
compiler in existence has to solve this problem many times over. One unifying
problem is that all of these systems are designed to solve one particular, and
usually, narrow problem: MLIR on the other hand would like to solve many of
these problems within a single infrastructure. Here are a few related graph
rewrite systems, along with the pros and cons of their work (The most similar
design to the infrastructure present in MLIR is the LLVM DAG-to-DAG instruction
selection algorithm).

### AST-Level Pattern Matchers

The literature is full of source-to-source translators which transform
identities in order to improve performance (e.g. transforming `X*0` into `0`).
One large example is the GCC `fold` function, which performs
[many optimizations](https://github.com/gcc-mirror/gcc/blob/master/gcc/fold-const.c)
on ASTs. Clang has
[similar routines](https://clang.llvm.org/docs/InternalsManual.html#constant-folding-in-the-clang-ast)
for simple constant folding of expressions (as required by the C++ standard) but
doesn't perform general optimizations on its ASTs.

The primary downside of AST optimizers is that you can't see across operations
that have multiple uses. It is
[well known in literature](https://llvm.org/pubs/2008-06-LCTES-ISelUsingSSAGraphs.pdf)
that DAG pattern matching is more powerful than tree pattern matching, but on
the other hand, DAG pattern matching can lead to duplication of computation
which needs to be checked for.

### "Combiners" and other peephole optimizers

Compilers end up with a lot of peephole optimizers for various things, e.g. the
GCC
["combine" routines](https://github.com/gcc-mirror/gcc/blob/master/gcc/combine.c)
(which try to merge two machine instructions into a single one), the LLVM
[Inst Combine](https://github.com/llvm/llvm-project/tree/main/llvm/lib/Transforms/InstCombine)
[pass](https://llvm.org/docs/Passes.html#instcombine-combine-redundant-instructions),
LLVM's
[DAG Combiner](https://github.com/llvm-mirror/llvm/blob/master/lib/CodeGen/SelectionDAG/DAGCombiner.cpp),
the Swift compiler's
[SIL Combiner](https://github.com/apple/swift/tree/master/lib/SILOptimizer/SILCombiner),
etc. These generally match one or more operations and produce zero or more
operations as a result. The LLVM
[Legalization](https://github.com/llvm/llvm-project/tree/main/llvm/lib/CodeGen/SelectionDAG)
infrastructure has a different outer loop but otherwise works the same way.

These passes have a lot of diversity, but also have a unifying structure: they
mostly have a worklist outer loop which visits operations. They then use a
visitor pattern (or equivalent) to switch over the class of operation and
dispatch to a method. That method contains a long list of hand-written C++ code
that pattern-matches various special cases. LLVM introduced a "match" function
that allows writing patterns in a somewhat more declarative style using template
metaprogramming (MLIR has similar facilities). Here's a simple example:

```c++
  // Y - (X + 1) --> ~X + Y
  if (match(Op1, m_OneUse(m_Add(m_Value(X), m_One()))))
    return BinaryOperator::CreateAdd(Builder.CreateNot(X), Op0);
```

Here is a somewhat more complicated one (this is not the biggest or most
complicated :)

```c++
  // C2 is ODD
  // LHS = XOR(Y,C1), Y = AND(Z,C2), C1==(C2+1) => LHS == NEG(OR(Z, ~C2))
  // ADD(LHS, RHS) == SUB(RHS, OR(Z, ~C2))
  if (match(LHS, m_Xor(m_Value(Y), m_APInt(C1))))
    if (C1->countTrailingZeros() == 0)
      if (match(Y, m_And(m_Value(Z), m_APInt(C2))) && *C1 == (*C2 + 1)) {
        Value NewOr = Builder.CreateOr(Z, ~(*C2));
        return Builder.CreateSub(RHS, NewOr, "sub");
      }
```

These systems are simple to set up, and pattern matching templates have some
advantages (they are extensible for new sorts of sub-patterns, look compact at
point of use). On the other hand, they have lots of well known problems, for
example:

*   These patterns are very error prone to write, and contain lots of
    redundancies.
*   The IR being matched often has identities (e.g. when matching commutative
    operators) and the C++ code has to handle it manually - take a look at
    [the full code](https://github.com/llvm/llvm-project/blob/c0b5000bd848303320c03f80fbf84d71e74518c9/llvm/lib/Transforms/InstCombine/InstCombineAddSub.cpp#L767)
    for `checkForNegativeOperand` that defines the second pattern).
*   The matching code compiles slowly, both because it generates tons of code
    and because the templates instantiate slowly.
*   Adding new patterns (e.g. for count leading zeros in the example above) is
    awkward and doesn't often happen.
*   The cost model for these patterns is not really defined - it is emergent
    based on the order the patterns are matched in code.
*   They are non-extensible without rebuilding the compiler.
*   It isn't practical to apply theorem provers and other tools to these
    patterns - they cannot be reused for other purposes.

In addition to structured "combiners" like these, there are lots of ad-hoc
systems like the
[LLVM Machine code peephole optimizer](http://llvm.org/viewvc/llvm-project/llvm/trunk/lib/CodeGen/PeepholeOptimizer.cpp?view=markup)
which are related.

### LLVM's DAG-to-DAG Instruction Selection Infrastructure

The instruction selection subsystem in LLVM is the result of many years worth of
iteration and discovery, driven by the need for LLVM to support code generation
for lots of targets, the complexity of code generators for modern instruction
sets (e.g. X86), and the fanatical pursuit of reusing code across targets. Eli
Bendersky wrote a
[nice short overview](https://eli.thegreenplace.net/2013/02/25/a-deeper-look-into-the-llvm-code-generator-part-1)
of how this works, and the
[LLVM documentation](https://llvm.org/docs/CodeGenerator.html#select-instructions-from-dag)
describes it in more depth including its advantages and limitations. It allows
writing patterns like this.

```
def : Pat<(or GR64:$src, (not (add GR64:$src, 1))),
          (BLCI64rr GR64:$src)>;
```

This example defines a matcher for the
["blci" instruction](https://en.wikipedia.org/wiki/Bit_Manipulation_Instruction_Sets#TBM_\(Trailing_Bit_Manipulation\))
in the
[X86 target description](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/X86/X86InstrInfo.td),
there are many others in that file (look for `Pat<>` patterns, since they aren't
entangled in details of the compiler like assembler/disassembler generation
logic).

For the purposes of MLIR, there is much to like about this system, for example:

*   It is defined in a declarative format.
*   It is extensible to target-defined operations.
*   It automates matching across identities, like commutative patterns.
*   It allows custom abstractions and intense factoring of target-specific
    commonalities.
*   It generates compact code - it compiles into a state machine, which is
    interpreted.
*   It allows the instruction patterns to be defined and reused for multiple
    purposes.
*   The patterns are "type checked" at compile time, detecting lots of bugs
    early and eliminating redundancy from the pattern specifications.
*   It allows the use of general C++ code for weird/complex cases.

While there is a lot that is good here, there are also a few undesirable bits:

*   The representation is specifically designed and only applicable for
    instruction selection, meaning that the directly adjacent problems like the
    DAGCombiner and Legalizer can't use it.
*   This isn't extensible at compiler runtime, you have to rebuild the compiler
    to extend it.
*   The error messages when failing to match a pattern
    [are not exactly optimal](https://www.google.com/search?q=llvm+cannot+select).
*   It has lots of implementation problems and limitations (e.g. can't write a
    pattern for a multi-result operation) as a result of working with the
    awkward SelectionDAG representation and being designed and implemented on
    demand.
*   Organic growth over time has left lots of sharp edges.

### Summary

MLIR faces a wide range of pattern matching and graph rewrite problems, and one
of the major advantages of having a common representation for code at multiple
levels is that it allows for investing in - and highly leveraging - a single
infrastructure for doing this sort of work.

## Goals

We'd like the to encompass many problems in the MLIR space, including 1-to-N
expansions (e.g. such as in type legalization during instruction selection when
an add of one bit width may be split into multiple adds of a smaller bit width),
M-to-1 patterns (e.g. when converting a multiply+add into a single muladd
operation), as well as general M-to-N patterns (e.g. instruction selection for
target instructions). Patterns have a benefit associated with them, and the
common infrastructure should be responsible for sorting out the highest benefit
match for a given application.

We separate the task of picking a particular optimal pattern from a given root
node, the algorithm used to rewrite an entire graph given a particular set of
goals, and the definition of the patterns themselves. We do this because DAG
tile pattern matching is NP complete. Additionally, we would like to support
iterative rewrite algorithms that progressively transform the input program
through multiple steps. Furthermore, we would like to support many different
sorts of clients across the MLIR stack, and they may have different tolerances
for compile time cost, different demands for optimality, and other algorithmic
goals or constraints.

We aim for MLIR transformations to be easy to implement and reduce the
likelihood for compiler bugs. We expect there to be a very large number of
patterns that are defined over time, and we believe that these sorts of patterns
will have a very large number of legality/validity constraints - many of which
are difficult to reason about in a consistent way, may be target specific, and
whose implementation may be particularly bug-prone. As such, we aim to design
the API around pattern definition to be simple, resilient to programmer errors,
and allow separation of concerns between the legality of the nodes generated
from the idea of the pattern being defined.

Finally, error handling is a topmost concern, we want pattern match failures to
be diagnosable in a reasonable way. This is a difficult problem in general, as
the space of malfunction is too great to be fully enumerated and handled
optimally, but MLIR is already designed to represent the provenance of an
operation well. The aim of the pattern rewriting infrastructure is simply to
propagate that provenance information precisely, as well as diagnose pattern
match failures with the rationale for why a set of patterns do not apply.

### Non goals

The pattern infrastructure does not aim to solve all compiler problems, it is
simply a DAG-to-DAG pattern matching system. Compiler algorithms that require
global dataflow analysis (e.g. common subexpression elimination, conditional
constant propagation, and many many others) will not be directly solved by this
infrastructure.

This infrastructure is limited to DAG patterns, which (by definition) prevent
the patterns from seeing across cycles in a graph. In an SSA-based IR like MLIR,
this means that these patterns don't see across basic block arguments. We
consider this acceptable given the set of problems we are trying to solve - we
don't know of any other system that attempts to do so, and consider the payoff
of worrying about this to be low.

This design includes the ability for DAG patterns to have associated benefits,
but those benefits are defined in terms of magic numbers (typically equal to the
number of nodes being replaced). For any given application, the units of magic
numbers will have to be defined.
