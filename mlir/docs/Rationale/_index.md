# Rationale

This section contains a collection of documents describing the motivation and
rationale for some of the design decisions behind MLIR.

[MLIR: Incremental Application to Graph Algorithms in ML Frameworks](MLIRForGraphAlgorithms.md)
:   A discussion of how the adoption of MLIR can be taken in incremental steps,
    with each step providing tangible benefits along the way. Refutes the idea
    that full adoption of MLIR is required before we can reap the benefits of
    MLIR.

[MLIR Rationale](Rationale.md)
:   Introduces the motivation for MLIR and captures design discussions and
    decisions made for various core features of MLIR.

[Generic DAG Rewriter Infrastructure Rationale](RationaleGenericDAGRewriter.md)
:   Details the rationale behind a general DAG-to-DAG rewrite infrastructure for
    MLIR.

[Linalg Dialect Rationale: The Case for Compiler-Friendly Custom Operations](RationaleLinalgDialect.md)
:   Describes the key design principles that led to the existing implementation
    of Linalg and lessons learned along the way.

[MLIR: The case for a simplified polyhedral form](RationaleSimplifiedPolyhedralForm.md)
:   An early design proposal exploring the tradeoffs of using a simplified form
    for polyhedral compiler techniques in MLIR instead of the traditional
    polyhedral schedule list form.

[Usage of 'const' in MLIR, for core IR types](UsageOfConst.md)
:   Explains the rationale for eschewing the use of `const` entirely for the
    core IR types in MLIR.
