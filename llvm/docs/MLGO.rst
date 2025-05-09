====
MLGO
====

Introduction
============

MLGO is a framework for integrating ML techniques systematically in LLVM. It is
designed primarily to replace heuristics within LLVM with machine learned
models. Currently there is upstream infrastructure for the following
heuristics:

* Inlining for size
* Register allocation (LLVM greedy eviction heuristic) for performance

This document is an outline of the tooling that composes MLGO.

Corpus Tooling
==============

..
    TODO(boomanaiden154): Write this section.

Model Runner Interfaces
=======================

..
    TODO(mtrofin): Write this section.
