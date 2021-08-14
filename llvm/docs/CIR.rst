===============================
CIR - Clang IR Design and Implementation
===============================

.. contents::
   :local:

Introduction
============

This document aims to provide an overview of the design and
implementation of a Clang IR, a high level IR allowing more
analysis and future optimizations.

Usage in Clang
==============


Usage in Clang happens right now as part of replacing current
IssueWarnings
AnalysisWarnings.IssueWarnings

CFG usage in ``AnalysisBasedWarning.cpp`` to use CIR instead of
Clang's CFG, as part of ``PopFunctionScopeInfo``. 