=====================================
Performance Tips for Frontend Authors
=====================================

.. contents::
   :local:
   :depth: 2

Abstract
========

The intended audience of this document is developers of language frontends 
targeting LLVM IR. This document is home to a collection of tips on how to 
generate IR that optimizes well.  As with any optimizer, LLVM has its strengths
and weaknesses.  In some cases, surprisingly small changes in the source IR 
can have a large effect on the generated code.  

Avoid loads and stores of large aggregate type
================================================

LLVM currently does not optimize well loads and stores of large :ref:`aggregate
types <t_aggregate>` (i.e. structs and arrays).  As an alternative, consider 
loading individual fields from memory.

Aggregates that are smaller than the largest (performant) load or store 
instruction supported by the targeted hardware are well supported.  These can 
be an effective way to represent collections of small packed fields.  

Prefer zext over sext when legal
==================================

On some architectures (X86_64 is one), sign extension can involve an extra 
instruction whereas zero extension can be folded into a load.  LLVM will try to
replace a sext with a zext when it can be proven safe, but if you have 
information in your source language about the range of a integer value, it can 
be profitable to use a zext rather than a sext.  

Alternatively, you can :ref:`specify the range of the value using metadata 
<range-metadata>` and LLVM can do the sext to zext conversion for you.

Zext GEP indices to machine register width
============================================

Internally, LLVM often promotes the width of GEP indices to machine register
width.  When it does so, it will default to using sign extension (sext) 
operations for safety.  If your source language provides information about 
the range of the index, you may wish to manually extend indices to machine 
register width using a zext instruction.

Other things to consider
=========================

#. Make sure that a DataLayout is provided (this will likely become required in
   the near future, but is certainly important for optimization).

#. Add nsw/nuw flags as appropriate.  Reasoning about overflow is 
   generally hard for an optimizer so providing these facts from the frontend 
   can be very impactful.  

#. Use fast-math flags on floating point operations if legal.  If you don't 
   need strict IEEE floating point semantics, there are a number of additional 
   optimizations that can be performed.  This can be highly impactful for 
   floating point intensive computations.

#. Use inbounds on geps.  This can help to disambiguate some aliasing queries.

#. Add noalias/align/dereferenceable/nonnull to function arguments and return 
   values as appropriate

#. Mark functions as readnone/readonly or noreturn/nounwind when known.  The 
   optimizer will try to infer these flags, but may not always be able to.  
   Manual annotations are particularly important for external functions that 
   the optimizer can not analyze.

#. Use ptrtoint/inttoptr sparingly (they interfere with pointer aliasing 
   analysis), prefer GEPs

#. Use the lifetime.start/lifetime.end and invariant.start/invariant.end 
   intrinsics where possible.  Common profitable uses are for stack like data 
   structures (thus allowing dead store elimination) and for describing 
   life times of allocas (thus allowing smaller stack sizes).  

#. Use pointer aliasing metadata, especially tbaa metadata, to communicate 
   otherwise-non-deducible pointer aliasing facts

#. Use the "most-private" possible linkage types for the functions being defined
   (private, internal or linkonce_odr preferably)

#. Mark invariant locations using !invariant.load and TBAA's constant flags

#. Prefer globals over inttoptr of a constant address - this gives you 
   dereferencability information.  In MCJIT, use getSymbolAddress to provide 
   actual address.

#. Be wary of ordered and atomic memory operations.  They are hard to optimize 
   and may not be well optimized by the current optimizer.  Depending on your
   source language, you may consider using fences instead.

#. If calling a function which is known to throw an exception (unwind), use 
   an invoke with a normal destination which contains an unreachable 
   instruction.  This form conveys to the optimizer that the call returns 
   abnormally.  For an invoke which neither returns normally or requires unwind
   code in the current function, you can use a noreturn call instruction if 
   desired.  This is generally not required because the optimizer will convert
   an invoke with an unreachable unwind destination to a call instruction.

#. If you language uses range checks, consider using the IRCE pass.  It is not 
   currently part of the standard pass order.

#. For languages with numerous rarely executed guard conditions (e.g. null 
   checks, type checks, range checks) consider adding an extra execution or 
   two of LoopUnswith and LICM to your pass order.  The standard pass order, 
   which is tuned for C and C++ applications, may not be sufficient to remove 
   all dischargeable checks from loops.

#. Use profile metadata to indicate statically known cold paths, even if 
   dynamic profiling information is not available.  This can make a large 
   difference in code placement and thus the performance of tight loops.

#. When generating code for loops, try to avoid terminating the header block of
   the loop earlier than necessary.  If the terminator of the loop header 
   block is a loop exiting conditional branch, the effectiveness of LICM will
   be limited for loads not in the header.  (This is due to the fact that LLVM 
   may not know such a load is safe to speculatively execute and thus can't 
   lift an otherwise loop invariant load unless it can prove the exiting 
   condition is not taken.)  It can be profitable, in some cases, to emit such 
   instructions into the header even if they are not used along a rarely 
   executed path that exits the loop.  This guidance specifically does not 
   apply if the condition which terminates the loop header is itself invariant,
   or can be easily discharged by inspecting the loop index variables.

#. In hot loops, consider duplicating instructions from small basic blocks 
   which end in highly predictable terminators into their successor blocks.  
   If a hot successor block contains instructions which can be vectorized 
   with the duplicated ones, this can provide a noticeable throughput
   improvement.  Note that this is not always profitable and does involve a 
   potentially large increase in code size.

#. Avoid high in-degree basic blocks (e.g. basic blocks with dozens or hundreds
   of predecessors).  Among other issues, the register allocator is known to 
   perform badly with confronted with such structures.  The only exception to 
   this guidance is that a unified return block with high in-degree is fine.

#. When checking a value against a constant, emit the check using a consistent
   comparison type.  The GVN pass *will* optimize redundant equalities even if
   the type of comparison is inverted, but GVN only runs late in the pipeline.
   As a result, you may miss the opportunity to run other important 
   optimizations.  Improvements to EarlyCSE to remove this issue are tracked in 
   Bug 23333.

#. Avoid using arithmetic intrinsics unless you are *required* by your source 
   language specification to emit a particular code sequence.  The optimizer 
   is quite good at reasoning about general control flow and arithmetic, it is
   not anywhere near as strong at reasoning about the various intrinsics.  If 
   profitable for code generation purposes, the optimizer will likely form the 
   intrinsics itself late in the optimization pipeline.  It is *very* rarely 
   profitable to emit these directly in the language frontend.  This item
   explicitly includes the use of the :ref:`overflow intrinsics <int_overflow>`.

#. Avoid using the :ref:`assume intrinsic <int_assume>` until you've 
   established that a) there's no other way to express the given fact and b) 
   that fact is critical for optimization purposes.  Assumes are a great 
   prototyping mechanism, but they can have negative effects on both compile 
   time and optimization effectiveness.  The former is fixable with enough 
   effort, but the later is fairly fundamental to their designed purpose.

p.s. If you want to help improve this document, patches expanding any of the 
above items into standalone sections of their own with a more complete 
discussion would be very welcome.  


Adding to this document
=======================

If you run across a case that you feel deserves to be covered here, please send
a patch to `llvm-commits
<http://lists.llvm.org/mailman/listinfo/llvm-commits>`_ for review.

If you have questions on these items, please direct them to `llvm-dev 
<http://lists.llvm.org/mailman/listinfo/llvm-dev>`_.  The more relevant 
context you are able to give to your question, the more likely it is to be 
answered.

