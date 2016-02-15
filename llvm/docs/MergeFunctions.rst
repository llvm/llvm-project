=================================
MergeFunctions pass, how it works
=================================

.. contents::
   :local:

Introduction
============
Sometimes code contains equal functions, or functions that does exactly the same
thing even though they are non-equal on the IR level (e.g.: multiplication on 2
and 'shl 1'). It could happen due to several reasons: mainly, the usage of
templates and automatic code generators. Though, sometimes user itself could
write the same thing twice :-)

The main purpose of this pass is to recognize such functions and merge them.

Why would I want to read this document?
---------------------------------------
Document is the extension to pass comments and describes the pass logic. It
describes algorithm that is used in order to compare functions, it also
explains how we could combine equal functions correctly, keeping module valid.

Material is brought in top-down form, so reader could start learn pass from
ideas and end up with low-level algorithm details, thus preparing him for
reading the sources.

So main goal is do describe algorithm and logic here; the concept. This document
is good for you, if you *don't want* to read the source code, but want to
understand pass algorithms. Author tried not to repeat the source-code and
cover only common cases, and thus avoid cases when after minor code changes we
need to update this document.


What should I know to be able to follow along with this document?
-----------------------------------------------------------------

Reader should be familiar with common compile-engineering principles and LLVM
code fundamentals. In this article we suppose reader is familiar with
`Single Static Assingment <http://en.wikipedia.org/wiki/Static_single_assignment_form>`_
concepts. Understanding of
`IR structure <http://llvm.org/docs/LangRef.html#high-level-structure>`_ is
also important.

We will use such terms as
"`module <http://llvm.org/docs/LangRef.html#high-level-structure>`_",
"`function <http://llvm.org/docs/ProgrammersManual.html#the-function-class>`_",
"`basic block <http://en.wikipedia.org/wiki/Basic_block>`_",
"`user <http://llvm.org/docs/ProgrammersManual.html#the-user-class>`_",
"`value <http://llvm.org/docs/ProgrammersManual.html#the-value-class>`_",
"`instruction <http://llvm.org/docs/ProgrammersManual.html#the-instruction-class>`_".

As a good start point, Kaleidoscope tutorial could be used:

:doc:`tutorial/index`

Especially it's important to understand chapter 3 of tutorial:

:doc:`tutorial/LangImpl3`

Reader also should know how passes work in LLVM, they could use next article as
a reference and start point here:

:doc:`WritingAnLLVMPass`

What else? Well perhaps reader also should have some experience in LLVM pass
debugging and bug-fixing.

What I gain by reading this document?
-------------------------------------
Main purpose is to provide reader with comfortable form of algorithms
description, namely the human reading text. Since it could be hard to
understand algorithm straight from the source code: pass uses some principles
that have to be explained first.

Author wishes to everybody to avoid case, when you read code from top to bottom
again and again, and yet you don't understand why we implemented it that way.

We hope that after this article reader could easily debug and improve
MergeFunctions pass and thus help LLVM project.

Narrative structure
-------------------
Article consists of three parts. First part explains pass functionality on the
top-level. Second part describes the comparison procedure itself. The third
part describes the merging process.

In every part author also tried to put the contents into the top-down form.
First, the top-level methods will be described, while the terminal ones will be
at the end, in the tail of each part. If reader will see the reference to the
method that wasn't described yet, they will find its description a bit below.

Basics
======

How to do it?
-------------
Do we need to merge functions? Obvious thing is: yes that's a quite possible
case, since usually we *do* have duplicates. And it would be good to get rid of
them. But how to detect such a duplicates? The idea is next: we split functions
onto small bricks (parts), then we compare "bricks" amount, and if it equal,
compare "bricks" themselves, and then do our conclusions about functions
themselves.

What the difference it could be? For example, on machine with 64-bit pointers
(let's assume we have only one address space),  one function stores 64-bit
integer, while another one stores a pointer. So if the target is a machine
mentioned above, and if functions are identical, except the parameter type (we
could consider it as a part of function type), then we can treat ``uint64_t``
and``void*`` as equal.

It was just an example; possible details are described a bit below.

As another example reader may imagine two more functions. First function
performs multiplication on 2, while the second one performs arithmetic right
shift on 1.

Possible solutions
^^^^^^^^^^^^^^^^^^
Let's briefly consider possible options about how and what we have to implement
in order to create full-featured functions merging, and also what it would
meant for us.

Equal functions detection, obviously supposes "detector" method to be
implemented, latter should answer the question "whether functions are equal".
This "detector" method consists of tiny "sub-detectors", each of them answers
exactly the same question, but for function parts.

As the second step, we should merge equal functions. So it should be a "merger"
method. "Merger" accepts two functions *F1* and *F2*, and produces *F1F2*
function, the result of merging.

Having such a routines in our hands, we can process whole module, and merge all
equal functions.

In this case, we have to compare every function with every another function. As
reader could notice, this way seems to be quite expensive. Of course we could
introduce hashing and other helpers, but it is still just an optimization, and
thus the level of O(N*N) complexity.

Can we reach another level? Could we introduce logarithmical search, or random
access lookup? The answer is: "yes".

Random-access
"""""""""""""
How it could be done? Just convert each function to number, and gather all of
them in special hash-table. Functions with equal hash are equal. Good hashing
means, that every function part must be taken into account. That means we have
to convert every function part into some number, and then add it into hash.
Lookup-up time would be small, but such approach adds some delay due to hashing
routine.

Logarithmical search
""""""""""""""""""""
We could introduce total ordering among the functions set, once we had it we
could then implement a logarithmical search. Lookup time still depends on N,
but adds a little of delay (*log(N)*).

Present state
"""""""""""""
Both of approaches (random-access and logarithmical) has been implemented and
tested. And both of them gave a very good improvement. And what was most
surprising, logarithmical search was faster; sometimes up to 15%. Hashing needs
some extra CPU time, and it is the main reason why it works slower; in most of
cases total "hashing" time was greater than total "logarithmical-search" time.

So, preference has been granted to the "logarithmical search".

Though in the case of need, *logarithmical-search* (read "total-ordering") could
be used as a milestone on our way to the *random-access* implementation.

Every comparison is based either on the numbers or on flags comparison. In
*random-access* approach we could use the same comparison algorithm. During
comparison we exit once we find the difference, but here we might have to scan
whole function body every time (note, it could be slower). Like in
"total-ordering", we will track every numbers and flags, but instead of
comparison, we should get numbers sequence and then create the hash number. So,
once again, *total-ordering* could be considered as a milestone for even faster
(in theory) random-access approach.

MergeFunctions, main fields and runOnModule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are two most important fields in class:

``FnTree``  – the set of all unique functions. It keeps items that couldn't be
merged with each other. It is defined as:

``std::set<FunctionNode> FnTree;``

Here ``FunctionNode`` is a wrapper for ``llvm::Function`` class, with
implemented “<” operator among the functions set (below we explain how it works
exactly; this is a key point in fast functions comparison).

``Deferred`` – merging process can affect bodies of functions that are in
``FnTree`` already. Obviously such functions should be rechecked again. In this
case we remove them from ``FnTree``, and mark them as to be rescanned, namely
put them into ``Deferred`` list.

runOnModule
"""""""""""
The algorithm is pretty simple:

1. Put all module's functions into the *worklist*.

2. Scan *worklist*'s functions twice: first enumerate only strong functions and
then only weak ones:

   2.1. Loop body: take function from *worklist*  (call it *FCur*) and try to
   insert it into *FnTree*: check whether *FCur* is equal to one of functions
   in *FnTree*. If there *is* equal function in *FnTree* (call it *FExists*):
   merge function *FCur* with *FExists*. Otherwise add function from *worklist*
   to *FnTree*.

3. Once *worklist* scanning and merging operations is complete, check *Deferred*
list. If it is not empty: refill *worklist* contents with *Deferred* list and
do step 2 again, if *Deferred* is empty, then exit from method.

Comparison and logarithmical search
"""""""""""""""""""""""""""""""""""
Let's recall our task: for every function *F* from module *M*, we have to find
equal functions *F`* in shortest time, and merge them into the single function.

Defining total ordering among the functions set allows to organize functions
into the binary tree. The lookup procedure complexity would be estimated as
O(log(N)) in this case. But how to define *total-ordering*?

We have to introduce a single rule applicable to every pair of functions, and
following this rule then evaluate which of them is greater. What kind of rule
it could be? Let's declare it as "compare" method, that returns one of 3
possible values:

-1, left is *less* than right,

0, left and right are *equal*,

1, left is *greater* than right.

Of course it means, that we have to maintain
*strict and non-strict order relation properties*:

* reflexivity (``a <= a``, ``a == a``, ``a >= a``),
* antisymmetry (if ``a <= b`` and ``b <= a`` then ``a == b``),
* transitivity (``a <= b`` and ``b <= c``, then ``a <= c``)
* asymmetry (if ``a < b``, then ``a > b`` or ``a == b``).

As it was mentioned before, comparison routine consists of
"sub-comparison-routines", each of them also consists
"sub-comparison-routines", and so on, finally it ends up with a primitives
comparison.

Below, we will use the next operations:

#. ``cmpNumbers(number1, number2)`` is method that returns -1 if left is less
   than right; 0, if left and right are equal; and 1 otherwise.

#. ``cmpFlags(flag1, flag2)`` is hypothetical method that compares two flags.
   The logic is the same as in ``cmpNumbers``, where ``true`` is 1, and
   ``false`` is 0.

The rest of article is based on *MergeFunctions.cpp* source code
(*<llvm_dir>/lib/Transforms/IPO/MergeFunctions.cpp*). We would like to ask
reader to keep this file open nearby, so we could use it as a reference for
further explanations.

Now we're ready to proceed to the next chapter and see how it works.

Functions comparison
====================
At first, let's define how exactly we compare complex objects.

Complex objects comparison (function, basic-block, etc) is mostly based on its
sub-objects comparison results. So it is similar to the next "tree" objects
comparison:

#. For two trees *T1* and *T2* we perform *depth-first-traversal* and have
   two sequences as a product: "*T1Items*" and "*T2Items*".

#. Then compare chains "*T1Items*" and "*T2Items*" in
   most-significant-item-first order. Result of items comparison would be the
   result of *T1* and *T2* comparison itself.

FunctionComparator::compare(void)
---------------------------------
Brief look at the source code tells us, that comparison starts in
“``int FunctionComparator::compare(void)``” method.

1. First parts to be compared are function's attributes and some properties that
outsides “attributes” term, but still could make function different without
changing its body. This part of comparison is usually done within simple
*cmpNumbers* or *cmpFlags* operations (e.g.
``cmpFlags(F1->hasGC(), F2->hasGC())``). Below is full list of function's
properties to be compared on this stage:

  * *Attributes* (those are returned by ``Function::getAttributes()``
    method).

  * *GC*, for equivalence, *RHS* and *LHS* should be both either without
    *GC* or with the same one.

  * *Section*, just like a *GC*: *RHS* and *LHS* should be defined in the
    same section.

  * *Variable arguments*. *LHS* and *RHS* should be both either with or
    without *var-args*.

  * *Calling convention* should be the same.

2. Function type. Checked by ``FunctionComparator::cmpType(Type*, Type*)``
method. It checks return type and parameters type; the method itself will be
described later.

3. Associate function formal parameters with each other. Then comparing function
bodies, if we see the usage of *LHS*'s *i*-th argument in *LHS*'s body, then,
we want to see usage of *RHS*'s *i*-th argument at the same place in *RHS*'s
body, otherwise functions are different. On this stage we grant the preference
to those we met later in function body (value we met first would be *less*).
This is done by “``FunctionComparator::cmpValues(const Value*, const Value*)``”
method (will be described a bit later).

4. Function body comparison. As it written in method comments:

“We do a CFG-ordered walk since the actual ordering of the blocks in the linked
list is immaterial. Our walk starts at the entry block for both functions, then
takes each block from each terminator in order. As an artifact, this also means
that unreachable blocks are ignored.”

So, using this walk we get BBs from *left* and *right* in the same order, and
compare them by “``FunctionComparator::compare(const BasicBlock*, const
BasicBlock*)``” method.

We also associate BBs with each other, like we did it with function formal
arguments (see ``cmpValues`` method below).

FunctionComparator::cmpType
---------------------------
Consider how types comparison works.

1. Coerce pointer to integer. If left type is a pointer, try to coerce it to the
integer type. It could be done if its address space is 0, or if address spaces
are ignored at all. Do the same thing for the right type.

2. If left and right types are equal, return 0. Otherwise we need to give
preference to one of them. So proceed to the next step.

3. If types are of different kind (different type IDs). Return result of type
IDs comparison, treating them as a numbers (use ``cmpNumbers`` operation).

4. If types are vectors or integers, return result of their pointers comparison,
comparing them as numbers.

5. Check whether type ID belongs to the next group (call it equivalent-group):

   * Void

   * Float

   * Double

   * X86_FP80

   * FP128

   * PPC_FP128

   * Label

   * Metadata.

   If ID belongs to group above, return 0. Since it's enough to see that
   types has the same ``TypeID``. No additional information is required.

6. Left and right are pointers. Return result of address space comparison
(numbers comparison).

7. Complex types (structures, arrays, etc.). Follow complex objects comparison
technique (see the very first paragraph of this chapter). Both *left* and
*right* are to be expanded and their element types will be checked the same
way. If we get -1 or 1 on some stage, return it. Otherwise return 0.

8. Steps 1-6 describe all the possible cases, if we passed steps 1-6 and didn't
get any conclusions, then invoke ``llvm_unreachable``, since it's quite
unexpectable case.

cmpValues(const Value*, const Value*)
-------------------------------------
Method that compares local values.

This method gives us an answer on a very curious quesion: whether we could treat
local values as equal, and which value is greater otherwise. It's better to
start from example:

Consider situation when we're looking at the same place in left function "*FL*"
and in right function "*FR*". And every part of *left* place is equal to the
corresponding part of *right* place, and (!) both parts use *Value* instances,
for example:

.. code-block:: llvm

   instr0 i32 %LV   ; left side, function FL
   instr0 i32 %RV   ; right side, function FR

So, now our conclusion depends on *Value* instances comparison.

Main purpose of this method is to determine relation between such values.

What we expect from equal functions? At the same place, in functions "*FL*" and
"*FR*" we expect to see *equal* values, or values *defined* at the same place
in "*FL*" and "*FR*".

Consider small example here:

.. code-block:: llvm

  define void %f(i32 %pf0, i32 %pf1) {
    instr0 i32 %pf0 instr1 i32 %pf1 instr2 i32 123
  }

.. code-block:: llvm

  define void %g(i32 %pg0, i32 %pg1) {
    instr0 i32 %pg0 instr1 i32 %pg0 instr2 i32 123
  }

In this example, *pf0* is associated with *pg0*, *pf1* is associated with *pg1*,
and we also declare that *pf0* < *pf1*, and thus *pg0* < *pf1*.

Instructions with opcode "*instr0*" would be *equal*, since their types and
opcodes are equal, and values are *associated*.

Instruction with opcode "*instr1*" from *f* is *greater* than instruction with
opcode "*instr1*" from *g*; here we have equal types and opcodes, but "*pf1* is
greater than "*pg0*".

And instructions with opcode "*instr2*" are equal, because their opcodes and
types are equal, and the same constant is used as a value.

What we assiciate in cmpValues?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Function arguments. *i*-th argument from left function associated with
  *i*-th argument from right function.
* BasicBlock instances. In basic-block enumeration loop we associate *i*-th
  BasicBlock from the left function with *i*-th BasicBlock from the right
  function.
* Instructions.
* Instruction operands. Note, we can meet *Value* here we have never seen
  before. In this case it is not a function argument, nor *BasicBlock*, nor
  *Instruction*. It is global value. It is constant, since its the only
  supposed global here. Method also compares:
* Constants that are of the same type.
* If right constant could be losslessly bit-casted to the left one, then we
  also compare them.

How to implement cmpValues?
^^^^^^^^^^^^^^^^^^^^^^^^^^^
*Association* is a case of equality for us. We just treat such values as equal.
But, in general, we need to implement antisymmetric relation. As it was
mentioned above, to understand what is *less*, we can use order in which we
meet values. If both of values has the same order in function (met at the same
time), then treat values as *associated*. Otherwise – it depends on who was
first.

Every time we run top-level compare method, we initialize two identical maps
(one for the left side, another one for the right side):

``map<Value, int> sn_mapL, sn_mapR;``

The key of the map is the *Value* itself, the *value* – is its order (call it
*serial number*).

To add value *V* we need to perform the next procedure:

``sn_map.insert(std::make_pair(V, sn_map.size()));``

For the first *Value*, map will return *0*, for second *Value* map will return
*1*, and so on.

Then we can check whether left and right values met at the same time with simple
comparison:

``cmpNumbers(sn_mapL[Left], sn_mapR[Right]);``

Of course, we can combine insertion and comparison:

.. code-block:: c++

  std::pair<iterator, bool>
    LeftRes = sn_mapL.insert(std::make_pair(Left, sn_mapL.size())), RightRes
    = sn_mapR.insert(std::make_pair(Right, sn_mapR.size()));
  return cmpNumbers(LeftRes.first->second, RightRes.first->second);

Let's look, how whole method could be implemented.

1. we have to start from the bad news. Consider function self and
cross-referencing cases:

.. code-block:: c++

  // self-reference unsigned fact0(unsigned n) { return n > 1 ? n
  * fact0(n-1) : 1; } unsigned fact1(unsigned n) { return n > 1 ? n *
  fact1(n-1) : 1; }

  // cross-reference unsigned ping(unsigned n) { return n!= 0 ? pong(n-1) : 0;
  } unsigned pong(unsigned n) { return n!= 0 ? ping(n-1) : 0; }

..

  This comparison has been implemented in initial *MergeFunctions* pass
  version. But, unfortunately, it is not transitive. And this is the only case
  we can't convert to less-equal-greater comparison. It is a seldom case, 4-5
  functions of 10000 (checked on test-suite), and, we hope, reader would
  forgive us for such a sacrifice in order to get the O(log(N)) pass time.

2. If left/right *Value* is a constant, we have to compare them. Return 0 if it
is the same constant, or use ``cmpConstants`` method otherwise.

3. If left/right is *InlineAsm* instance. Return result of *Value* pointers
comparison.

4. Explicit association of *L* (left value) and *R*  (right value). We need to
find out whether values met at the same time, and thus are *associated*. Or we
need to put the rule: when we treat *L* < *R*. Now it is easy: just return
result of numbers comparison:

.. code-block:: c++

   std::pair<iterator, bool>
     LeftRes = sn_mapL.insert(std::make_pair(Left, sn_mapL.size())),
     RightRes = sn_mapR.insert(std::make_pair(Right, sn_mapR.size()));
   if (LeftRes.first->second == RightRes.first->second) return 0;
   if (LeftRes.first->second < RightRes.first->second) return -1;
   return 1;

Now when *cmpValues* returns 0, we can proceed comparison procedure. Otherwise,
if we get (-1 or 1), we need to pass this result to the top level, and finish
comparison procedure.

cmpConstants
------------
Performs constants comparison as follows:

1. Compare constant types using ``cmpType`` method. If result is -1 or 1, goto
step 2, otherwise proceed to step 3.

2. If types are different, we still can check whether constants could be
losslessly bitcasted to each other. The further explanation is modification of
``canLosslesslyBitCastTo`` method.

   2.1 Check whether constants are of the first class types
   (``isFirstClassType`` check):

   2.1.1. If both constants are *not* of the first class type: return result
   of ``cmpType``.

   2.1.2. Otherwise, if left type is not of the first class, return -1. If
   right type is not of the first class, return 1.

   2.1.3. If both types are of the first class type, proceed to the next step
   (2.1.3.1).

   2.1.3.1. If types are vectors, compare their bitwidth using the
   *cmpNumbers*. If result is not 0, return it.

   2.1.3.2. Different types, but not a vectors:

   * if both of them are pointers, good for us, we can proceed to step 3.
   * if one of types is pointer, return result of *isPointer* flags
     comparison (*cmpFlags* operation).
   * otherwise we have no methods to prove bitcastability, and thus return
     result of types comparison (-1 or 1).

Steps below are for the case when types are equal, or case when constants are
bitcastable:

3. One of constants is a "*null*" value. Return the result of
``cmpFlags(L->isNullValue, R->isNullValue)`` comparison.

4. Compare value IDs, and return result if it is not 0:

.. code-block:: c++

  if (int Res = cmpNumbers(L->getValueID(), R->getValueID()))
    return Res;

5. Compare the contents of constants. The comparison depends on kind of
constants, but on this stage it is just a lexicographical comparison. Just see
how it was described in the beginning of "*Functions comparison*" paragraph.
Mathematically it is equal to the next case: we encode left constant and right
constant (with similar way *bitcode-writer* does). Then compare left code
sequence and right code sequence.

compare(const BasicBlock*, const BasicBlock*)
---------------------------------------------
Compares two *BasicBlock* instances.

It enumerates instructions from left *BB* and right *BB*.

1. It assigns serial numbers to the left and right instructions, using
``cmpValues`` method.

2. If one of left or right is *GEP* (``GetElementPtr``), then treat *GEP* as
greater than other instructions, if both instructions are *GEPs* use ``cmpGEP``
method for comparison. If result is -1 or 1, pass it to the top-level
comparison (return it).

   3.1. Compare operations. Call ``cmpOperation`` method. If result is -1 or
   1, return it.

   3.2. Compare number of operands, if result is -1 or 1, return it.

   3.3. Compare operands themselves, use ``cmpValues`` method. Return result
   if it is -1 or 1.

   3.4. Compare type of operands, using ``cmpType`` method. Return result if
   it is -1 or 1.

   3.5. Proceed to the next instruction.

4. We can finish instruction enumeration in 3 cases:

   4.1. We reached the end of both left and right basic-blocks. We didn't
   exit on steps 1-3, so contents is equal, return 0.

   4.2. We have reached the end of the left basic-block. Return -1.

   4.3. Return 1 (the end of the right basic block).

cmpGEP
------
Compares two GEPs (``getelementptr`` instructions).

It differs from regular operations comparison with the only thing: possibility
to use ``accumulateConstantOffset`` method.

So, if we get constant offset for both left and right *GEPs*, then compare it as
numbers, and return comparison result.

Otherwise treat it like a regular operation (see previous paragraph).

cmpOperation
------------
Compares instruction opcodes and some important operation properties.

1. Compare opcodes, if it differs return the result.

2. Compare number of operands. If it differs – return the result.

3. Compare operation types, use *cmpType*. All the same – if types are
different, return result.

4. Compare *subclassOptionalData*, get it with ``getRawSubclassOptionalData``
method, and compare it like a numbers.

5. Compare operand types.

6. For some particular instructions check equivalence (relation in our case) of
some significant attributes. For example we have to compare alignment for
``load`` instructions.

O(log(N))
---------
Methods described above implement order relationship. And latter, could be used
for nodes comparison in a binary tree. So we can organize functions set into
the binary tree and reduce the cost of lookup procedure from
O(N*N) to O(log(N)).

Merging process, mergeTwoFunctions
==================================
Once *MergeFunctions* detected that current function (*G*) is equal to one that
were analyzed before (function *F*) it calls ``mergeTwoFunctions(Function*,
Function*)``.

Operation affects ``FnTree`` contents with next way: *F* will stay in
``FnTree``. *G* being equal to *F* will not be added to ``FnTree``. Calls of
*G* would be replaced with something else. It changes bodies of callers. So,
functions that calls *G* would be put into ``Deferred`` set and removed from
``FnTree``, and analyzed again.

The approach is next:

1. Most wished case: when we can use alias and both of *F* and *G* are weak. We
make both of them with aliases to the third strong function *H*. Actually *H*
is *F*. See below how it's made (but it's better to look straight into the
source code). Well, this is a case when we can just replace *G* with *F*
everywhere, we use ``replaceAllUsesWith`` operation here (*RAUW*).

2. *F* could not be overridden, while *G* could. It would be good to do the
next: after merging the places where overridable function were used, still use
overridable stub. So try to make *G* alias to *F*, or create overridable tail
call wrapper around *F* and replace *G* with that call.

3. Neither *F* nor *G* could be overridden. We can't use *RAUW*. We can just
change the callers: call *F* instead of *G*.  That's what
``replaceDirectCallers`` does.

Below is detailed body description.

If “F” may be overridden
------------------------
As follows from ``mayBeOverridden`` comments: “whether the definition of this
global may be replaced by something non-equivalent at link time”. If so, that's
ok: we can use alias to *F* instead of *G* or change call instructions itself.

HasGlobalAliases, removeUsers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First consider the case when we have global aliases of one function name to
another. Our purpose is  make both of them with aliases to the third strong
function. Though if we keep *F* alive and without major changes we can leave it
in ``FnTree``. Try to combine these two goals.

Do stub replacement of *F* itself with an alias to *F*.

1. Create stub function *H*, with the same name and attributes like function
*F*. It takes maximum alignment of *F* and *G*.

2. Replace all uses of function *F* with uses of function *H*. It is the two
steps procedure instead. First of all, we must take into account, all functions
from whom *F* is called would be changed: since we change the call argument
(from *F* to *H*). If so we must to review these caller functions again after
this procedure. We remove callers from ``FnTree``, method with name
``removeUsers(F)`` does that (don't confuse with ``replaceAllUsesWith``):

   2.1. ``Inside removeUsers(Value*
   V)`` we go through the all values that use value *V* (or *F* in our context).
   If value is instruction, we go to function that holds this instruction and
   mark it as to-be-analyzed-again (put to ``Deferred`` set), we also remove
   caller from ``FnTree``.

   2.2. Now we can do the replacement: call ``F->replaceAllUsesWith(H)``.

3. *H* (that now "officially" plays *F*'s role) is replaced with alias to *F*.
Do the same with *G*: replace it with alias to *F*. So finally everywhere *F*
was used, we use *H* and it is alias to *F*, and everywhere *G* was used we
also have alias to *F*.

4. Set *F* linkage to private. Make it strong :-)

No global aliases, replaceDirectCallers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If global aliases are not supported. We call ``replaceDirectCallers`` then. Just
go through all calls of *G* and replace it with calls of *F*. If you look into
method you will see that it scans all uses of *G* too, and if use is callee (if
user is call instruction and *G* is used as what to be called), we replace it
with use of *F*.

If “F” could not be overridden, fix it!
"""""""""""""""""""""""""""""""""""""""

We call ``writeThunkOrAlias(Function *F, Function *G)``. Here we try to replace
*G* with alias to *F* first. Next conditions are essential:

* target should support global aliases,
* the address itself of  *G* should be not significant, not named and not
  referenced anywhere,
* function should come with external, local or weak linkage.

Otherwise we write thunk: some wrapper that has *G's* interface and calls *F*,
so *G* could be replaced with this wrapper.

*writeAlias*

As follows from *llvm* reference:

“Aliases act as *second name* for the aliasee value”. So we just want to create
second name for *F* and use it instead of *G*:

1. create global alias itself (*GA*),

2. adjust alignment of *F* so it must be maximum of current and *G's* alignment;

3. replace uses of *G*:

   3.1. first mark all callers of *G* as to-be-analyzed-again, using
   ``removeUsers`` method (see chapter above),

   3.2. call ``G->replaceAllUsesWith(GA)``.

4. Get rid of *G*.

*writeThunk*

As it written in method comments:

“Replace G with a simple tail call to bitcast(F). Also replace direct uses of G
with bitcast(F). Deletes G.”

In general it does the same as usual when we want to replace callee, except the
first point:

1. We generate tail call wrapper around *F*, but with interface that allows use
it instead of *G*.

2. “As-usual”: ``removeUsers`` and ``replaceAllUsesWith`` then.

3. Get rid of *G*.

That's it.
==========
We have described how to detect equal functions, and how to merge them, and in
first chapter we have described how it works all-together. Author hopes, reader
have some picture from now, and it helps him improve and debug ­this pass.

Reader is welcomed to send us any questions and proposals ;-)
