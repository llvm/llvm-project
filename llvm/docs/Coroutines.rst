=====================================
Coroutines in LLVM
=====================================

.. contents::
   :local:
   :depth: 3

.. warning::
  This is a work in progress. Compatibility across LLVM releases is not 
  guaranteed.

Introduction
============

.. _coroutine handle:

LLVM coroutines are functions that have one or more `suspend points`_. 
When a suspend point is reached, the execution of a coroutine is suspended and
control is returned back to its caller. A suspended coroutine can be resumed 
to continue execution from the last suspend point or it can be destroyed. 

In the following example, we call function `f` (which may or may not be a 
coroutine itself) that returns a handle to a suspended coroutine 
(**coroutine handle**) that is used by `main` to resume the coroutine twice and
then destroy it:

.. code-block:: llvm

  define i32 @main() {
  entry:
    %hdl = call i8* @f(i32 4)
    call void @llvm.coro.resume(i8* %hdl)
    call void @llvm.coro.resume(i8* %hdl)
    call void @llvm.coro.destroy(i8* %hdl)
    ret i32 0
  }

.. _coroutine frame:

In addition to the function stack frame which exists when a coroutine is 
executing, there is an additional region of storage that contains objects that 
keep the coroutine state when a coroutine is suspended. This region of storage
is called **coroutine frame**. It is created when a coroutine is called and 
destroyed when a coroutine runs to completion or destroyed by a call to 
the `coro.destroy`_ intrinsic. 

An LLVM coroutine is represented as an LLVM function that has calls to
`coroutine intrinsics`_ defining the structure of the coroutine.
After lowering, a coroutine is split into several
functions that represent three different ways of how control can enter the 
coroutine: 

1. a ramp function, which represents an initial invocation of the coroutine that
   creates the coroutine frame and executes the coroutine code until it 
   encounters a suspend point or reaches the end of the function;

2. a coroutine resume function that is invoked when the coroutine is resumed;

3. a coroutine destroy function that is invoked when the coroutine is destroyed.

.. note:: Splitting out resume and destroy functions are just one of the 
   possible ways of lowering the coroutine. We chose it for initial 
   implementation as it matches closely the mental model and results in 
   reasonably nice code.

Coroutines by Example
=====================

Coroutine Representation
------------------------

Let's look at an example of an LLVM coroutine with the behavior sketched
by the following pseudo-code.

.. code-block:: c++

  void *f(int n) {
     for(;;) {
       print(n++);
       <suspend> // returns a coroutine handle on first suspend
     }     
  } 

This coroutine calls some function `print` with value `n` as an argument and
suspends execution. Every time this coroutine resumes, it calls `print` again with an argument one bigger than the last time. This coroutine never completes by itself and must be destroyed explicitly. If we use this coroutine with 
a `main` shown in the previous section. It will call `print` with values 4, 5 
and 6 after which the coroutine will be destroyed.

The LLVM IR for this coroutine looks like this:

.. code-block:: llvm

  define i8* @f(i32 %n) {
  entry:
    %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
    %size = call i32 @llvm.coro.size.i32()
    %alloc = call i8* @malloc(i32 %size)
    %hdl = call noalias i8* @llvm.coro.begin(token %id, i8* %alloc)
    br label %loop
  loop:
    %n.val = phi i32 [ %n, %entry ], [ %inc, %loop ]
    %inc = add nsw i32 %n.val, 1
    call void @print(i32 %n.val)
    %0 = call i8 @llvm.coro.suspend(token none, i1 false)
    switch i8 %0, label %suspend [i8 0, label %loop
                                  i8 1, label %cleanup]
  cleanup:
    %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
    call void @free(i8* %mem)
    br label %suspend
  suspend:
    %unused = call i1 @llvm.coro.end(i8* %hdl, i1 false)
    ret i8* %hdl
  }

The `entry` block establishes the coroutine frame. The `coro.size`_ intrinsic is
lowered to a constant representing the size required for the coroutine frame. 
The `coro.begin`_ intrinsic initializes the coroutine frame and returns the 
coroutine handle. The second parameter of `coro.begin` is given a block of memory 
to be used if the coroutine frame needs to be allocated dynamically.
The `coro.id`_ intrinsic serves as coroutine identity useful in cases when the
`coro.begin`_ intrinsic get duplicated by optimization passes such as 
jump-threading.

The `cleanup` block destroys the coroutine frame. The `coro.free`_ intrinsic, 
given the coroutine handle, returns a pointer of the memory block to be freed or
`null` if the coroutine frame was not allocated dynamically. The `cleanup` 
block is entered when coroutine runs to completion by itself or destroyed via
call to the `coro.destroy`_ intrinsic.

The `suspend` block contains code to be executed when coroutine runs to 
completion or suspended. The `coro.end`_ intrinsic marks the point where 
a coroutine needs to return control back to the caller if it is not an initial 
invocation of the coroutine. 

The `loop` blocks represents the body of the coroutine. The `coro.suspend`_ 
intrinsic in combination with the following switch indicates what happens to 
control flow when a coroutine is suspended (default case), resumed (case 0) or 
destroyed (case 1).

Coroutine Transformation
------------------------

One of the steps of coroutine lowering is building the coroutine frame. The
def-use chains are analyzed to determine which objects need be kept alive across
suspend points. In the coroutine shown in the previous section, use of virtual register 
`%n.val` is separated from the definition by a suspend point, therefore, it 
cannot reside on the stack frame since the latter goes away once the coroutine 
is suspended and control is returned back to the caller. An i32 slot is 
allocated in the coroutine frame and `%n.val` is spilled and reloaded from that
slot as needed.

We also store addresses of the resume and destroy functions so that the 
`coro.resume` and `coro.destroy` intrinsics can resume and destroy the coroutine
when its identity cannot be determined statically at compile time. For our 
example, the coroutine frame will be:

.. code-block:: llvm

  %f.frame = type { void (%f.frame*)*, void (%f.frame*)*, i32 }

After resume and destroy parts are outlined, function `f` will contain only the 
code responsible for creation and initialization of the coroutine frame and 
execution of the coroutine until a suspend point is reached:

.. code-block:: llvm

  define i8* @f(i32 %n) {
  entry:
    %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
    %alloc = call noalias i8* @malloc(i32 24)
    %0 = call noalias i8* @llvm.coro.begin(token %id, i8* %alloc)
    %frame = bitcast i8* %0 to %f.frame*
    %1 = getelementptr %f.frame, %f.frame* %frame, i32 0, i32 0
    store void (%f.frame*)* @f.resume, void (%f.frame*)** %1
    %2 = getelementptr %f.frame, %f.frame* %frame, i32 0, i32 1
    store void (%f.frame*)* @f.destroy, void (%f.frame*)** %2
   
    %inc = add nsw i32 %n, 1
    %inc.spill.addr = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i32 0, i32 2
    store i32 %inc, i32* %inc.spill.addr
    call void @print(i32 %n)
   
    ret i8* %frame
  }

Outlined resume part of the coroutine will reside in function `f.resume`:

.. code-block:: llvm

  define internal fastcc void @f.resume(%f.frame* %frame.ptr.resume) {
  entry:
    %inc.spill.addr = getelementptr %f.frame, %f.frame* %frame.ptr.resume, i64 0, i32 2
    %inc.spill = load i32, i32* %inc.spill.addr, align 4
    %inc = add i32 %n.val, 1
    store i32 %inc, i32* %inc.spill.addr, align 4
    tail call void @print(i32 %inc)
    ret void
  }

Whereas function `f.destroy` will contain the cleanup code for the coroutine:

.. code-block:: llvm

  define internal fastcc void @f.destroy(%f.frame* %frame.ptr.destroy) {
  entry:
    %0 = bitcast %f.frame* %frame.ptr.destroy to i8*
    tail call void @free(i8* %0)
    ret void
  }

Avoiding Heap Allocations
-------------------------
 
A particular coroutine usage pattern, which is illustrated by the `main` 
function in the overview section, where a coroutine is created, manipulated and 
destroyed by the same calling function, is common for coroutines implementing
RAII idiom and is suitable for allocation elision optimization which avoid 
dynamic allocation by storing the coroutine frame as a static `alloca` in its 
caller.

In the entry block, we will call `coro.alloc`_ intrinsic that will return `true`
when dynamic allocation is required, and `false` if dynamic allocation is 
elided.

.. code-block:: llvm

  entry:
    %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
    %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
    br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
  dyn.alloc:
    %size = call i32 @llvm.coro.size.i32()
    %alloc = call i8* @CustomAlloc(i32 %size)
    br label %coro.begin
  coro.begin:
    %phi = phi i8* [ null, %entry ], [ %alloc, %dyn.alloc ]
    %hdl = call noalias i8* @llvm.coro.begin(token %id, i8* %phi)

In the cleanup block, we will make freeing the coroutine frame conditional on
`coro.free`_ intrinsic. If allocation is elided, `coro.free`_ returns `null`
thus skipping the deallocation code:

.. code-block:: llvm

  cleanup:
    %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
    %need.dyn.free = icmp ne i8* %mem, null
    br i1 %need.dyn.free, label %dyn.free, label %if.end
  dyn.free:
    call void @CustomFree(i8* %mem)
    br label %if.end
  if.end:
    ...

With allocations and deallocations represented as described as above, after
coroutine heap allocation elision optimization, the resulting main will be:

.. code-block:: llvm

  define i32 @main() {
  entry:
    call void @print(i32 4)
    call void @print(i32 5)
    call void @print(i32 6)
    ret i32 0
  }

Multiple Suspend Points
-----------------------

Let's consider the coroutine that has more than one suspend point:

.. code-block:: c++

  void *f(int n) {
     for(;;) {
       print(n++);
       <suspend>
       print(-n);
       <suspend>
     }
  }

Matching LLVM code would look like (with the rest of the code remaining the same
as the code in the previous section):

.. code-block:: llvm

  loop:
    %n.addr = phi i32 [ %n, %entry ], [ %inc, %loop.resume ]
    call void @print(i32 %n.addr) #4
    %2 = call i8 @llvm.coro.suspend(token none, i1 false)
    switch i8 %2, label %suspend [i8 0, label %loop.resume
                                  i8 1, label %cleanup]
  loop.resume:
    %inc = add nsw i32 %n.addr, 1
    %sub = xor i32 %n.addr, -1
    call void @print(i32 %sub)
    %3 = call i8 @llvm.coro.suspend(token none, i1 false)
    switch i8 %3, label %suspend [i8 0, label %loop
                                  i8 1, label %cleanup]

In this case, the coroutine frame would include a suspend index that will 
indicate at which suspend point the coroutine needs to resume. The resume 
function will use an index to jump to an appropriate basic block and will look 
as follows:

.. code-block:: llvm

  define internal fastcc void @f.Resume(%f.Frame* %FramePtr) {
  entry.Resume:
    %index.addr = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i64 0, i32 2
    %index = load i8, i8* %index.addr, align 1
    %switch = icmp eq i8 %index, 0
    %n.addr = getelementptr inbounds %f.Frame, %f.Frame* %FramePtr, i64 0, i32 3
    %n = load i32, i32* %n.addr, align 4
    br i1 %switch, label %loop.resume, label %loop

  loop.resume:
    %sub = xor i32 %n, -1
    call void @print(i32 %sub)
    br label %suspend
  loop:
    %inc = add nsw i32 %n, 1
    store i32 %inc, i32* %n.addr, align 4
    tail call void @print(i32 %inc)
    br label %suspend

  suspend:
    %storemerge = phi i8 [ 0, %loop ], [ 1, %loop.resume ]
    store i8 %storemerge, i8* %index.addr, align 1
    ret void
  }

If different cleanup code needs to get executed for different suspend points, 
a similar switch will be in the `f.destroy` function.

.. note ::

  Using suspend index in a coroutine state and having a switch in `f.resume` and
  `f.destroy` is one of the possible implementation strategies. We explored 
  another option where a distinct `f.resume1`, `f.resume2`, etc. are created for
  every suspend point, and instead of storing an index, the resume and destroy 
  function pointers are updated at every suspend. Early testing showed that the
  current approach is easier on the optimizer than the latter so it is a 
  lowering strategy implemented at the moment.

Distinct Save and Suspend
-------------------------

In the previous example, setting a resume index (or some other state change that 
needs to happen to prepare a coroutine for resumption) happens at the same time as
a suspension of a coroutine. However, in certain cases, it is necessary to control 
when coroutine is prepared for resumption and when it is suspended.

In the following example, a coroutine represents some activity that is driven
by completions of asynchronous operations `async_op1` and `async_op2` which get
a coroutine handle as a parameter and resume the coroutine once async
operation is finished.

.. code-block:: text

  void g() {
     for (;;)
       if (cond()) {
          async_op1(<coroutine-handle>); // will resume once async_op1 completes
          <suspend>
          do_one();
       }
       else {
          async_op2(<coroutine-handle>); // will resume once async_op2 completes
          <suspend>
          do_two();
       }
     }
  }

In this case, coroutine should be ready for resumption prior to a call to 
`async_op1` and `async_op2`. The `coro.save`_ intrinsic is used to indicate a
point when coroutine should be ready for resumption (namely, when a resume index
should be stored in the coroutine frame, so that it can be resumed at the 
correct resume point):

.. code-block:: llvm

  if.true:
    %save1 = call token @llvm.coro.save(i8* %hdl)
    call void @async_op1(i8* %hdl)
    %suspend1 = call i1 @llvm.coro.suspend(token %save1, i1 false)
    switch i8 %suspend1, label %suspend [i8 0, label %resume1
                                         i8 1, label %cleanup]
  if.false:
    %save2 = call token @llvm.coro.save(i8* %hdl)
    call void @async_op2(i8* %hdl)
    %suspend2 = call i1 @llvm.coro.suspend(token %save2, i1 false)
    switch i8 %suspend1, label %suspend [i8 0, label %resume2
                                         i8 1, label %cleanup]

.. _coroutine promise:

Coroutine Promise
-----------------

A coroutine author or a frontend may designate a distinguished `alloca` that can
be used to communicate with the coroutine. This distinguished alloca is called
**coroutine promise** and is provided as the second parameter to the 
`coro.id`_ intrinsic.

The following coroutine designates a 32 bit integer `promise` and uses it to
store the current value produced by a coroutine.

.. code-block:: llvm

  define i8* @f(i32 %n) {
  entry:
    %promise = alloca i32
    %pv = bitcast i32* %promise to i8*
    %id = call token @llvm.coro.id(i32 0, i8* %pv, i8* null, i8* null)
    %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
    br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
  dyn.alloc:
    %size = call i32 @llvm.coro.size.i32()
    %alloc = call i8* @malloc(i32 %size)
    br label %coro.begin
  coro.begin:
    %phi = phi i8* [ null, %entry ], [ %alloc, %dyn.alloc ]
    %hdl = call noalias i8* @llvm.coro.begin(token %id, i8* %phi)
    br label %loop
  loop:
    %n.val = phi i32 [ %n, %coro.begin ], [ %inc, %loop ]
    %inc = add nsw i32 %n.val, 1
    store i32 %n.val, i32* %promise
    %0 = call i8 @llvm.coro.suspend(token none, i1 false)
    switch i8 %0, label %suspend [i8 0, label %loop
                                  i8 1, label %cleanup]
  cleanup:
    %mem = call i8* @llvm.coro.free(token %id, i8* %hdl)
    call void @free(i8* %mem)
    br label %suspend
  suspend:
    %unused = call i1 @llvm.coro.end(i8* %hdl, i1 false)
    ret i8* %hdl
  }

A coroutine consumer can rely on the `coro.promise`_ intrinsic to access the
coroutine promise.

.. code-block:: llvm

  define i32 @main() {
  entry:
    %hdl = call i8* @f(i32 4)
    %promise.addr.raw = call i8* @llvm.coro.promise(i8* %hdl, i32 4, i1 false)
    %promise.addr = bitcast i8* %promise.addr.raw to i32*
    %val0 = load i32, i32* %promise.addr
    call void @print(i32 %val0)
    call void @llvm.coro.resume(i8* %hdl)
    %val1 = load i32, i32* %promise.addr
    call void @print(i32 %val1)
    call void @llvm.coro.resume(i8* %hdl)
    %val2 = load i32, i32* %promise.addr
    call void @print(i32 %val2)
    call void @llvm.coro.destroy(i8* %hdl)
    ret i32 0
  }

After example in this section is compiled, result of the compilation will be:

.. code-block:: llvm

  define i32 @main() {
  entry:
    tail call void @print(i32 4)
    tail call void @print(i32 5)
    tail call void @print(i32 6)
    ret i32 0
  }

.. _final:
.. _final suspend:

Final Suspend
-------------

A coroutine author or a frontend may designate a particular suspend to be final,
by setting the second argument of the `coro.suspend`_ intrinsic to `true`.
Such a suspend point has two properties:

* it is possible to check whether a suspended coroutine is at the final suspend
  point via `coro.done`_ intrinsic;

* a resumption of a coroutine stopped at the final suspend point leads to 
  undefined behavior. The only possible action for a coroutine at a final
  suspend point is destroying it via `coro.destroy`_ intrinsic.

From the user perspective, the final suspend point represents an idea of a 
coroutine reaching the end. From the compiler perspective, it is an optimization
opportunity for reducing number of resume points (and therefore switch cases) in
the resume function.

The following is an example of a function that keeps resuming the coroutine
until the final suspend point is reached after which point the coroutine is 
destroyed:

.. code-block:: llvm

  define i32 @main() {
  entry:
    %hdl = call i8* @f(i32 4)
    br label %while
  while:
    call void @llvm.coro.resume(i8* %hdl)
    %done = call i1 @llvm.coro.done(i8* %hdl)
    br i1 %done, label %end, label %while
  end:
    call void @llvm.coro.destroy(i8* %hdl)
    ret i32 0
  }

Usually, final suspend point is a frontend injected suspend point that does not
correspond to any explicitly authored suspend point of the high level language.
For example, for a Python generator that has only one suspend point:

.. code-block:: python

  def coroutine(n):
    for i in range(n):
      yield i

Python frontend would inject two more suspend points, so that the actual code
looks like this:

.. code-block:: c

  void* coroutine(int n) {
    int current_value; 
    <designate current_value to be coroutine promise>
    <SUSPEND> // injected suspend point, so that the coroutine starts suspended
    for (int i = 0; i < n; ++i) {
      current_value = i; <SUSPEND>; // corresponds to "yield i"
    }
    <SUSPEND final=true> // injected final suspend point
  }

and python iterator `__next__` would look like:

.. code-block:: c++

  int __next__(void* hdl) {
    coro.resume(hdl);
    if (coro.done(hdl)) throw StopIteration();
    return *(int*)coro.promise(hdl, 4, false);
  }

Intrinsics
==========

Coroutine Manipulation Intrinsics
---------------------------------

Intrinsics described in this section are used to manipulate an existing
coroutine. They can be used in any function which happen to have a pointer
to a `coroutine frame`_ or a pointer to a `coroutine promise`_.

.. _coro.destroy:

'llvm.coro.destroy' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Syntax:
"""""""

::

      declare void @llvm.coro.destroy(i8* <handle>)

Overview:
"""""""""

The '``llvm.coro.destroy``' intrinsic destroys a suspended
coroutine.

Arguments:
""""""""""

The argument is a coroutine handle to a suspended coroutine.

Semantics:
""""""""""

When possible, the `coro.destroy` intrinsic is replaced with a direct call to 
the coroutine destroy function. Otherwise it is replaced with an indirect call 
based on the function pointer for the destroy function stored in the coroutine
frame. Destroying a coroutine that is not suspended leads to undefined behavior.

.. _coro.resume:

'llvm.coro.resume' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

      declare void @llvm.coro.resume(i8* <handle>)

Overview:
"""""""""

The '``llvm.coro.resume``' intrinsic resumes a suspended coroutine.

Arguments:
""""""""""

The argument is a handle to a suspended coroutine.

Semantics:
""""""""""

When possible, the `coro.resume` intrinsic is replaced with a direct call to the
coroutine resume function. Otherwise it is replaced with an indirect call based 
on the function pointer for the resume function stored in the coroutine frame. 
Resuming a coroutine that is not suspended leads to undefined behavior.

.. _coro.done:

'llvm.coro.done' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

      declare i1 @llvm.coro.done(i8* <handle>)

Overview:
"""""""""

The '``llvm.coro.done``' intrinsic checks whether a suspended coroutine is at 
the final suspend point or not.

Arguments:
""""""""""

The argument is a handle to a suspended coroutine.

Semantics:
""""""""""

Using this intrinsic on a coroutine that does not have a `final suspend`_ point 
or on a coroutine that is not suspended leads to undefined behavior.

.. _coro.promise:

'llvm.coro.promise' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

      declare i8* @llvm.coro.promise(i8* <ptr>, i32 <alignment>, i1 <from>)

Overview:
"""""""""

The '``llvm.coro.promise``' intrinsic obtains a pointer to a 
`coroutine promise`_ given a coroutine handle and vice versa.

Arguments:
""""""""""

The first argument is a handle to a coroutine if `from` is false. Otherwise, 
it is a pointer to a coroutine promise.

The second argument is an alignment requirements of the promise. 
If a frontend designated `%promise = alloca i32` as a promise, the alignment 
argument to `coro.promise` should be the alignment of `i32` on the target 
platform. If a frontend designated `%promise = alloca i32, align 16` as a 
promise, the alignment argument should be 16.
This argument only accepts constants.

The third argument is a boolean indicating a direction of the transformation.
If `from` is true, the intrinsic returns a coroutine handle given a pointer 
to a promise. If `from` is false, the intrinsics return a pointer to a promise 
from a coroutine handle. This argument only accepts constants.

Semantics:
""""""""""

Using this intrinsic on a coroutine that does not have a coroutine promise
leads to undefined behavior. It is possible to read and modify coroutine
promise of the coroutine which is currently executing. The coroutine author and
a coroutine user are responsible to makes sure there is no data races.

Example:
""""""""

.. code-block:: llvm

  define i8* @f(i32 %n) {
  entry:
    %promise = alloca i32
    %pv = bitcast i32* %promise to i8*
    ; the second argument to coro.id points to the coroutine promise.
    %id = call token @llvm.coro.id(i32 0, i8* %pv, i8* null, i8* null)
    ...
    %hdl = call noalias i8* @llvm.coro.begin(token %id, i8* %alloc)
    ...
    store i32 42, i32* %promise ; store something into the promise
    ...
    ret i8* %hdl
  }

  define i32 @main() {
  entry:
    %hdl = call i8* @f(i32 4) ; starts the coroutine and returns its handle
    %promise.addr.raw = call i8* @llvm.coro.promise(i8* %hdl, i32 4, i1 false)
    %promise.addr = bitcast i8* %promise.addr.raw to i32*    
    %val = load i32, i32* %promise.addr ; load a value from the promise
    call void @print(i32 %val)
    call void @llvm.coro.destroy(i8* %hdl)
    ret i32 0
  }

.. _coroutine intrinsics:

Coroutine Structure Intrinsics
------------------------------
Intrinsics described in this section are used within a coroutine to describe
the coroutine structure. They should not be used outside of a coroutine.

.. _coro.size:

'llvm.coro.size' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

    declare i32 @llvm.coro.size.i32()
    declare i64 @llvm.coro.size.i64()

Overview:
"""""""""

The '``llvm.coro.size``' intrinsic returns the number of bytes
required to store a `coroutine frame`_.

Arguments:
""""""""""

None

Semantics:
""""""""""

The `coro.size` intrinsic is lowered to a constant representing the size of
the coroutine frame. 

.. _coro.begin:

'llvm.coro.begin' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare i8* @llvm.coro.begin(token <id>, i8* <mem>)

Overview:
"""""""""

The '``llvm.coro.begin``' intrinsic returns an address of the coroutine frame.

Arguments:
""""""""""

The first argument is a token returned by a call to '``llvm.coro.id``' 
identifying the coroutine.

The second argument is a pointer to a block of memory where coroutine frame
will be stored if it is allocated dynamically.

Semantics:
""""""""""

Depending on the alignment requirements of the objects in the coroutine frame
and/or on the codegen compactness reasons the pointer returned from `coro.begin` 
may be at offset to the `%mem` argument. (This could be beneficial if 
instructions that express relative access to data can be more compactly encoded 
with small positive and negative offsets).

A frontend should emit exactly one `coro.begin` intrinsic per coroutine.

.. _coro.free:

'llvm.coro.free' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare i8* @llvm.coro.free(token %id, i8* <frame>)

Overview:
"""""""""

The '``llvm.coro.free``' intrinsic returns a pointer to a block of memory where 
coroutine frame is stored or `null` if this instance of a coroutine did not use
dynamically allocated memory for its coroutine frame.

Arguments:
""""""""""

The first argument is a token returned by a call to '``llvm.coro.id``' 
identifying the coroutine.

The second argument is a pointer to the coroutine frame. This should be the same
pointer that was returned by prior `coro.begin` call.

Example (custom deallocation function):
"""""""""""""""""""""""""""""""""""""""

.. code-block:: llvm

  cleanup:
    %mem = call i8* @llvm.coro.free(token %id, i8* %frame)
    %mem_not_null = icmp ne i8* %mem, null
    br i1 %mem_not_null, label %if.then, label %if.end
  if.then:
    call void @CustomFree(i8* %mem)
    br label %if.end
  if.end:
    ret void

Example (standard deallocation functions):
""""""""""""""""""""""""""""""""""""""""""

.. code-block:: llvm

  cleanup:
    %mem = call i8* @llvm.coro.free(token %id, i8* %frame)
    call void @free(i8* %mem)
    ret void

.. _coro.alloc:

'llvm.coro.alloc' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare i1 @llvm.coro.alloc(token <id>)

Overview:
"""""""""

The '``llvm.coro.alloc``' intrinsic returns `true` if dynamic allocation is
required to obtain a memory for the coroutine frame and `false` otherwise.

Arguments:
""""""""""

The first argument is a token returned by a call to '``llvm.coro.id``' 
identifying the coroutine.

Semantics:
""""""""""

A frontend should emit at most one `coro.alloc` intrinsic per coroutine.
The intrinsic is used to suppress dynamic allocation of the coroutine frame
when possible.

Example:
""""""""

.. code-block:: llvm

  entry:
    %id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
    %dyn.alloc.required = call i1 @llvm.coro.alloc(token %id)
    br i1 %dyn.alloc.required, label %coro.alloc, label %coro.begin

  coro.alloc:
    %frame.size = call i32 @llvm.coro.size()
    %alloc = call i8* @MyAlloc(i32 %frame.size)
    br label %coro.begin

  coro.begin:
    %phi = phi i8* [ null, %entry ], [ %alloc, %coro.alloc ]
    %frame = call i8* @llvm.coro.begin(token %id, i8* %phi)

.. _coro.noop:

'llvm.coro.noop' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare i8* @llvm.coro.noop()

Overview:
"""""""""

The '``llvm.coro.noop``' intrinsic returns an address of the coroutine frame of
a coroutine that does nothing when resumed or destroyed.

Arguments:
""""""""""

None

Semantics:
""""""""""

This intrinsic is lowered to refer to a private constant coroutine frame. The
resume and destroy handlers for this frame are empty functions that do nothing.
Note that in different translation units llvm.coro.noop may return different pointers.

.. _coro.frame:

'llvm.coro.frame' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare i8* @llvm.coro.frame()

Overview:
"""""""""

The '``llvm.coro.frame``' intrinsic returns an address of the coroutine frame of
the enclosing coroutine.

Arguments:
""""""""""

None

Semantics:
""""""""""

This intrinsic is lowered to refer to the `coro.begin`_ instruction. This is
a frontend convenience intrinsic that makes it easier to refer to the
coroutine frame.

.. _coro.id:

'llvm.coro.id' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare token @llvm.coro.id(i32 <align>, i8* <promise>, i8* <coroaddr>, 
                                                          i8* <fnaddrs>)

Overview:
"""""""""

The '``llvm.coro.id``' intrinsic returns a token identifying a coroutine.

Arguments:
""""""""""

The first argument provides information on the alignment of the memory returned 
by the allocation function and given to `coro.begin` by the first argument. If 
this argument is 0, the memory is assumed to be aligned to 2 * sizeof(i8*).
This argument only accepts constants.

The second argument, if not `null`, designates a particular alloca instruction
to be a `coroutine promise`_.

The third argument is `null` coming out of the frontend. The CoroEarly pass sets
this argument to point to the function this coro.id belongs to. 

The fourth argument is `null` before coroutine is split, and later is replaced 
to point to a private global constant array containing function pointers to 
outlined resume and destroy parts of the coroutine.


Semantics:
""""""""""

The purpose of this intrinsic is to tie together `coro.id`, `coro.alloc` and
`coro.begin` belonging to the same coroutine to prevent optimization passes from
duplicating any of these instructions unless entire body of the coroutine is
duplicated.

A frontend should emit exactly one `coro.id` intrinsic per coroutine.

.. _coro.end:

'llvm.coro.end' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare i1 @llvm.coro.end(i8* <handle>, i1 <unwind>)

Overview:
"""""""""

The '``llvm.coro.end``' marks the point where execution of the resume part of 
the coroutine should end and control should return to the caller.


Arguments:
""""""""""

The first argument should refer to the coroutine handle of the enclosing
coroutine. A frontend is allowed to supply null as the first parameter, in this
case `coro-early` pass will replace the null with an appropriate coroutine 
handle value.

The second argument should be `true` if this coro.end is in the block that is 
part of the unwind sequence leaving the coroutine body due to an exception and 
`false` otherwise.

Semantics:
""""""""""
The purpose of this intrinsic is to allow frontends to mark the cleanup and
other code that is only relevant during the initial invocation of the coroutine
and should not be present in resume and destroy parts. 

This intrinsic is lowered when a coroutine is split into
the start, resume and destroy parts. In the start part, it is a no-op,
in resume and destroy parts, it is replaced with `ret void` instruction and
the rest of the block containing `coro.end` instruction is discarded.
In landing pads it is replaced with an appropriate instruction to unwind to 
caller. The handling of coro.end differs depending on whether the target is 
using landingpad or WinEH exception model.

For landingpad based exception model, it is expected that frontend uses the 
`coro.end`_ intrinsic as follows:

.. code-block:: llvm

    ehcleanup:
      %InResumePart = call i1 @llvm.coro.end(i8* null, i1 true)
      br i1 %InResumePart, label %eh.resume, label %cleanup.cont

    cleanup.cont:
      ; rest of the cleanup

    eh.resume:
      %exn = load i8*, i8** %exn.slot, align 8
      %sel = load i32, i32* %ehselector.slot, align 4
      %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn, 0
      %lpad.val29 = insertvalue { i8*, i32 } %lpad.val, i32 %sel, 1
      resume { i8*, i32 } %lpad.val29

The `CoroSpit` pass replaces `coro.end` with ``True`` in the resume functions,
thus leading to immediate unwind to the caller, whereas in start function it
is replaced with ``False``, thus allowing to proceed to the rest of the cleanup
code that is only needed during initial invocation of the coroutine.

For Windows Exception handling model, a frontend should attach a funclet bundle
referring to an enclosing cleanuppad as follows:

.. code-block:: llvm

    ehcleanup: 
      %tok = cleanuppad within none []
      %unused = call i1 @llvm.coro.end(i8* null, i1 true) [ "funclet"(token %tok) ]
      cleanupret from %tok unwind label %RestOfTheCleanup

The `CoroSplit` pass, if the funclet bundle is present, will insert 
``cleanupret from %tok unwind to caller`` before
the `coro.end`_ intrinsic and will remove the rest of the block.

The following table summarizes the handling of `coro.end`_ intrinsic.

+--------------------------+-------------------+-------------------------------+
|                          | In Start Function | In Resume/Destroy Functions   |
+--------------------------+-------------------+-------------------------------+
|unwind=false              | nothing           |``ret void``                   |
+------------+-------------+-------------------+-------------------------------+
|            | WinEH       | nothing           |``cleanupret unwind to caller``|
|unwind=true +-------------+-------------------+-------------------------------+
|            | Landingpad  | nothing           | nothing                       |
+------------+-------------+-------------------+-------------------------------+

.. _coro.suspend:
.. _suspend points:

'llvm.coro.suspend' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare i8 @llvm.coro.suspend(token <save>, i1 <final>)

Overview:
"""""""""

The '``llvm.coro.suspend``' marks the point where execution of the coroutine 
need to get suspended and control returned back to the caller.
Conditional branches consuming the result of this intrinsic lead to basic blocks
where coroutine should proceed when suspended (-1), resumed (0) or destroyed 
(1).

Arguments:
""""""""""

The first argument refers to a token of `coro.save` intrinsic that marks the 
point when coroutine state is prepared for suspension. If `none` token is passed,
the intrinsic behaves as if there were a `coro.save` immediately preceding
the `coro.suspend` intrinsic.

The second argument indicates whether this suspension point is `final`_.
The second argument only accepts constants. If more than one suspend point is
designated as final, the resume and destroy branches should lead to the same
basic blocks.

Example (normal suspend point):
"""""""""""""""""""""""""""""""

.. code-block:: llvm

    %0 = call i8 @llvm.coro.suspend(token none, i1 false)
    switch i8 %0, label %suspend [i8 0, label %resume
                                  i8 1, label %cleanup]

Example (final suspend point):
""""""""""""""""""""""""""""""

.. code-block:: llvm

  while.end:
    %s.final = call i8 @llvm.coro.suspend(token none, i1 true)
    switch i8 %s.final, label %suspend [i8 0, label %trap
                                        i8 1, label %cleanup]
  trap: 
    call void @llvm.trap()
    unreachable

Semantics:
""""""""""

If a coroutine that was suspended at the suspend point marked by this intrinsic
is resumed via `coro.resume`_ the control will transfer to the basic block
of the 0-case. If it is resumed via `coro.destroy`_, it will proceed to the
basic block indicated by the 1-case. To suspend, coroutine proceed to the 
default label.

If suspend intrinsic is marked as final, it can consider the `true` branch
unreachable and can perform optimizations that can take advantage of that fact.

.. _coro.save:

'llvm.coro.save' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare token @llvm.coro.save(i8* <handle>)

Overview:
"""""""""

The '``llvm.coro.save``' marks the point where a coroutine need to update its 
state to prepare for resumption to be considered suspended (and thus eligible 
for resumption). 

Arguments:
""""""""""

The first argument points to a coroutine handle of the enclosing coroutine.

Semantics:
""""""""""

Whatever coroutine state changes are required to enable resumption of
the coroutine from the corresponding suspend point should be done at the point 
of `coro.save` intrinsic.

Example:
""""""""

Separate save and suspend points are necessary when a coroutine is used to 
represent an asynchronous control flow driven by callbacks representing
completions of asynchronous operations.

In such a case, a coroutine should be ready for resumption prior to a call to 
`async_op` function that may trigger resumption of a coroutine from the same or
a different thread possibly prior to `async_op` call returning control back
to the coroutine:

.. code-block:: llvm

    %save1 = call token @llvm.coro.save(i8* %hdl)
    call void @async_op1(i8* %hdl)
    %suspend1 = call i1 @llvm.coro.suspend(token %save1, i1 false)
    switch i8 %suspend1, label %suspend [i8 0, label %resume1
                                         i8 1, label %cleanup]

.. _coro.param:

'llvm.coro.param' Intrinsic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

  declare i1 @llvm.coro.param(i8* <original>, i8* <copy>)

Overview:
"""""""""

The '``llvm.coro.param``' is used by a frontend to mark up the code used to
construct and destruct copies of the parameters. If the optimizer discovers that
a particular parameter copy is not used after any suspends, it can remove the
construction and destruction of the copy by replacing corresponding coro.param
with `i1 false` and replacing any use of the `copy` with the `original`.

Arguments:
""""""""""

The first argument points to an `alloca` storing the value of a parameter to a 
coroutine. 

The second argument points to an `alloca` storing the value of the copy of that
parameter.

Semantics:
""""""""""

The optimizer is free to always replace this intrinsic with `i1 true`.

The optimizer is also allowed to replace it with `i1 false` provided that the 
parameter copy is only used prior to control flow reaching any of the suspend
points. The code that would be DCE'd if the `coro.param` is replaced with 
`i1 false` is not considered to be a use of the parameter copy.

The frontend can emit this intrinsic if its language rules allow for this 
optimization.

Example:
""""""""
Consider the following example. A coroutine takes two parameters `a` and `b`
that has a destructor and a move constructor.

.. code-block:: c++

  struct A { ~A(); A(A&&); bool foo(); void bar(); };

  task<int> f(A a, A b) {
    if (a.foo())
      return 42;

    a.bar();
    co_await read_async(); // introduces suspend point
    b.bar();
  }

Note that, uses of `b` is used after a suspend point and thus must be copied
into a coroutine frame, whereas `a` does not have to, since it never used 
after suspend.

A frontend can create parameter copies for `a` and `b` as follows:

.. code-block:: text

  task<int> f(A a', A b') {
    a = alloca A;
    b = alloca A;
    // move parameters to its copies
    if (coro.param(a', a)) A::A(a, A&& a');
    if (coro.param(b', b)) A::A(b, A&& b');
    ...
    // destroy parameters copies
    if (coro.param(a', a)) A::~A(a);
    if (coro.param(b', b)) A::~A(b);
  }

The optimizer can replace coro.param(a',a) with `i1 false` and replace all uses
of `a` with `a'`, since it is not used after suspend.

The optimizer must replace coro.param(b', b) with `i1 true`, since `b` is used
after suspend and therefore, it has to reside in the coroutine frame.

Coroutine Transformation Passes
===============================
CoroEarly
---------
The pass CoroEarly lowers coroutine intrinsics that hide the details of the
structure of the coroutine frame, but, otherwise not needed to be preserved to
help later coroutine passes. This pass lowers `coro.frame`_, `coro.done`_, 
and `coro.promise`_ intrinsics.

.. _CoroSplit:

CoroSplit
---------
The pass CoroSplit buides coroutine frame and outlines resume and destroy parts 
into separate functions.

CoroElide
---------
The pass CoroElide examines if the inlined coroutine is eligible for heap 
allocation elision optimization. If so, it replaces 
`coro.begin` intrinsic with an address of a coroutine frame placed on its caller
and replaces `coro.alloc` and `coro.free` intrinsics with `false` and `null`
respectively to remove the deallocation code. 
This pass also replaces `coro.resume` and `coro.destroy` intrinsics with direct 
calls to resume and destroy functions for a particular coroutine where possible.

CoroCleanup
-----------
This pass runs late to lower all coroutine related intrinsics not replaced by
earlier passes.

Areas Requiring Attention
=========================
#. A coroutine frame is bigger than it could be. Adding stack packing and stack 
   coloring like optimization on the coroutine frame will result in tighter
   coroutine frames.

#. Take advantage of the lifetime intrinsics for the data that goes into the
   coroutine frame. Leave lifetime intrinsics as is for the data that stays in
   allocas.

#. The CoroElide optimization pass relies on coroutine ramp function to be
   inlined. It would be beneficial to split the ramp function further to 
   increase the chance that it will get inlined into its caller.

#. Design a convention that would make it possible to apply coroutine heap
   elision optimization across ABI boundaries.

#. Cannot handle coroutines with `inalloca` parameters (used in x86 on Windows).

#. Alignment is ignored by coro.begin and coro.free intrinsics.

#. Make required changes to make sure that coroutine optimizations work with
   LTO.

#. More tests, more tests, more tests
